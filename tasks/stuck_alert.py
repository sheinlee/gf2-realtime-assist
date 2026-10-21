# -*- coding: utf-8 -*-
"""卡住提醒 + 通知分级（ISSUES.md I-23；你 10/2 说「长时间卡住需要提醒我」）。

和 support_pick.py 一样，本文件**不定义任何 BaseTask 子类**，只提供混入和纯函数
（插件版的任务扫描器给每个 .py 找第一个 BaseTask 子类注册成任务，这里没有）。

10/2 两次停手都没人知道：
  - 10:23 7-9 第 1 阶段放不上助战，停手，一直停到 11:16（54 分钟）；
  - 11:18 第 2 阶段又放不上，停手，一直停到 12:43（86 分钟）。
停手那一刻弹过一次通知，之后就再没动静。而平时每关的「接管本关 / 继续前进 / 结算 / 本关完成」也都弹通知
（一天几百条），停手那一条混在里面，看不出来。

现在分两部分：

1. 卡住提醒。画面一直没有进展、你也没碰游戏，满「卡住多久提醒我(分钟)」就弹一条「需要你处理」的通知
   （带截图）；之后每隔「之后每隔多久再提醒(分钟)」再弹一条，直到有进展或你来处理。
   - 「没有进展」：读到的字（去掉空白，太短的不要）和这一段开头那一帧基本一样（重合度 ≥ 0.6），
     而且一直是同一种界面。**数字保留**：战斗里回合数、击杀数在变，就说明还在打。
     画面变了、换了界面、你动了游戏，就从头算。
   - 它不管的页面（关卡外的主界面 / 菜单、关卡里你自己打开的菜单、关掉「自动选关」时的章节地图……），
     如果最后动的是你，就不算卡住：那是你在用游戏，或者你不玩了，不是它卡了。它管的界面照算，
     它停手等你的时候（放不上助战、卡住交还、情报拼图不够）在哪个界面都照算。
2. 通知分级。真要你处理的停手（放不上助战、情报拼图不够、点不动作战开始、卡在认不出的画面、换章停下……）
   当场就弹，标题是「需要你处理」，带截图。平时的进度默认只写日志、不弹系统通知（开关「平时的进度也弹通知」），
   提醒才不会被淹没；配了手机推送（设置 → 通知：企业微信 / Telegram / QQ 机器人 / Discord）也只推要紧的。

宿主要提供：_now、_game_foreground、_user_idle_seconds、_sr_own_end、_sr_own（StageRunnerMixin），
ok 的 log_info / log_error / notification（没有 notification 时只写日志）。
"""
import re
import time

from support_pick import STOP_EXCEPTIONS

ALERT_TITLE = '需要你处理'
STUCK_FIRST_DEFAULT = 5          # 分钟：卡住多久提醒第一次
STUCK_REPEAT_DEFAULT = 10        # 分钟：之后每隔多久再提醒
STUCK_SIMILAR = 0.6              # 和这一段开头那一帧的重合度（Jaccard）不低于它 = 还是那个画面
STUCK_USER_RECENT = 10.0         # 你这么多秒内动过游戏 = 你在跟前，从头算
# 卡住这一段开始之前多久以内的那条提示，算这次卡住的来由（停手的提示常常比「画面不再变」早一点：
# 它停手前还在原画面上点了几下地格）
STUCK_NOTE_MARGIN = 180.0

WHERE_NAMES = {'deploy': '部署界面', 'battle': '战斗中', 'settling': '结算 / 阶段切换',
               'stage_menu': '关卡里的菜单 / 面板', 'map': '关卡详情面板', 'chapter': '章节地图',
               'idle': '关卡外', 'stage': '关卡里'}


def stuck_sig(names):
    """画面的指纹：每块字去掉空白，单个字符的不要。数字保留（战斗里回合数、击杀数在变 = 还在打）。"""
    out = set()
    for n in names or ():
        s = re.sub(r'\s+', '', n or '')
        if len(s) >= 2:
            out.add(s)
    return frozenset(out)


def sig_same(a, b):
    if not a and not b:
        return True
    return len(a & b) / float(len(a | b) or 1) >= STUCK_SIMILAR


class StuckWatch:
    """一段「没有进展」的计时。纯逻辑，测试直接喂时刻和画面。"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.since = None        # 这一段从什么时候开始
        self.ref = None          # 这一段开头那一帧的指纹（和它比，不和上一帧比：慢慢漂移的也算变了）
        self.where = None        # 这一段在哪种界面
        self.sent = 0            # 这一段已经提醒了几次
        self.next_t = None       # 下一次提醒的时刻

    def restart(self, now, sig, where):
        self.since, self.ref, self.where, self.sent, self.next_t = now, sig, where, 0, None

    def feed(self, now, sig, where, user_t, tool_t, mine, first, repeat):
        """喂一帧。该提醒了返回已经卡了多少秒，否则返回 None。

        :param user_t: 你最近一次动游戏的时刻（没见过就 None）
        :param tool_t: 它自己最近一次键鼠动作结束的时刻（没有就 None）
        :param mine: 这种界面归它管。不归它管的界面上最后动的是你，就不算卡住
        :param first: 卡住多少秒提醒第一次（<= 0 = 不提醒）
        :param repeat: 之后每隔多少秒再提醒（<= 0 = 只提醒一次）
        """
        if first <= 0:
            self.clear()
            return None
        if (self.since is None or where != self.where or not sig_same(sig, self.ref)
                or (user_t is not None and now - user_t <= STUCK_USER_RECENT)):
            self.restart(now, sig, where)
            return None
        if not mine and user_t is not None and (tool_t is None or user_t >= tool_t):
            self.restart(now, sig, where)       # 你自己走到这儿的：不算它卡住
            return None
        stuck = now - self.since
        if stuck < first or (self.next_t is not None and now < self.next_t):
            return None
        if self.sent and repeat <= 0:
            return None
        self.sent += 1
        self.next_t = now + repeat if repeat > 0 else None
        return stuck


class StuckAlertMixin:
    """通知分级 + 卡住提醒。在任务的基类列表里要排在 SupportBattleMixin 前面（log_info 先经过这里）。"""

    # ---------------------------------------------------------------- 通知分级

    def log_info(self, message, *args, **kwargs):
        self._route_notify(message, kwargs)
        return super().log_info(message, *args, **kwargs)

    def log_error(self, message, *args, **kwargs):
        self._route_notify(message, kwargs)
        return super().log_error(message, *args, **kwargs)

    def _route_notify(self, message, kwargs):
        """平时的进度：开关关着（默认）就只写日志、不弹通知。"""
        if kwargs.get('notify') and not getattr(self, '_notify_routine', True):
            kwargs['notify'] = False

    def _note_notice(self, message):
        try:
            self._last_notice = (self._now(), time.time(), str(message))
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            pass

    def alert_user(self, message, exception=None):
        """真要你处理的停手：写日志，再一定弹一条「需要你处理」的通知（带截图）。记下这句话，卡住提醒里会带上它。
        别的混入（chapter_nav、stage_runner）用 getattr 找它，宿主没有（测试桩）就照旧 log_info(notify=True)。"""
        if exception is None:
            super().log_info(message)
            text = str(message)
        else:
            super().log_error(message, exception)
            detail = exception.args[0] if getattr(exception, 'args', None) else exception
            text = '%s（%s）' % (message, detail)
        self._note_notice(text)
        self._push_notice(text, ALERT_TITLE)

    def _push_notice(self, text, title):
        fn = getattr(self, 'notification', None)
        if fn is None:
            return
        try:
            fn(text, title=title, error=True, tray=True, screenshot=True)
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            if not getattr(self, '_push_failed', False):
                self._push_failed = True
                super().log_info('弹通知出错（不影响流程，之后不再提示）：%r' % (e,))

    # ---------------------------------------------------------------- 卡住提醒

    def _stuck_minutes(self, key, default):
        cfg = getattr(self, 'config', None) or {}
        try:
            return max(0.0, float(cfg.get(key, default)))
        except (TypeError, ValueError, AttributeError):
            return float(default)

    def _stuck_see_user(self, now):
        """记下你最近一次动游戏的时刻：游戏在前台、而且不是它自己点出来的。游戏在后台时你在别处的键鼠不算。"""
        try:
            if not self._game_foreground():
                return
            idle = self._user_idle_seconds()
            if idle is None or idle < 0:
                return
            t = now - idle
            if self._sr_own_end(t) is not None:
                return
            if t > (getattr(self, '_stuck_user_t', None) or -1e18):
                self._stuck_user_t = t
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return

    def _stuck_tick(self, names, where, mine=True):
        """每一轮轮询、run_stage 的每一帧调一次。出任何错都不影响流程（只说一次）。"""
        try:
            watch = getattr(self, '_stuck_watch', None)
            if watch is None:
                watch = self._stuck_watch = StuckWatch()
            now = self._now()
            self._stuck_see_user(now)
            own = getattr(self, '_sr_own', None) or ()
            tool_t = own[-1][1] if own else None
            stuck = watch.feed(now, stuck_sig(names), where, getattr(self, '_stuck_user_t', None), tool_t, mine,
                               self._stuck_minutes('卡住多久提醒我(分钟)', STUCK_FIRST_DEFAULT) * 60.0,
                               self._stuck_minutes('之后每隔多久再提醒(分钟)', STUCK_REPEAT_DEFAULT) * 60.0)
            if stuck is not None:
                self._stuck_remind(stuck, where, watch)
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            if not getattr(self, '_stuck_failed', False):
                self._stuck_failed = True
                super().log_info('卡住提醒出错（不影响流程，之后不再提示）：%r' % (e,))

    def _stuck_remind(self, stuck, where, watch):
        mins = max(1, int(round(stuck / 60.0)))
        what = WHERE_NAMES.get(where, where)
        note = getattr(self, '_last_notice', None)      # 最近一次「需要你处理」（alert_user）
        if note and note[0] >= watch.since - STUCK_NOTE_MARGIN:
            text = '已经卡了 %d 分钟（%s），你也没碰游戏。卡住前的提示（%s）：%s' % (
                mins, what, time.strftime('%H:%M', time.localtime(note[1])), note[2])
        else:
            text = '已经 %d 分钟没有进展了（%s），你也没碰游戏 —— 去看一眼吧' % (mins, what)
        super().log_info('卡住提醒（第 %d 次）：%s' % (watch.sent, text))
        self._push_notice(text, '%s：卡住 %d 分钟' % (ALERT_TITLE, mins))

    def _stuck_reset(self):
        """重新打开任务：上一次的计时、提示都不算了。"""
        self._stuck_watch = None
        self._stuck_user_t = None
        self._last_notice = None
