# -*- coding: utf-8 -*-
"""实时辅助：关卡你点，剩下的它打。

挂在「实时触发」里持续轮询游戏画面，一旦发现你进了关卡就接管：

    你点开一个关卡
      -> 它点「挑战」
      -> 跳剧情 / 跳动画
      -> 切「好友助战」、按元素筛、找到指定角色（带战力门槛）
      -> 放到空地格上
      -> 点「作战开始」
      -> 自动战斗、吃掉结算弹窗
      -> 回到地图，继续等你点下一关

和「推图(借好友助战)」的区别只有一个：**这个不自己选关卡**。上游那套按 OCR
扫地图挑关卡的逻辑在有些章节会选错，这里把选关这一步交回给你，其余完全一样。

进本之后不再调上游的 auto_battle / skip_dialogs，改由 stage_runner.run_stage 逐帧认界面
（2026-09-29：上游在认不出的画面上每 2.5 秒盲点一次右上角、最长 15 分钟，你手动接管后
还在点）。现在不盲点右上角，你动键鼠的时候它不点任何东西。
"""

import os
import re
import time

import cv2

from ok import TriggerTask

from src.core.BaseGfTask import BaseGfTask

from support_pick import (SupportBattleMixin, SupportStateError,
                          TileNotSelectedError, UserTakeoverError, STOP_EXCEPTIONS, prune_page_images)
from stage_select import StageSelectMixin, is_chapter_map, panel_goals, STAGE_CLICK_GIVE_UP
from stage_runner import (StageRunnerMixin, IN_STAGE_MARKERS, LOADING_RE, STUCK_DEBUG_DIR, text_skeleton,
                          battle_frame)
from chapter_nav import ChapterNavMixin, event_map_info, map_head_title, story_map_title
from stage_memory import chain_code
from stuck_alert import StuckAlertMixin, STUCK_FIRST_DEFAULT, STUCK_REPEAT_DEFAULT
from own_units import LineupShortError, OwnUnitsMixin

# 「退出重进换好友」把界面搞丢之后，连着这么多次就不再退关了 —— 只借人、不重进。
# 不停手，只是不再做那个会把界面弄丢的动作。
REENTER_GIVE_UP = 2

START_BATTLE = '作战开始'
# 关卡详情面板上的进入按钮：「挑战」= 要打的关，「观看」= 纯剧情关
ENTER_STAGE = ['挑战', '观看']

# ---- 判断「现在到底在哪」用的标志文字 ----
# 战斗进行中：右下角的回合按钮。看到它就说明已经在关卡里了。
IN_BATTLE = ['行动结束', '行动完成', '还有可部署']
# 结算 / 阶段切换：也算关卡内，绝不能再去点「挑战」
SETTLING = ['继续前进', '任务完成', '任务失败', '战斗失败', '对战胜利', '对战失败', '确认结算']

# ---- 卡住之后别马上又接管同一个画面 ----
# 2026-09-29 11:31:50~12:19:14：「继续前进」一直在画面上、点了没反应，上游 auto_battle
# 每点一次就递归一层，48 分钟点了 858 次（11:59 递归爆栈之后轮询又从「结算中」接管，接着点）。
# run_stage 现在对点不动的按钮连点 6 次就不点了、150 秒后交还（stalled）；但轮询下一轮又会把
# 同一个画面认成「结算中」重新接管。所以交还之后记住那一帧的文字：画面还是那样就只看不点，
# 画面变了（你处理过了）立刻恢复；最多等这么久再试一次（万一只是游戏当时没响应）。
STALL_HOLD_SECONDS = 300
STALL_HOLD_SIMILAR = 0.6       # 文字集合的重合度（Jaccard）不低于它 = 还是那个画面
# 借助战途中你操作了游戏（接手）之后：你这么久没再操作、部署数也没变，就接着借（I-52）
TAKEOVER_RESUME_IDLE = 30
# 同一段停手里最多这样自己接着借几次（之后照旧等你；画面变了 / 你放了人 / 打完一关重新算）
AUTO_RESUME_MAX = 2

# ---- 点开一看打过了：按返回键关关卡详情面板（_panel_says_cleared）----
# 上游 ok-gf2 的推图（ClearMapTask）点开不想打的关也是按返回键（Esc）关面板、留在地图上
PANEL_CLOSE_WAIT = 15.0        # 按完返回键这么久之内又看到亮着的面板 = 没关掉
PANEL_CLOSE_BACKS = 2          # 最多按几次（再多就怕按过头、退出了章节地图）


# ---- 界面记录（只记不点）----
# 10/1 你说「完成章节后退出自动选择下一章」，让我照你刚才的操作学。可轮询每 2 秒整屏 OCR 一次，却只在界面
# 「类型」变了时记一行（章节地图 / 关卡外…），读到的字不进日志、认不出的页面也不存图 —— 你去的上级页面长什么样、
# 点了哪，一点都看不出来。现在：
#   - 认不出的页面（关卡外）每遇到一个新的：日志里记一行读到的字（带大致位置，屏幕百分比），存一张图，同名 .json
#     里有全部文字和坐标；回到记过的页面只记一行「又回到」，看得出你在页面之间怎么走的；
#   - 你在操作时（或刚停手不久），章节地图 / 关卡详情面板换了新的也记，看得出你进了哪一章、点开了哪一关；
#   - 你在操作时每轮顺手看一眼光标在哪（不是精确的点击位置，只是提示），记在下一张图的记录里。
# 加载画面、几乎没字的过渡画面不记；同一个页面（文字重合度够高）不重复记；图单独限量，不挤别的现场图。
PAGE_LOG_SIMILAR = 0.6         # 和记过的页面文字骨架重合度（Jaccard）到这个数算同一页
PAGE_LOG_RECENT = 40           # 和最近多少个记过的页面比
PAGE_LOG_MIN_BOXES = 2         # 字块少于这个的过渡画面不记
PAGE_LOG_USER_WINDOW = 20.0    # 你停手不到这么久：章节地图 / 关卡详情面板换了新的也记
PAGE_LOG_GAP = 3.0             # 两次存图至少隔这么久（画面在过渡时别连存几张）
PAGE_LOG_KEEP = 60             # 界面记录的图最多留这么多张（单独限量）
PAGE_LOG_TEXTS = 16            # 日志里列出前这么多块字
PAGE_CURSOR_KEEP = 8           # 光标位置留最近几次
PAGE_CURSOR_ACTIVE = 3.0       # 你最近这么多秒内动过键鼠才看光标


def page_sig(boxes):
    """页面的「指纹」：每块字的骨架（数字换成 #，见 text_skeleton），太短的不要。"""
    out = set()
    for b in boxes or ():
        k = text_skeleton(getattr(b, 'name', None))
        if k and len(k) >= 2:
            out.add(k)
    return frozenset(out)


def page_similar(a, b):
    return len(a & b) / float(len(a | b) or 1)


def page_is_loading(boxes):
    """加载画面：有「加载中」；或者几乎没别的字、只有一个百分比。
    不能见到百分比就算 —— 章节地图、关卡详情上也常有「100%」这种字，10/1 10:19 就这样漏记了两页。"""
    names = [b.name for b in boxes]
    if any('加载中' in n for n in names):
        return True
    return len(names) <= 3 and any(LOADING_RE.search(n) for n in names)


def screen_sig(names):
    """画面的「指纹」：读到的文字去掉数字（倒计时、战力之类会跳），太短的不要。"""
    out = set()
    for n in names or ():
        n = re.sub(r'\d+', '', n or '').strip()
        if len(n) >= 2:
            out.add(n)
    return out


def same_screen(a, b):
    if not a and not b:
        return True
    return len(a & b) / float(len(a | b) or 1) >= STALL_HOLD_SIMILAR


class RealtimeAssistTask(StuckAlertMixin, OwnUnitsMixin, SupportBattleMixin, StageSelectMixin, StageRunnerMixin, ChapterNavMixin,
                         BaseGfTask, TriggerTask):
    """实时触发任务：轮询画面，见到关卡就自动打完。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '实时辅助:点关卡自动打'
        self.description = '你只点一下关卡，进本/跳剧情/借助战/部署/开打/结算全自动'
        self._init_support_state()
        self._reenter_failures = 0
        self._await_manual = False    # 已停手、等你手动接管
        self._await_resume = None
        self._notify_routine = False  # 平时的进度只写日志、不弹通知（开关「平时的进度也弹通知」，见 stuck_alert）
        self.trigger_interval = 2.0
        self.default_config.update({
            '自动选关': True,
            '自动点挑战': True,
            '记住打过的关卡': True,
            '助战角色名': '威玛西娜',
            '备选助战角色名': '可露凯',
            '助战元素筛选': 2,
            '备选助战元素筛选': 6,
            '助战最低战力': 10000,
            '找不到目标时借最强的': True,
            '选不中地格时停手等接管': True,
            '替补最低战力': 20000,
            '助战后自动部署': True,
            '缺人时从我的人形补人': True,
            '情报拼图不够时用补给补充': True,
            '助战列表翻页次数': 3,
            '找不到时重进关卡次数': 2,
            '界面等待(秒)': 1.5,
            '轮询间隔(秒)': 2.0,
            '动画时点屏幕唤出跳过': True,
            '教学时点右上角的叉': True,
            '你操作时暂停点击': True,
            '找不到地格时挪视角': True,
            '记录认不出的界面': True,
            '章节打完自动进下一章': True,
            '换章前先打另一个页签': True,
            '离开章节前领收集奖励': True,
            '卡住多久提醒我(分钟)': STUCK_FIRST_DEFAULT,
            '之后每隔多久再提醒(分钟)': STUCK_REPEAT_DEFAULT,
            '平时的进度也弹通知': False,
        })
        self.config_description.update({
            '自动选关': (
                '打完一关回到章节地图后，自动挑下一个「已解锁且没打过」的关卡点开\n'
                '判据：卡片右上角有橙色对勾 = 打过了；卡片上有挂锁 = 没解锁；都没有 = 待打\n'
                '同屏有多个待打的就选最靠左的（章节是往右推进的）\n'
                '另外记着：自己打赢过的不再打（见「记住打过的关卡」）；点了两次都没打开的不再点\n'
                '关掉的话就要你自己点关卡，它从「挑战」那一步才开始接管'
            ),
            '记住打过的关卡': (
                '开（默认）：自己打赢的关按「游戏客户端 + 活动名 + 编号」记进 configs/stage_memory.json，重开任务、重启程序也记得 ——\n'
                '两个客户端（国际服 / 国服，装在不同目录）的号各记各的\n'
                '卡片上的勾没认出来（新活动换了样式）也不会再打一遍。「继续前进」连着打的每一关都记\n'
                '读不出活动名的地图、剧情点只记这次运行。删掉那个文件 = 全忘掉\n'
                '记录里打过、卡片上却看着可打的关：这张图别的都没得打时点开面板看一眼，挑战目标没亮就照常打（比如换了账号）\n'
                '关：只记这次运行（重开任务就忘了）'
            ),
            '缺人时从我的人形补人': (
                '点「作战开始」弹「本关至少需要出击N人」时，从「我的人形」补到 N 个人再开打：\n'
                '挑没上场的里作战效能最高的；点开卡片读名字，和助战同一个角色的（含「·衍光」这种不同版本）、名字读不出的都不上\n'
                '补之前先把元素筛选切回「全部」；列表是空的（暂无可选择人形 / 下拉框成了「编队N」）、找不到空地格、凑不够：停手告诉你差几个\n'
                '关掉 = 弹这句提示就停手告诉你（以前是再点 3 次、5 分钟后重试又去借一次助战）'
            ),
            '情报拼图不够时用补给补充': (
                '点「挑战」弹「情报拼图补充」时：选中间那格（补充道具）点「确认」，补上了接着点「挑战」打\n'
                '右边那格（晶体）一下都不点；点完看选中框确实到了中间才按「确认」\n'
                '中间那格是 0 个、认不准、补过一次还弹（每关只补一次，补完 3 分钟内又弹也不补）：照旧停手告诉你\n'
                '关掉 = 弹这个框就停手，等你自己补'
            ),
            '自动点挑战': (
                '你在地图上点开关卡、右侧出现「挑战」时，自动替你点进去\n'
                '关掉的话就要你自己点「挑战」，它从部署界面才开始接管\n'
                '（部署界面的「作战开始」是独一无二的标志，认错的可能性极低；\n'
                '  而「挑战」两个字在别的界面也可能出现，介意就关掉这项）'
            ),
            '助战角色名': (
                '要借的助战角色名，必须和游戏内显示的名字一致（默认：威玛西娜）\n'
                '留空 = 不借助战，只负责进本/跳剧情/开打/结算'
            ),
            '备选助战角色名': (
                '「助战角色名」在这份好友列表里找不到（或战力不到门槛）时，改借这个角色（默认：可露凯）\n'
                '就在同一份好友列表里找（按「备选助战元素筛选」那一列），不会为了主选去退出重进；'
                '两个都没有才退出重进换一批好友\n'
                '可以填多个，用逗号隔开，按先后优先；留空 = 不用备选'
            ),
            '助战元素筛选': (
                '「好友助战」下方元素条点第几个图标（1~6，从左数），0 = 不筛选\n'
                '默认 2 = 威玛西娜那一列，点完通常只剩几张卡'
            ),
            '备选助战元素筛选': (
                '备选角色在元素条的第几列（1~6，从左数）。主选那一列里没有够强的主选时，换到这一列找备选\n'
                '默认 6 = 可露凯那一列；填 0 = 和主选在同一列里一起找（找不到再全量查找）'
            ),
            '助战最低战力': (
                '作战效能低于这个数就换下一张；整列都不达标就退出重进换一批好友\n'
                '重进机会用完后会退而求其次选最强的那张。填 0 = 不看战力'
            ),
            '选不中地格时停手等接管': (
                '开：选不中空地格就不按「作战开始」，停手提醒你接管；'
                '你手动把人放上去（顶部「部署人形」的数字涨了）它就替你点「作战开始」接着打，'
                '你自己开打或退出关卡它也会自动恢复。期间不会乱点，也不会把任务关掉\n'
                '关：少一个人照常开打（阵容不满也能打的关卡可以关掉这项）'
            ),
            '找不到目标时借最强的': (
                '重进机会用完仍找不到指定角色时，改借列表里战力最高的其他助战；'
                '关掉就空手开打'
            ),
            '替补最低战力': (
                '替补的**优先**门槛：够这个数的里面挑战力最高的；'
                '一张都不够就退而挑全场最高的（带个弱的也比不带强）'
            ),
            '助战后自动部署': '选中助战后自动点一个青色空地格把她放上场',
            '助战列表翻页次数': '好友助战列表最多往下翻几页去找人',
            '找不到时重进关卡次数': (
                '列表里没有够强的她时，退出本关重进刷新列表，最多试几次。0 = 关闭\n'
                '只对一关的第一个部署界面生效，「继续前进」之后的阶段不会退关'
            ),
            '界面等待(秒)': (
                '每次点击后额外等多久，给界面动画留时间\n'
                '点太快会在助战面板还没滑到位时误触，把面板收回去。卡顿就调大'
            ),
            '轮询间隔(秒)': '多久看一次画面。调小反应更快但更耗 CPU',
            '动画时点屏幕唤出跳过': (
                '只在认不出的过场 / 对话画面（字很少）上点一下屏幕中央，点完立刻找右上角的「跳过」点掉\n'
                '战斗画面、字多的界面（菜单、你打开的界面）一律不点；连点几下没反应就停手只看\n'
                '不再盲点右上角。关掉 = 只点看得见的「跳过」，从不盲点'
            ),
            '教学时点右上角的叉': (
                '战斗里弹出教学 / 引导时：画面停着、点一下翻一页的，接着点屏幕中间往下翻（每阶段最多多点 15 下），\n'
                '右上角出现关闭的「×」就点掉它跳过教学。只点认出来的叉，从不盲点右上角；你刚动过键鼠 10 秒内不点\n'
                '关掉 = 教学画面上只点开场那 6 下，剩下的等你自己处理'
            ),
            '章节打完自动进下一章': (
                '仅实时辅助，只管「战役推进 · 往期活动」里的活动章节：章节地图上一个页签的关卡都打完了，\n'
                '就切到另一个页签（剧情 / 物资）；两个都打完，点左上角「<」回往期活动，点刚打完那张卡右边的下一张\n'
                '（完成度 100% 的跳过），进去接着自动选关。主线（剧情战役）一章打完游戏会自己退回章节列表：选它下面那一章\n'
                '（解锁了的），点「确认」进去接着选关。哪一步认不准就停手提醒；你一动键鼠就停。关掉 = 打完就停在地图 / 列表上'
            ),
            '换章前先打另一个页签': '「章节打完自动进下一章」开着时：一个页签打完先切到另一个页签打完再换章。关掉 = 一个页签打完就换章',
            '离开章节前领收集奖励': ('「章节打完自动进下一章」开着时：离开前左下角收集数显示「可领取」，先点开把「领取」都点掉\n'
                           '往期活动和主线（剧情战役）都领；主线在地图上「没有可推的」时就领（剩下的锁着也领）'),
            '记录认不出的界面': (
                '只记不点：认不出的页面（关卡外，比如活动 / 章节选择页）每遇到一个新的，日志里记下读到的字和大致位置，\n'
                '存一张图到 debug_tiles（单独限量 60 张）；你在操作时，换到的新章节地图 / 关卡详情面板也记，\n'
                '还顺手记光标大概在哪。给「照你的操作学」用。关掉 = 不记'
            ),
            '你操作时暂停点击': (
                '游戏在前台、且你刚动过键鼠的 2.5 秒内，它不点任何东西（包括作战开始、借助战、跳过、确认）\n'
                '你停手 2.5 秒后自动接着来。关掉 = 不管你在不在操作都照常点\n'
                '借助战 / 换章途中：只是点开游戏看、挪鼠标 -> 先停一下，你停手或切走就接着；\n'
                '你操作了游戏（在游戏画面里点了鼠标、按了键 / 滚了滚轮）-> 放弃这次借人、不按作战开始，停手等你'
                '（你放好人它替你开打；你自己开打或退出它从新画面接着来；你 30 秒没操作、部署人数没变，它接着借）'
            ),
            '找不到地格时挪视角': (
                '当前画面里找不到、或点不中空地格时，滚轮缩放（滚轮能用时）/ WASD 平移镜头再找\n'
                '一次最多点 24 下；两个方向都试，只记住真能挪动的方式\n'
                '关掉的话只在当前画面里找，找不到就改成先选人、靠空地格闪烁再找一次'
            ),
            '卡住多久提醒我(分钟)': (
                '画面一直没有进展、你也没碰游戏，满这么久就弹一条「需要你处理」的通知（带截图）。0 = 不提醒\n'
                '它管的界面（关卡里、章节地图…）和它停手等你的时候都算；你自己走到的关卡外页面（主界面、菜单）不算\n'
                '配了手机推送（设置 → 通知：企业微信 / Telegram / QQ 机器人 / Discord）也会推过去'
            ),
            '之后每隔多久再提醒(分钟)': '第一次提醒之后还是卡着，每隔这么久再提醒一次，直到有进展或你来处理。0 = 只提醒一次',
            '平时的进度也弹通知': (
                '关（默认）：只有要你处理的事（停手、卡住）才弹系统通知；接管本关、继续前进、结算、本关完成这些平时的进度\n'
                '只写日志 —— 免得提醒被一大串通知淹没，配了手机推送也只推要紧的\n'
                '开：和以前一样，每一步都弹'
            ),
        })
        self.config_type.update({
            '助战元素筛选': {'min': 0, 'max': 6},
            '备选助战元素筛选': {'min': 0, 'max': 6},
            '助战最低战力': {'min': 0, 'max': 999999},
            '替补最低战力': {'min': 0, 'max': 999999},
            '找不到时重进关卡次数': {'min': 0, 'max': 10},
            '卡住多久提醒我(分钟)': {'min': 0, 'max': 120},
            '之后每隔多久再提醒(分钟)': {'min': 0, 'max': 240},
        })

    # ------------------------------------------------------------------ 配置

    def _int_config(self, key, default):
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _float_config(self, key, default):
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def support_options(self):
        """交给 SupportBattleMixin 用的 borrow_support 参数。"""
        return {
            'target_name': (self.config.get('助战角色名') or '').strip(),
            'backup_names': self.config.get('备选助战角色名') or '',
            'max_pages': self._int_config('助战列表翻页次数', 3),
            'deploy': bool(self.config.get('助战后自动部署', True)),
            'element_index': self._int_config('助战元素筛选', 2),
            'backup_element_index': self._int_config('备选助战元素筛选', 6),
            'min_power': self._int_config('助战最低战力', 10000),
            'fallback_strongest': bool(self.config.get('找不到目标时借最强的', True)),
            'fallback_min_power': self._int_config('替补最低战力', 20000),
            'stop_on_no_tile': bool(self.config.get('选不中地格时停手等接管', True)),
            # 退出重进连着失败过几次就别再退了 —— 那个动作本身会把界面弄丢
            'reenter_times': (0 if self._reenter_failures >= REENTER_GIVE_UP
                              else self._int_config('找不到时重进关卡次数', 2)),
            'ui_delay': self._float_config('界面等待(秒)', 1.5),
        }

    # ------------------------------------------------------------------ 轮询

    def detect_state(self, boxes):
        """从一帧 OCR 结果判断现在在哪一屏。

        顺序要紧：**关卡内的几种状态都排在「地图」前面**。断线重连后游戏经常直接
        回到战斗里，这时画面上别处也可能有「挑战」两个字，先判关卡内就不会误点。

        :return: 'deploy' 部署界面 / 'battle' 战斗中 / 'settling' 结算或阶段切换
                 / 'stage_menu' 关卡里你在用的菜单或面板（不动它）
                 / 'map' 关卡详情面板（可以点进去）/ 'idle' 其它（主界面、加载、重连…）
        """
        if self.find_boxes(boxes, match=START_BATTLE, boundary=self.box.bottom):
            return 'deploy'
        if self.find_boxes(boxes, match=IN_BATTLE):
            return 'battle'
        if self.find_boxes(boxes, match=SETTLING):
            return 'settling'
        # ☰「关卡信息」/ 挡住了作战开始的部署面板：顶上有关卡编号，不拦的话会被认成章节地图、
        # 在你的菜单上自动选关。run_stage 现在会提前交还（stalled / left），轮询能看到这些画面了
        if self.find_boxes(boxes, match=IN_STAGE_MARKERS):
            return 'stage_menu'
        # 自动战斗的 HUD 上没有 IN_BATTLE 那几个词，只有左上角的回合计数：你接手开打之后，整场战斗、战后的剧情对话都曾
        # 被认成「关卡外」，没人点「跳过」（10/02 三次共等了你约 24 分钟，I-24）。排在 stage_menu 后面：战斗里打开 ☰ 时
        # 背后也有回合数，那时不能接管
        if battle_frame(boxes, self.width, self.height):
            return 'battle'
        if self.find_boxes(boxes, match=ENTER_STAGE, boundary=self.box.right):
            return 'map'
        # 章节地图：画面上有一串「AA-2-4」这样的关卡编号。排在最后 ——
        # 部署界面顶上也可能显示当前关卡编号，那种情况必须先被上面几条认走。
        if is_chapter_map(boxes, self.width, self.height):
            return 'chapter'
        return 'idle'

    def _log_state(self, state):
        """状态变了才打一行，免得每 2 秒刷一条。"""
        if state != getattr(self, '_last_state', None):
            self._last_state = state
            self.log_info('当前界面：%s' % {
                'deploy': '部署界面', 'battle': '战斗中', 'settling': '结算/阶段切换',
                'stage_menu': '关卡里的菜单/面板（不动它）',
                'map': '关卡详情面板', 'chapter': '章节地图',
                'idle': '关卡外（主界面/加载/重连…）'}[state])

    def run(self):
        self.trigger_interval = max(0.5, self._float_config('轮询间隔(秒)', 2.0))
        self._ui_delay = self._float_config('界面等待(秒)', 1.5)
        self._tap_to_reveal_skip = bool(self.config.get('动画时点屏幕唤出跳过', True))
        self._close_tutorial_x = bool(self.config.get('教学时点右上角的叉', True))
        self._record_pages = bool(self.config.get('记录认不出的界面', True))
        self._auto_next_chapter = bool(self.config.get('章节打完自动进下一章', True))
        self._switch_tabs_first = bool(self.config.get('换章前先打另一个页签', True))
        self._claim_rewards_first = bool(self.config.get('离开章节前领收集奖励', True))
        self._yield_to_user = bool(self.config.get('你操作时暂停点击', True))
        self._input_watch()                # 分「操作游戏」和「只是看看」的线程，先开起来（之前的输入它没看着，I-52）
        self._move_view_for_tile = bool(self.config.get('找不到地格时挪视角', True))
        self._notify_routine = bool(self.config.get('平时的进度也弹通知', False))
        self._refill_stamina = bool(self.config.get('情报拼图不够时用补给补充', True))
        self._sr_current = None           # 上一关 run_stage 抛异常出来时没清掉的「当前这关」
        self.next_frame()

        # 全屏只 OCR 一次，所有判断都在这同一帧上做 —— 否则两次 OCR 之间画面变了，
        # 会出现「按旧画面判、对新画面点」这种最难查的错。
        boxes = self.ocr()
        state = self.detect_state(boxes)
        self._log_state(state)
        # 卡住提醒（stuck_alert）：放在所有「这一轮不动」的提前返回之前，停手等你的时候也照样计时
        self._stuck_tick([b.name for b in boxes if getattr(b, 'name', None)], state, self._stuck_mine(state))
        self._track_stage_pick(state)
        self._page_log(state, boxes)       # 只记不点（见 PAGE_LOG_SIMILAR 上面那段）
        # 领奖励领到一半你接手了：连着两帧既没有收集数面板、也没有「获得道具」就忘掉（只看不点，放在「你在动」的关口前面 ——
        # 不然你一路动着鼠标离开地图，它就一直记着，之后把你自己的弹窗当成它的去点，I-63 复查）
        self._claim_forget_tick(boxes)
        if state in ('chapter', 'map'):
            self._stage_phase = None      # 回到关卡外了，下一关从第 1 阶段算

        # 刚因为卡住交还过：还是那个画面就只看不点（见 STALL_HOLD_SECONDS）
        if self._held_by_stall(state, boxes):
            return False

        # 补人时场上可能多了个没核对 / 不能上的人（own_units 的 risky）：这一关往下走之前（你开打 / 退出）不碰部署界面、
        # 不替你开打。认不出的画面（黑屏、弹窗、你开的别的页面）、关掉再打开任务都不算往下走了 ——
        # 不然下一轮一接管部署界面就会带着那个人点「作战开始」（10/11 第二轮复查）
        if self._held_by_risky(state):
            return False

        # 你正在动键鼠（游戏在前台）：这一轮只看不点，别跟你抢。
        # 'idle'（主界面 / 加载…）/ 'stage_menu' 这一轮本来就什么都不点，不用判断（免得你自己玩时刷「让手」日志）
        if state not in ('idle', 'stage_menu') and (self.user_busy() or self._sr_unsure):
            return False

        # 停手等接管：自己不再碰界面，但**盯着「部署人形 x/y」** ——
        # 你手动把人放上去，它就替你按「作战开始」接着往下打。
        # 你要是直接退出关卡或自己开了打，界面一变也会自动恢复。
        if getattr(self, '_await_manual', False):
            if state == 'deploy':
                return self._wait_manual_deploy()
            if state == 'stage_menu':
                return False               # 你在部署界面开着面板摆人 / 开着菜单：接着等
            if state == 'idle' and getattr(self, '_await_need', None):
                # 「至少需要出击N人」等你放人：一帧认不出的画面（黑屏 / 你开的人形页面）不算你接管完了 ——
                # 不然下一轮一看到部署界面就重新借助战、再点「作战开始」（10/11 复查）
                return False
            self._await_manual = False
            self._await_resume = None
            self._auto_resumes = 0
            self._await_need = None
            self.log_info('界面已经变了，当作你接管完毕，恢复自动', notify=True)

        # 换章领收集奖励领到一半你接手了：面板 / 「获得道具」是它自己点开的，你停手 30 秒就接着领完（I-63）
        if state in ('idle', 'chapter') and self.config.get('自动选关', True):
            resumed = self._maybe_resume_claim(boxes)
            if resumed is not None:
                return resumed

        # 1) 部署（选人）界面 —— 「作战开始」只会出现在这里，是最可靠的接管信号。
        #    每次接管都把上一轮的状态清掉，从零开始重新选人。
        if state == 'deploy':
            self._reset_stage_state()
            return self._clear_stage()

        # 2) 已经在关卡里（多半是断线重连回来的）：直接从当前进度接着打完，
        #    **不再做任何关卡外的识别和点击**
        if state in ('battle', 'settling'):
            return self._resume_stage(state)

        # 情报拼图（体力）不够：先不自动开关卡（10/1 11:30 挑战 -> 补充框 -> 退回 -> 挑战 循环了 8 分钟）。
        # 你自己点开关卡（刚动过键鼠、画面是关卡详情）就当你补充好了，试一次；部署 / 战斗 / 结算这些照常接
        hold = getattr(self, '_stamina_hold', None)
        if hold and state in ('chapter', 'map'):
            user_t = self._user_recent_seconds()
            if state == 'map' and user_t is not None and user_t <= 10.0:
                self._stamina_hold = None
                self.log_info('你点开了关卡：当你补充过情报拼图了，接着试')
            else:
                if not hold.get('logged'):
                    hold['logged'] = True
                    self.log_info('情报拼图不够（「%s」），先不自动开关卡 —— 你补充好之后点开关卡，我接着打' % hold['what'],
                                  notify=True)
                return False

        # 主线一章打完，游戏自己退回「剧情战役」章节列表：选下一章、点「确认」进去（chapter_nav；
        # 10/1 12:07 打完 5.0 最后一关，停在了列表上）。别的「关卡外」页面不动
        if state == 'idle':
            if self.config.get('自动选关', True):
                return self._maybe_story_next(boxes)
            return False

        # 3) 章节地图：自己挑一个没打过的关卡点开，下一轮就会走到 map 分支
        if state == 'chapter':
            self._note_story_map(boxes)        # 主线的话记下是哪一章（打完退回列表时对得上）
            if not self.config.get('自动选关', True):
                return False
            # 打赢的关按「活动名 + 编号」记进文件（stage_memory）：选关时拿它查，打完拿它记
            self._map_title = self._current_map_title(boxes)
            code = self.pick_next_stage(boxes)
            self._pending_pick = code
            self._pending_title = self._map_title
            if code is None:
                # 刚下了「没有可推的」：这一页打完了就切页签，两页都打完就回往期活动选下一章（chapter_nav）
                return self._maybe_next_chapter(boxes)
            return True

        # 4) 你（或上一步）在地图上点开了关卡，右侧面板出现「挑战」/「观看」
        if state != 'map':
            return False
        if not self.config.get('自动点挑战', True):
            return False
        entry = self.find_boxes(boxes, match=ENTER_STAGE, boundary=self.box.right)
        name = entry[0].name
        if '挑战' in name and self._panel_says_cleared(boxes):
            return True
        self.log_info('检测到你点开了关卡，替你点「%s」' % name, notify=True)
        # 以前固定等 2 + 界面等待（3.5 秒）。run_stage 自己会等面板滑走（没看到胜负时关卡外
        # 要连着两帧；关卡里什么都没见过时至少等 3 秒才算回到了关卡外），这里不用等那么久
        self.click_box(entry, after_sleep=1.0 + min(self.ui_delay, 0.8))
        if '观看' in name:
            return self._watch_only()
        return self._clear_stage()

    # ------------------------------------------------------------------ 界面记录（只记不点）

    def _user_recent_seconds(self):
        """你最近一次键鼠输入是多少秒前（游戏在前台、而且不是它自己点出来的）；不在前台 / 读不到返回 None。
        和 user_busy 一样的判据，但不打「你在操作」那几行日志（关卡外这些画面上它本来就不调 user_busy）。"""
        try:
            if not self._game_foreground():
                return None
            idle = self._user_idle_seconds()
            if idle is None or idle < 0:
                return None
            if self._sr_own_end(self._now() - idle) is not None:
                return None                # 那是它自己的动作
            return float(idle)
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return None

    def _cursor_in_frame(self):
        """光标现在在游戏画面上的哪（画面坐标），不在画面上 / 读不到返回 None。只读，不动光标。"""
        try:
            import win32api
            it = getattr(getattr(self, 'executor', None), 'interaction', None)
            cap = getattr(it, 'capture', None)
            if cap is None or not hasattr(cap, 'get_abs_cords'):
                return None
            W, H = self.width, self.height
            x0, y0 = cap.get_abs_cords(0, 0)
            x1, y1 = cap.get_abs_cords(W, H)
            if x1 == x0 or y1 == y0:
                return None
            cx, cy = win32api.GetCursorPos()
            fx = (cx - x0) * W / float(x1 - x0)
            fy = (cy - y0) * H / float(y1 - y0)
            if 0 <= fx < W and 0 <= fy < H:
                return int(fx), int(fy)
            return None
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return None

    def _page_log(self, state, boxes):
        """认不出的页面 / 你操作时换到的新页面：记下读到的字、存一张图。只记不点，出任何错都不影响轮询。"""
        try:
            if not getattr(self, '_record_pages', True):
                return
            now = self._now()
            user_t = self._user_recent_seconds()
            if user_t is not None and user_t <= PAGE_CURSOR_ACTIVE:
                pos = self._cursor_in_frame()
                if pos is not None:
                    trail = getattr(self, '_page_cursor', None)
                    if trail is None:
                        trail = self._page_cursor = []
                    trail.append((now, pos[0], pos[1]))
                    del trail[:-PAGE_CURSOR_KEEP]
            if state == 'idle':
                why = 'idle'
            elif state in ('chapter', 'map') and user_t is not None and user_t <= PAGE_LOG_USER_WINDOW:
                why = 'user'
            else:
                self._page_last = None
                return
            named = [b for b in boxes if getattr(b, 'name', None)]
            if len(named) < PAGE_LOG_MIN_BOXES or page_is_loading(named):
                return
            sig = page_sig(named)
            if not sig:
                return
            last = getattr(self, '_page_last', None)
            if last is not None and last[0] == state and page_similar(sig, last[1]) >= PAGE_LOG_SIMILAR:
                return                     # 还是上一轮那页
            seen = getattr(self, '_page_seen', None)
            if seen is None:
                seen = self._page_seen = []
            old = [p for p in seen if p['state'] == state and page_similar(sig, p['sig']) >= PAGE_LOG_SIMILAR]
            if old:
                self._page_last = (state, sig)
                self.log_info('界面：又回到记过的那页（%s，%s）：%s' % (
                    self._page_state_name(state), old[-1]['image'] or '没存图',
                    ' / '.join(b.name for b in named[:5])))
                return
            if now - getattr(self, '_page_saved_t', -1e9) < PAGE_LOG_GAP:
                return                     # 刚存过一张：画面多半还在过渡，下一轮再看
            self._page_last = (state, sig)
            self._page_saved_t = now
            W, H = float(self.width or 1), float(self.height or 1)
            texts = ' / '.join('%s(%d,%d)' % (b.name, round(100 * (b.x + b.width / 2.0) / W),
                                               round(100 * (b.y + b.height / 2.0) / H))
                               for b in named[:PAGE_LOG_TEXTS])
            cursor = [(round(t - now, 1), x, y) for t, x, y in (getattr(self, '_page_cursor', None) or ())
                      if now - t <= 15.0]
            path = self._save_page_image(state, why, user_t, cursor, seen[-1] if seen else None)
            seen.append(dict(state=state, sig=sig, image=os.path.basename(path) if path else None))
            del seen[:-PAGE_LOG_RECENT]
            self.log_info('界面记录（%s%s）：%s%s%s' % (
                self._page_state_name(state),
                ('，你 %.0f 秒前动过键鼠' % user_t) if user_t is not None and user_t <= PAGE_LOG_USER_WINDOW else '',
                texts,
                ('；光标（几秒前, x, y）%s' % (cursor[-3:],)) if cursor else '',
                ('；图 %s' % path) if path else ''))
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            if not getattr(self, '_page_log_failed', False):
                self._page_log_failed = True
                self.log_info('界面记录出错（不影响流程，之后不再提示）：%r' % (e,))

    @staticmethod
    def _page_state_name(state):
        return {'idle': '关卡外', 'chapter': '章节地图', 'map': '关卡详情面板'}.get(state, state)

    def _save_page_image(self, state, why, user_t, cursor, prev):
        """存界面记录的图（debug_tiles/日期/HHMMSS_page.jpg）和同名 .json；单独限量 PAGE_LOG_KEEP 张。"""
        try:
            frame = self.frame
            if frame is None:
                return None
            folder = os.path.join(STUCK_DEBUG_DIR, time.strftime('%Y%m%d'))
            os.makedirs(folder, exist_ok=True)
            stem = os.path.join(folder, time.strftime('%H%M%S'))
            path, i = stem + '_page.jpg', 2
            while os.path.exists(path):                # 同一秒里第二张：别把上一张盖掉
                path, i = '%s-%d_page.jpg' % (stem, i), i + 1
            cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(path)
            note = getattr(self, '_save_case_note', None)
            if note:
                note(path, 'page', {'state': state, 'why': why,
                                    'user_idle_seconds': None if user_t is None else round(user_t, 1),
                                    'cursor': cursor,
                                    'prev_page': None if not prev else prev.get('image')})
            prune_page_images(STUCK_DEBUG_DIR, PAGE_LOG_KEEP)
            return path
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            self.log_info('存界面记录图失败（不影响流程）：%r' % (e,))
            return None

    # ------------------------------------------------------------------ 选关的记忆

    def _track_stage_pick(self, state):
        """看上一轮在章节地图上点的那一关到底点开了没有。

        图像判据总有认错的时候（新活动换一套卡片样式就可能翻车），所以再加两道不看画面的保险：
        - 点了之后下一轮**还停在章节地图**（没弹出关卡详情）—— 多半点到了没解锁的关卡；
          连着两次就这次运行里不再点它，免得每 2 秒点一下死循环。
        - 自己点进去并且打完 / 看完的关卡记下来，就算画面上没认出通关标记也不会再打一遍。
        """
        if state == 'chapter':
            self._panel_closing = None        # 面板关掉了（见 _panel_says_cleared）
        pending = getattr(self, '_pending_pick', None)
        if pending:
            self._pending_pick = None
            if state == 'chapter':
                self.note_stage_click_failed(pending)
                self._set_active(None)
            else:
                self._set_active(pending, getattr(self, '_pending_title', None), auto=True)
        elif state == 'chapter':
            # 回到章节地图、又不是刚点完 —— 上一关的记录作废（可能是你自己退出来的）
            self._set_active(None)

    def _set_active(self, code, title=None, auto=False):
        """这次点进去的关：编号、所在的活动名、是不是它自己在地图上选的，「继续前进」连到了这一串的第几关（0 = 点进去的那关）。"""
        self._active_code = code
        self._active_title = title
        self._active_auto = bool(code) and auto
        self._chain_link = 0
        self._chain_armed = False

    def _panel_says_cleared(self, boxes):
        """它自己在地图上选的关，点「挑战」之前先看一眼关卡详情面板（通用关卡识别第 2 层，见 stage_select.panel_goals）：
        「挑战目标」的图标亮着 = 打过了（卡片上的勾没认出来）。那就不点「挑战」：记下来（进文件），按返回键关掉面板，
        下一轮回到章节地图选别的。你自己点开的关照旧替你点「挑战」—— 打过的关你也可能想再打。

        按了返回键面板还在（15 秒内又看到亮着的面板）：最多再按一次，之后不动、交给你（卡住提醒会响），
        一直等到面板没了（回到章节地图）或你动了手 —— 不能等着等着又去点「挑战」。
        :return: True = 这一轮不点「挑战」
        """
        now = self._now()
        closing = getattr(self, '_panel_closing', None)
        # 按完返回键不久，或者两次都按了还没关掉（一直等到面板没了 / 你动手）：现在看到的还是那个面板
        recent = closing is not None and (now - closing['t'] < PANEL_CLOSE_WAIT or closing['backs'] >= PANEL_CLOSE_BACKS)
        if recent:
            user_t = self._user_recent_seconds()
            if user_t is not None and user_t < now - closing['t']:
                recent, self._panel_closing = False, None     # 按完返回键你动过手：现在开着的面板是你点开的
        code = getattr(self, '_active_code', None)
        if not recent and not (code and getattr(self, '_active_auto', False)):
            return False
        r = panel_goals(self.frame, boxes, self.width, self.height)
        if r['cleared'] is not True:
            if r['cleared'] is False:
                self._panel_closing = None
                if code and getattr(self, '_pick_verify', None) == code:
                    # 记录里打过、面板上挑战目标没亮：记录不是这个号 / 这个难度的（I-53）。这次运行这关不再信记录，照常打
                    self._pick_verify = None
                    self._run_mem(getattr(self, '_active_title', None)).setdefault('ledger_wrong', set()).add(code)
                    self.log_info('「%s」记录里打过，点开一看挑战目标都没亮 —— 记录对不上这个号 / 这个难度，照常打' % code,
                                  notify=True)
            return False
        if recent:
            if closing['backs'] >= PANEL_CLOSE_BACKS:
                if not closing.get('logged'):
                    closing['logged'] = True
                    self.alert_user('「%s」打过了，按了 %d 次返回键关卡详情面板还开着 —— 先不动，你关一下面板我接着选关'
                                    % (closing['code'], closing['backs']))
                return True
            closing['backs'] += 1
            closing['t'] = now
            self.log_info('关卡详情面板还开着，再按一次返回键')
            self.back(after_sleep=0.8 + min(self.ui_delay, 0.8))
            return True
        self.log_info('「%s」点开一看已经打过了（挑战目标的图标亮着：%s），不点「挑战」，关掉面板换下一关'
                      % (code, ' / '.join('%s %.2f' % (t[:8], f) for t, f in r['rows'])))
        self.mark_stage_cleared(code, title=getattr(self, '_active_title', None), how='panel',
                                why='（关卡详情的挑战目标亮着）')
        self._set_active(None)
        self._panel_closing = dict(code=code, t=now, backs=1)
        self.back(after_sleep=0.8 + min(self.ui_delay, 0.8))
        return True

    def _current_map_title(self, boxes):
        """章节地图顶上的活动名（「镜茧与囚蝶」「琥珀月光」）/ 主线章节名（「异频共振」）；读不出返回 None（只记这次运行）。"""
        W, H = self.width, self.height
        info = event_map_info(boxes, W, H)
        if info is not None and info.get('title'):
            return info['title']
        smap = story_map_title(boxes, W, H)
        if smap:
            return smap[1]
        return map_head_title(boxes, W, H)      # 当期活动：大字是「剧情模式」、没有右下角页签

    # 「继续前进」= 结算页直接开下一关（10/02 日志 28 次全是，见 stage_memory 顶上）。stage_runner 在两个时刻调这两个钩子

    def _chain_code(self):
        """这一串里现在在打的那关的编号（点进去的那关往后数 _chain_link 关）；不是自己点进去的返回 None。"""
        return chain_code(getattr(self, '_active_code', None), getattr(self, '_chain_link', 0) or 0)

    def _chain_won(self):
        """结算页上出现「继续前进」：正在打的这关打赢了。当场记下 —— 后面卡住、你重开任务也不丢。"""
        self._chain_armed = True
        self._mark_chain_cleared()

    def _chain_started(self):
        """「继续前进」之后又看到了部署界面 / 战斗：开的是下一关。点不动的「继续前进」（下一关情报拼图不够）不会走到这里。"""
        if not getattr(self, '_chain_armed', False):
            return
        self._chain_armed = False
        self._chain_link = (getattr(self, '_chain_link', 0) or 0) + 1
        code = self._chain_code()
        if code:
            self.log_info('「继续前进」开的是下一关：按顺序是「%s」（这一串从「%s」开始的第 %d 关）'
                          % (code, self._active_code, self._chain_link + 1))

    def _mark_chain_cleared(self, how='won'):
        code = self._chain_code()
        if not code:
            return
        link = getattr(self, '_chain_link', 0) or 0
        if how and link:
            how, why = 'chain', '（「继续前进」从「%s」连过来的，编号按顺序推算）' % self._active_code
        else:
            why = ''
        self.mark_stage_cleared(code, title=getattr(self, '_active_title', None), how=how, why=why)

    def _stuck_mine(self, state):
        """这个界面上「没进展」算不算它卡住（见 stuck_alert）：它停手等你的时候在哪都算；关卡里、
        开着自动选关 / 自动点挑战时的章节地图 / 关卡详情面板归它管。关卡外、关卡里你开着的菜单不归它管 ——
        最后动的是你的话，那是你在用游戏（或者不玩了），不是它卡了。"""
        if (getattr(self, '_await_manual', False) or getattr(self, '_stall_hold', None)
                or getattr(self, '_stamina_hold', None) or getattr(self, '_risky_hold', None)):
            return True
        if state in ('deploy', 'battle', 'settling'):
            return True
        stopped = getattr(self, '_nav_stopped_t', None)
        if stopped is not None:
            # 自动换章停在它自己点出来的画面上（10/13 21:35「获得道具」弹窗，认成「关卡外」，没人提醒、停了 35 分钟，I-57）：
            # 你点了游戏画面之前都算它卡住；只是切过来看看不算你接手（I-52）
            ops = self._user_ops_since(stopped)
            touched = bool(ops) if ops is not None else (getattr(self, '_stuck_user_t', None) or -1e18) > stopped
            if not touched:
                return True
            self._nav_stopped_t = None
        if state == 'chapter':
            return bool(self.config.get('自动选关', True))
        if state == 'map':
            return bool(self.config.get('自动点挑战', True))
        return False

    def _hold_stall(self, res):
        """run_stage 卡住交还了：记下最后那一帧的文字，轮询先别重新接管同一个画面。"""
        self._stall_hold = dict(sig=screen_sig(res.get('names')), t=self._now(), logged=False)

    def _held_by_stall(self, state, boxes):
        hold = getattr(self, '_stall_hold', None)
        if not hold:
            return False
        # 关卡外（章节地图 / 关卡详情 / 主界面）一定不是卡住的那个关卡内画面
        if state not in ('chapter', 'map', 'idle'):
            same = same_screen(screen_sig(b.name for b in boxes), hold['sig'])
            waited = self._now() - hold['t']
            if same and waited < STALL_HOLD_SECONDS:
                if not hold['logged']:
                    hold['logged'] = True
                    self.log_info('还是刚才卡住的那个画面，先不动 —— 你处理一下（画面一变就接着自动；'
                                  '%d 分钟后也会再试一次）' % (STALL_HOLD_SECONDS // 60), notify=True)
                return True
            self.log_info('画面变了，接着自动' if not same else
                          '卡住的画面等了 %d 分钟，再试一次' % (STALL_HOLD_SECONDS // 60))
        self._stall_hold = None
        return False

    def _finish_active_stage(self, how='won'):
        """这一关（「继续前进」连打的话是这一串的最后一关）结束了：记下打完。how=None（剧情点）只记这次运行。"""
        self._mark_chain_cleared(how)
        self._set_active(None)
        self._auto_resumes = 0

    def enable(self):
        """重新打开任务时，上一次「停手等接管」留下的状态不能带过来 ——
        否则一打开就停在部署界面一动不动地等一个早就不相关的计数。"""
        self._await_manual = False
        self._await_resume = None
        self._auto_resumes = 0
        self._await_need = None       # 「至少需要出击N人」停手时的 N
        # _risky_hold（补人时场上可能多了个没核对的人）**不清**：关掉再打开任务不等于你处理过了场上的人
        self._handover_count = None
        for run in (getattr(self, '_run_memory', None) or {}).values():     # 每张图的（stage_select._run_mem）
            run['dead'].clear()       # 「点了没反应」的关卡：这期间你可能已经把它解锁了
            run['lost'].clear()       # 打输过的关卡：重新打开任务 = 你想再试试
        self._pending_pick = None
        self._stall_hold = None
        self._stamina_hold = None     # 你关了再开 = 你处理过了
        self._refill_t = None         # 用补给补过情报拼图的时刻（补完不久又弹就不再补，见 stage_runner._sr_refill）
        self._nav_chapters = {}       # 自动换章的记忆（哪章失败过、切过几次页签）也清掉
        self._nav_claiming = None     # 领奖励领到一半的那一章（I-63）、换章停下的时刻也不带过来
        self._nav_stopped_t = None
        self._stage_end = None        # 主线进下一章只认这次运行里自己刚打完的
        self._story_map = None
        self._panel_closing = None    # 「点开一看打过了」按返回键关面板的进度
        self._stuck_reset()           # 卡住提醒的计时、最近一条提示也清掉
        super().enable()

    def _resume_stage(self, state):
        """接手一局已经打到一半的关卡（断线重连 / 中途才打开辅助）。"""
        self.log_info('检测到已经在关卡内（%s），从当前进度接管，不再点「挑战」'
                      % ('战斗中' if state == 'battle' else '结算中'), notify=True)
        # 战斗中 / 结算中都当作已经开打：战斗画面上只点「跳过」和结算按钮，不盲点屏幕。
        # 此时不会触发借助战：_should_borrow 只在看到「作战开始」时才动手。
        return self._clear_stage(announce=False, in_battle=True)

    # ------------------------------------------------------------------ 接管

    def _wait_manual_deploy(self):
        """停手期间盯着顶部「部署人形 x/y」：涨了就说明你把人放上去了。

        涨了之后替你按「作战开始」接着打，**并跳过借助战** —— 人已经在场上了。
        读不到计数就先拿当前值当基准，下一轮再比。
        """
        now = self.read_deploy_count()
        before = getattr(self, '_handover_count', None)
        need = getattr(self, '_await_need', None)
        if before is None:
            self._handover_count = now
            # 「至少需要出击N人」停手那一刻读不到部署数，现在一读就已经够了：游戏是在不够 N 的时候弹的提示，
            # 所以够了一定是你放上去的 —— 当它从 N-1 涨上来（以前要等「再涨一次」，永远等不到）。
            # 场上可能多了个没核对的人时根本走不到这里（_risky_hold：不碰部署界面）
            if not (need and now and now[0] >= need):
                return False
            before = (now[0] - 1, now[1])
        if now and now[0] < before[0]:
            # 你撤回了一个人（比如阵容被游戏自己填满之后，先撤一个再放助战）——
            # 基准跟着往下调，之后放上去才认得出「涨了」
            self._handover_count = now
            self._await_resume = None      # 你在改阵容：不自己接着借了（借上去会少一个人开打），等你放好（I-52 复查）
            return False
        if not now or now[0] <= before[0]:
            return self._maybe_resume_borrow(now)
        if need and now[0] < need:
            return False                   # 「至少需要出击N人」：放够 N 个才开打，免得再弹一次
        self.log_info('看到你把人放上去了（部署人形 %d/%d -> %d/%d），替你点「作战开始」'
                      % (before[0], before[1], now[0], now[1]), notify=True)
        self._await_manual = False
        self._await_resume = None
        self._auto_resumes = 0
        self._await_need = None
        self._handover_count = None
        self._skip_borrow_once = True      # 你放的人，别再去借一个
        return self._clear_stage(announce=False)

    def _arm_resume(self, e):
        """借助战途中你接手（e 是关口抛的：你的键鼠）：记下停手时的部署数，之后 _maybe_resume_borrow 看能不能接着借。
        「画面已经不是部署界面了」那种不是你的键鼠，不自己重来（不然可能一直重来）；部署数读不出也不重来 ——
        读不出就分不清你放没放人。同一段停手里最多 AUTO_RESUME_MAX 次。返回 True = 会接着借。"""
        self._await_resume = None
        base = self._handover_count
        if not getattr(e, 'by_input', False) or not base:
            return False
        if (getattr(self, '_auto_resumes', 0) or 0) >= AUTO_RESUME_MAX:
            return False
        now = self._now()
        self._await_resume = dict(t=now, touch=now, base=tuple(base))
        return True

    def _maybe_resume_borrow(self, count):
        """借助战途中你接手之后（I-52）：部署数还是停手时那个、你 TAKEOVER_RESUME_IDLE 秒没操作游戏了 -> 接着借。
        以前要你自己放人、开打或者退出，不然一直停着（10/12：「被我打断后直接停止不会恢复了」）。
        部署数读不出来就不动（分不清你放没放人）。分不清「操作」和「看看」（input_watch 没开 / 那段没看着）时
        按老判据：游戏在前台时你最近一次键鼠输入（stuck_alert 每一轮都记，你在动、它让着你的那几轮也记）。"""
        r = getattr(self, '_await_resume', None)
        if not r or getattr(self, '_await_need', None):
            return False
        if not count or tuple(count) != r['base']:
            return False
        now = self._now()
        ops = self._user_ops_since(r['t'])
        if ops:
            r['touch'] = max([r['touch']] + [t for t, _ in ops])
        elif ops is None:
            seen = [getattr(self, '_stuck_user_t', None)]
            recent = self._user_recent_seconds()
            if recent is not None:
                seen.append(now - recent)
            r['touch'] = max([r['touch']] + [t for t in seen if t is not None])
        if now - r['touch'] < TAKEOVER_RESUME_IDLE:
            return False
        self._auto_resumes = (getattr(self, '_auto_resumes', 0) or 0) + 1
        self.log_info('你 %d 秒没%s了，部署人形还是 %d/%d：接着借助战（要自己来就点一下游戏画面，我会再停）'
                      % (TAKEOVER_RESUME_IDLE, '操作游戏' if ops is not None else '碰游戏', count[0], count[1]),
                      notify=True)
        self._await_manual = False
        self._await_need = None
        self._handover_count = None
        self._await_resume = None
        self._reset_stage_state()
        return self._clear_stage(announce=False)

    def _handover_at_need(self, need, cnt):
        """「至少需要出击N人」停手（场上的人都核对过）：记下停手时的部署数当基准。
        停手那一刻已经够 N 了（最后一个人刚放上去你就动了手，或者放上了却没核对出来）：基准记成 N-1，
        你一停手就替你点一次「作战开始」（不然「涨了才开打」永远等不到）。"""
        if cnt and cnt[0] >= need:
            self._handover_count = (need - 1, cnt[1])
        else:
            self._handover_count = cnt
        return cnt

    def _hold_risky(self, what):
        """补人时场上可能多了个没核对 / 不能上的人：不碰这个部署界面、不替你开打，直到这一关往下走了。"""
        self._await_manual = False
        self._await_resume = None
        self._await_need = None
        self._handover_count = None
        self._risky_hold = dict(what=what, logged=False)

    def _held_by_risky(self, state):
        hold = getattr(self, '_risky_hold', None)
        if not hold:
            return False
        if state in ('battle', 'settling', 'chapter', 'map'):
            self._risky_hold = None
            self.log_info('这一关往下走了（%s），接着自动' % {'battle': '开打了', 'settling': '结算了', 'chapter': '回到章节地图',
                                                         'map': '关卡详情面板'}[state])
            return False
        if not hold['logged'] and state in ('deploy', 'stage_menu'):
            hold['logged'] = True
            self.log_info('补人时场上可能有个没核对的人（%s），这个部署界面我不碰 —— 你看过、自己点「作战开始」之后接着自动'
                          % hold['what'])
        return True

    @staticmethod
    def _short_ask(need, cnt, risky=None):
        """停手提醒里「要你做什么」那半句。"""
        if risky:
            return ('场上可能有个没核对的人：%s —— 请你看一眼场上的人（和助战同一个角色的要撤掉），放够 %d 个之后'
                    '自己点「作战开始」；这个部署界面我先不碰' % (risky, need))
        if not cnt:
            return ('部署数读不到 —— 请你放够 %d 个人；之后读得到部署数我就替你点「作战开始」，读不到请你自己点' % need)
        if cnt[1] < need:
            return '提示要 %d 个，可部署上限读成 %d/%d —— 请你看一眼，自己点「作战开始」' % (need, cnt[0], cnt[1])
        if cnt[0] >= need:
            return '看着已经 %d/%d 了，你一停手我就替你点「作战开始」' % cnt
        return ('现在 %d/%d，还差 %d 个 —— 请你从「我的人形」再放 %d 个（放到 %d/%d），放够了我替你点「作战开始」'
                % (cnt[0], cnt[1], need - cnt[0], need - cnt[0], need, cnt[1]))

    def _short_alert(self, need, cnt, why, added=(), skipped=(), risky=None):
        """「本关至少需要出击N人」补不上时的提醒：先说要你做什么（推送被截断也看得到），再说为什么、补上了谁、谁没上。"""
        text = '本关至少需要出击 %d 人：%s。原因：%s' % (need, self._short_ask(need, cnt, risky), why)
        if added:
            text += '；我从「我的人形」补上了%s' % '、'.join('「%s」' % n for n in added)
        if skipped:
            text += '；没上：%s' % '、'.join('「%s」%s' % (n, w) for n, w in list(skipped)[:4])
        return text

    def _reset_stage_state(self):
        """每次接管一个部署（选人）界面，都当成全新的一轮来选人。

        手动介入之后尤其要紧：你把它从战斗里拉回选人界面，它就该重新走一遍
        「借助战 -> 选地格 -> 部署」，而不是拿上一轮的「已经借过了」「这几格试过了」
        当真 —— 那样它会什么都不做直接按「作战开始」。

        不重置「连续找不到地格就不再挪视角」那个闩锁：那条是为了别每场都白花几十秒
        挪视角，和这里要重来的「选人」不是一回事（它在成功选中地格时自己清零）。
        """
        self._support_done = False
        self._tried_tiles = []

    def _note_stage_phase(self):
        """run_stage 半路停下（你接手 / 选不中地格 / 出错）时记下打到了第几阶段。
        同一关接着接管时从这里接着算：第 2 阶段起不退出重进换好友（退关会把前面阶段的战果丢掉）。
        9/29 18:52：第 2 阶段你接手之后它重新接管，日志写成「部署阶段（第 1 阶段）」。"""
        cur = getattr(self, '_sr_current', None)
        if cur is not None:
            self._stage_phase = max(cur.phase, getattr(self, '_stage_phase', None) or 1)

    def _clear_stage(self, announce=True, in_battle=False):
        """从（即将进入的）部署界面一路打到结算完毕。"""
        self._nav_stopped_t = None          # 又开打了：换章那次停下已经过去了
        self._nav_claiming = None
        if announce:
            self.log_info('接管本关：跳剧情 -> 借助战 -> 部署 -> 作战开始 -> 自动战斗', notify=True)
        try:
            # run_stage 逐帧认界面，看到「作战开始」时先借助战再开打。同一关中途重新接管：接着之前的阶段算
            res = self.run_stage(kind='battle', in_battle=in_battle,
                                 phase=getattr(self, '_stage_phase', None) or 1)
        except LineupShortError as e:
            self._note_stage_phase()
            # 「本关至少需要出击N人」补不上：停手等你放人。部署数到 N 就替你点「作战开始」
            cnt = self.read_deploy_count() or e.count
            if e.risky:
                self._hold_risky(e.risky)
            else:
                self._await_manual = True
                self._await_resume = None
                self._await_need = e.need
                self._handover_at_need(e.need, cnt)
            self.alert_user(self._short_alert(e.need, cnt, e.why, e.added, e.skipped, e.risky))
            return False
        except TileNotSelectedError as e:
            self._note_stage_phase()
            # 怎么都没能把助战放上场（选不中空地格，或选好了人却部署不上）：
            # **不按「作战开始」**。硬开打就是少一个人白送一局。
            # 停手并提醒手动接管；界面一变就自动恢复。
            self._await_manual = True
            self._await_resume = None
            self._await_need = None
            self._handover_count = self.read_deploy_count()
            self.alert_user('助战没能放上场，已停手 —— 请你手动点一个空地格、选好助战、'
                            '再点右下角「部署人形」；我看到「部署人形」的数字涨了就替你点'
                            '「作战开始」接着打', e)
            return False
        except UserTakeoverError as e:
            self._note_stage_phase()
            # 借人途中你接手了（动了键鼠 / 画面已经不是部署界面）：这次不借了，也**不按「作战开始」**
            # —— 人选了一半，按下去可能少一个人。和上面一样停手等接管：你放好人（部署数涨了）就替你开打；
            # 你自己开打了 / 退出了，下一轮看到界面变了就接着自动
            self._await_manual = True
            self._await_resume = None
            self._handover_count = self.read_deploy_count()
            need = getattr(e, 'need', None)          # 「至少需要出击N人」补人途中你动了手（_sr_fill）
            self._await_need = need
            if need:
                risky = getattr(e, 'risky', None)
                cnt = self._handover_count
                if risky:
                    self._hold_risky(risky)
                else:
                    self._handover_at_need(need, cnt)
                added = getattr(e, 'added', None) or []
                self.log_info('%s，这次不补了，停手交给你 —— %s%s；%s'
                              % (e, self._short_ask(need, cnt, risky),
                                 ('；我已经补上了%s' % '、'.join('「%s」' % n for n in added)) if added else '',
                                 '你开打或退出这一关之后我接着自动' if risky else '你自己开打或退出也行，界面一变我就接着自动'),
                              notify=True)
                return False
            # 借助战途中你接手：你 TAKEOVER_RESUME_IDLE 秒没操作游戏、部署数也没变，就接着借（I-52）
            again = self._arm_resume(e)
            self.log_info('%s，这次不借了，停手交给你 —— 你放好人（部署人形的数字涨了）我就替你点'
                          '「作战开始」；你自己开打或退出也行，界面一变我就接着自动%s'
                          % (e, ('；你 %d 秒不操作游戏、部署人形还是 %d/%d，我就接着借'
                                 % ((TAKEOVER_RESUME_IDLE,) + self._await_resume['base'])) if again else ''),
                          notify=True)
            return False
        except SupportStateError as e:
            self._note_stage_phase()
            # 「退出重进换好友」把界面留在了未知状态（多半已经退回章节地图了）。
            # **不停手** —— 下一轮轮询会重新认界面：在章节地图就重新选关，
            # 在部署界面就接着打。自己关掉反而要人来救。
            self._reenter_failures += 1
            self.log_error('退出重进把界面弄丢了，不打断，下一轮重新认界面再来',
                           e, notify=True)
            if self._reenter_failures == REENTER_GIVE_UP:
                self.log_info('连着 %d 次都栽在「退出重进换好友」上，后面不再退关了 —— '
                              '在当前这批好友里找，找不到她就借战力最高的顶上'
                              % REENTER_GIVE_UP, notify=True)
            return False
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            self._note_stage_phase()
            self.alert_user('本关自动战斗失败', e)
            return False
        return self._on_stage_result(res)

    def _on_stage_result(self, res):
        result = res.get('result')
        # 主线打完一章游戏会自己退回章节列表：刚打完才接着进下一章（「挑战」没点进去的不算）
        self._stage_end = (self._now(), result if res.get('entered', True) else 'not_entered')
        # 打完 / 输了 / 回到关卡外：这一关结束了。卡住 / 超时：还在关卡里，记下打到第几阶段
        self._stage_phase = res.get('phases') if result in ('stalled', 'timeout') else None
        if result == 'victory':
            self.log_info('本关完成，回章节地图挑下一关', notify=True)
            self._finish_active_stage()
            return True
        code = self._chain_code()            # 「继续前进」连打时输的是这一串里正在打的那关，不是点进去的第一关
        if result == 'failed':
            # 这次运行里不再自动选它：打不过的关卡回到地图还是「可打」，不拦着就会一直重打
            if code:
                run = self._run_mem(getattr(self, '_active_title', None))     # 记在这关所在的那张图上（I-36）
                run['dead'][code] = STAGE_CLICK_GIVE_UP
                run['lost'].add(code)         # 章节地图的卡片清单上写「打输过」，别写成「点了没反应」
            self.log_info('本关没打过%s' % ('，这次运行里不再自动选「%s」' % code if code else ''),
                          notify=True)
            self._set_active(None)
            return False
        if result == 'left':
            if not res.get('entered', True):
                # 一直停在关卡详情面板、关卡里什么都没见过：「挑战」没点进去（点击被吞 / 面板还没反应）。
                # 下一轮轮询看到面板会再点一次；这关还是这次要打的那关，别清掉，也别说成是你退出的
                self.log_info('「挑战」好像没点进去，下一轮再试')
                return False
            self.log_info('没看到结算就回到了关卡外（多半是你手动退出了），这关不算打完', notify=True)
            self._set_active(None)
            return False
        # stalled / timeout：交还给轮询。画面变了之后下一轮重新认界面（还在关卡里会从当前进度接着打）。
        # 这是真要你处理的停手：run_stage 停下时说了为什么（note）就照那句提醒，别一律说成「认不出画面」
        self._hold_stall(res)
        self.alert_user(self._stall_message('本关', res))
        return False

    @staticmethod
    def _stall_message(what, res):
        note = res.get('note')
        if note:
            head = '%s停下了，先交还给轮询：%s' % (what, note)
        else:
            head = '%s%s，先交还给轮询' % (what, '超时' if res.get('result') == 'timeout' else '卡在认不出的画面上')
        return head + (('，现场图 %s' % res['debug']) if res.get('debug') else '')

    def _watch_only(self):
        """纯剧情关：没有战斗，跳完回地图即可。"""
        self.log_info('这是剧情关，只跳剧情不打', notify=True)
        try:
            res = self.run_stage(kind='story')
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            self.alert_user('跳剧情失败', e)
            return False
        # 剧情关没有胜负：回到关卡外就算看完。一直停在关卡外、关卡里什么都没见过
        # （「观看」没点进去）不算
        self._stage_end = (self._now(), res['result'] if res.get('entered', True) else 'not_entered')
        if res['result'] == 'victory' or (res['result'] == 'left' and res.get('entered', True)):
            self._finish_active_stage(how=None)      # 剧情点重看不花情报拼图：只记这次运行，不进文件
            return True
        if res['result'] == 'left':
            self.log_info('「观看」好像没点进去，下一轮再看')
        else:
            self._hold_stall(res)
            self.alert_user(self._stall_message('剧情关', res))
        return False
