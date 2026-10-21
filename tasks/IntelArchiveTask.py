"""从机要密报列表开始，逐条解锁并领取奖励。F8 随时停止。"""
import ctypes
import time
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ok import BaseTask
from close_x import detect_close_x
from intel_archive import (text, center, pending_cards, detail_progress, counter, counter_loose, progress_crop,
                           card_progress)

# 一下点击的手势（秒）：光标挪到目标上先停一下、按下停一下、松开再停一下，整个手势里光标都在目标上。
# ok 的 do_click 是按下 / 松开连着发、约 30ms 后就把光标挪回去。推测（没坐实）：游戏在后台时帧率低，这一下常落到挪回去的位置上 ——
# 10/14 14:08～14:20 游戏在后台时「关正文」的叉连点三次没反应、「解锁消耗」点了没出确认框，停了 5 次；14:20 切到前台之后
# 17 个项目一次没卡。可 14:07 和 14:32 那两段也在后台，用的是同一种点击，跑得好好的，所以「在后台」不是全部原因。
# 每一下都记前台 / 后台（见 _click 的日志），下次能直接对上。
TAP_HOVER = .05
TAP_HOLD = .08
TAP_AFTER = .12
POPUP_SETTLE = .6            # 「获得道具」/ 正文刚弹出来时点不到：等它弹稳再点
PROJECT_TRIES = 2            # 「没找到可解锁的条目」跳过的项目，这一轮里最多进几次（10/14「心之花」第一次跳过、第二次一进去就有，原因不明）


class IntelArchiveTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '机要密报:批量解锁'
        self.description = '在机要密报列表启动；消耗各项目专用情报道具解锁并领取奖励。F8停止。'
        self.default_config.update({'界面等待(秒)': 1.2, '最多翻页次数': 80})

    def _check_stop(self):
        if ctypes.windll.user32.GetAsyncKeyState(0x77) & 0x8001:
            raise InterruptedError('F8 停止')

    def _read(self):
        self._check_stop()
        self.next_frame()
        return self.ocr() or []

    def _pause(self):
        deadline = time.monotonic() + max(.5, float(self.config.get('界面等待(秒)', 1.2)))
        while time.monotonic() < deadline:
            self._check_stop()
            self.sleep(.1)

    def _click(self, x, y):
        self._check_stop()
        self.log_info('机要密报点击 [%s] (%d,%d) %s' % (getattr(self, '_stage', '未知'), x, y, self._fg_note()))
        self._tap(int(x), int(y))
        self._pause()

    def _fg_note(self):
        """游戏在不在前台（写进点击日志：后台点击失灵是推测，下次好对上）。"""
        try:
            win = self.executor.interaction.hwnd_window
            return '前台' if win.is_foreground() else '后台'
        except Exception:
            return ''

    def _tap(self, x, y):
        """点一下（见 TAP_HOVER 上面那段）：后台时 operate 激活窗口、挡住键鼠，手势做完才把光标挪回去。
        交互后端不支持（不是 Genshin 那套）就用框架的 click。"""
        it = getattr(getattr(self, 'executor', None), 'interaction', None)
        if it is None or not all(hasattr(it, name) for name in ('operate', 'post', 'capture')):
            return self.click(int(x), int(y))
        import win32api
        lparam = (int(y) << 16) | int(x)
        result = {'done': False, 'error': None}

        def gesture():
            pressed = False
            try:
                ax, ay = it.capture.get_abs_cords(x, y)
                win32api.SetCursorPos((int(ax), int(ay)))
                it.post(0x200, 0, lparam)          # WM_MOUSEMOVE
                time.sleep(TAP_HOVER)
                pressed = True
                it.post(0x201, 1, lparam)          # WM_LBUTTONDOWN / MK_LBUTTON
                time.sleep(TAP_HOLD)
                it.post(0x202, 0, lparam)          # WM_LBUTTONUP
                pressed = False
                time.sleep(TAP_AFTER)
                result['done'] = True
            except BaseException as exc:
                result['error'] = exc               # operate 会吞异常，带回来
            finally:
                if pressed:
                    try:
                        it.post(0x202, 0, lparam)
                    except BaseException as exc:
                        result['error'] = exc

        try:
            it.operate(gesture, block=True)
        finally:
            # ok 的 operate 收尾（还原光标、解除 BlockInput）不在 try 里，那里一抛异常物理键鼠就一直锁着（同 support_pick._operate_blocking）
            unblock = getattr(it, 'unblock_input', None)
            if unblock:
                try:
                    unblock()
                except Exception:
                    pass
        # 框架在这里看任务有没有被停（GUI 的停止按钮）：不能吞掉它的异常，不然点了停止任务还接着跑、接着消耗道具（复查）
        reset = getattr(getattr(self, 'executor', None), 'reset_scene', None)
        if reset is not None:
            reset()
        if result['error']:
            raise result['error']
        if not result['done']:
            raise RuntimeError('点击没执行成功，已停止')

    def _relative(self, x, y):
        self._click(x * self.width, y * self.height)

    def _wait(self, predicate, seconds=15, retry=None):
        deadline = time.monotonic() + seconds
        next_retry = time.monotonic() + 3
        retries = 0
        boxes = []
        while time.monotonic() < deadline:
            boxes = self._read()
            value = predicate(boxes)
            if value:
                return value
            if retry is not None and retries < 2 and time.monotonic() >= next_retry:
                point = retry(boxes)
                if point:
                    retries += 1
                    self.log_info('界面未切换，重试点击 %d/2 [%s]' % (retries, getattr(self, '_stage', '未知')))
                    self._click(*point)
                next_retry = time.monotonic() + 3
            self.sleep(.2)
        self._diagnostic(boxes)
        raise RuntimeError('等待界面超时 [%s]，已停止；详见 debug_tiles/intel' % getattr(self, '_stage', '未知'))

    def _diagnostic(self, boxes):
        """异常现场只保存游戏客户区，不依赖其他实时任务恰好在记录。"""
        try:
            folder = Path('debug_tiles/intel')
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            cv2.imencode('.png', self.frame)[1].tofile(str(stem) + '.png')
            data = {'stage': getattr(self, '_stage', '未知'),
                    'project': getattr(self, '_project_name', ''),
                    'ocr': [{'text': b.name, 'x': b.x, 'y': b.y, 'width': b.width,
                             'height': b.height} for b in boxes],
                    'counter_retry': getattr(self, '_counter_retry', [])}
            Path(str(stem) + '.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            self.log_info('机要密报异常现场：' + str(stem))
        except Exception as exc:
            self.log_info('保存机要密报现场失败：' + str(exc))

    @staticmethod
    def _list(boxes):
        return any('机要密报' in text(b) for b in boxes) and any('解锁进度' in text(b) for b in boxes)

    def _progress(self, boxes):
        # 同一屏（同一份 OCR）会被问好几次：只算一次，日志也只打一次
        cached = getattr(self, '_progress_memo', None)
        if cached is not None and cached[0] is boxes:
            return cached[1]
        value = self._progress_read(boxes)
        self._progress_memo = (boxes, value)
        return value

    def _progress_read(self, boxes):
        value = detail_progress(boxes, self.width, self.height)
        if value:
            return value
        # 必须先看到详情页的“已解锁”，避免弹窗/正文背景上的数字误判。
        if not any('已解锁' in text(b) and center(b)[0] > .85*self.width
                   and center(b)[1] < .12*self.height for b in boxes):
            return None
        self._counter_retry = []
        seen = [text(b) for b in boxes if center(b)[0] > .85*self.width and center(b)[1] < .12*self.height]
        # 2/6 在 160px 可读，4/6 在同一倍率仍被读成 416，需要再次放大。
        # 必须真正识别出分隔符，不能把中间的 1 无条件替换成 /。
        for target_height in (160, 320, 480):
            retry = self.ocr(frame=progress_crop(self.frame, target_height)) or []
            self._counter_retry.append({'height': target_height, 'text': [text(b) for b in retry]})
            seen += [text(b) for b in retry]
            values = {counter(text(b)) for b in retry if counter(text(b))}
            values = {p for p in values if 0 <= p[0] <= p[1] <= 99 and p[1] > 0}
            if len(values) == 1:
                value = values.pop()
                self.log_info('计数区域放大识别恢复(%dpx)：%d/%d' % (target_height, *value))
                return value
        # 放大了也读不出斜杠（「9/9」三档都读成「919」，10/14 14:15 停手）：只有已知这个项目一共几条
        # （列表卡片上读到的 / 这一页前面读到的）、而且只有一种拆法和它对得上，才把那个 1 当斜杠
        total = getattr(self, '_known_total', None)
        if total:
            values = {counter_loose(t) for t in seen if counter_loose(t)}
            values = {p for p in values if p[1] == total}
            if len(values) == 1:
                value = values.pop()
                self.log_info('计数读成「%s」，按这个项目一共 %d 条认作 %d/%d' % (
                    next(t for t in seen if counter_loose(t) == value), total, *value))
                return value

    def _thumb(self):
        h, w = self.frame.shape[:2]
        return cv2.resize(cv2.cvtColor(self.frame[int(h*.12):int(h*.95), int(w*.24):int(w*.87)],
                                      cv2.COLOR_BGR2GRAY), (160, 120)).astype(float)

    def _scroll(self, detail=False):
        self._check_stop()
        self._stage = '详情向下滚动' if detail else '列表向下滚动'
        before = self._thumb()
        self._drag_list(.77 if detail else .70)
        self._pause()
        self._read()
        delta = float(np.mean(abs(self._thumb() - before)))
        self.log_info('%s：左键向上拖动 82%%→40%%，画面变化 %.2f' % (self._stage, delta))
        return delta >= 1.5

    def _drag_list(self, x_ratio):
        """整个手势只激活窗口一次；不经过缺 mouse 包的 scroll() 接口。"""
        import win32api
        it = self.executor.interaction
        if not all(hasattr(it, name) for name in ('operate', 'post', 'capture')):
            raise RuntimeError('当前交互后端不支持机要密报拖动，已停止')
        x, y0, y1 = int(self.width*x_ratio), int(self.height*.82), int(self.height*.40)
        result = {'done': False, 'error': None}

        def gesture():
            y = y0
            pressed = False
            try:
                ax, ay = it.capture.get_abs_cords(x, y)
                win32api.SetCursorPos((int(ax), int(ay)))
                it.post(0x200, 0, (y << 16) | x)  # WM_MOUSEMOVE
                self._check_stop()
                pressed = True
                it.post(0x201, 1, (y << 16) | x)  # WM_LBUTTONDOWN / MK_LBUTTON
                time.sleep(.06)
                for step in range(1, 25):
                    self._check_stop()
                    y = round(y0 + (y1-y0)*step/24)
                    ax, ay = it.capture.get_abs_cords(x, y)
                    win32api.SetCursorPos((int(ax), int(ay)))
                    it.post(0x200, 1, (y << 16) | x)
                    time.sleep(.025)
                result['done'] = True
            except BaseException as exc:
                # operate 会吞异常，手动带回，不能把输入失败当作到底。
                result['error'] = exc
            finally:
                if pressed:
                    try:
                        it.post(0x202, 0, (y << 16) | x)  # WM_LBUTTONUP
                    except BaseException as exc:
                        result['error'] = exc

        it.operate(gesture)
        if result['error']:
            raise result['error']
        if not result['done']:
            raise RuntimeError('列表拖动未执行成功，已停止，不能认定已到底')

    def _back(self):
        self._stage = '返回卡片列表'
        self._relative(.028, .048)
        # 补点「<」的判据要用到总数（详情页「919」那种只靠总数才读得出）：回到列表之后再清（复查）
        self._wait(self._list, retry=lambda bs: (.028*self.width, .048*self.height) if self._progress(bs) else None)
        self._known_total = None

    def _project(self):
        """处理一个项目的详情页。返回 True（全部解锁了）/ 'short'（材料不足，跳过）/ 'none'（没找到可解锁的条目，跳过）。"""
        self._stage = '读取项目进度'
        previous = None
        still = 0
        for _ in range(100):
            boxes = self._wait(lambda bs: bs if self._progress(bs) else None)
            progress = self._progress(boxes)
            self._known_total = progress[1]
            if progress[0] == progress[1]:
                self._back()
                return True
            unlock = [b for b in boxes if '解锁消耗' in text(b) and b.x > .23 * self.width]
            if not unlock:
                thumb = self._thumb()
                still = still + 1 if previous is not None and np.mean(abs(thumb-previous)) < 1.5 else 0
                if still >= 3:
                    self.log_info('详情连续三次未移动且未发现可解锁条目，跳过；可能到底或拖动未生效')
                    self._back()
                    return 'none'
                previous = thumb
                self._scroll(detail=True)
                continue
            previous, still = None, 0
            self._stage = '点击解锁消耗，等待确认弹窗'
            self._click(*center(min(unlock, key=lambda b: b.y)))
            # 必须同时识别解锁用途和确认按钮，才允许消耗材料。
            def dialog(bs):
                joined = ''.join(text(b) for b in bs)
                if '不足' in joined:
                    return ('short', bs)
                if '消耗' in joined and '解锁' in joined and '查阅' in joined:
                    buttons = [b for b in bs if text(b) in ('确认', '确定') and
                               .5*self.width < center(b)[0] < .75*self.width and
                               .6*self.height < center(b)[1] < .8*self.height]
                    if buttons:
                        return ('confirm', buttons)
            def retry_unlock(bs):
                if dialog(bs) or not self._progress(bs):
                    return None
                buttons = [b for b in bs if '解锁消耗' in text(b) and b.x > .23*self.width]
                return center(min(buttons, key=lambda b: b.y)) if buttons else None
            kind, targets = self._wait(dialog, retry=retry_unlock)
            if kind == 'short':
                cancel = [b for b in targets if text(b) == '取消']
                if not cancel:
                    raise RuntimeError('材料不足，已停止；请手动关闭提示')
                self._click(*center(cancel[0]))
                self._wait(lambda bs: self._progress(bs))
                self._back()
                return 'short'
            self._stage = '确认解锁，等待正文'
            self._click(*center(targets[0]))
            # 仅在详情计数/解锁确认框均已消失时，识别正文右上角的叉。
            def article(bs):
                if self._progress(bs) or any(text(b) in ('确认', '取消', '获得道具') for b in bs):
                    return None
                if len(bs) < 3:
                    return None
                return detect_close_x(self.frame, band=(.92, 0, 1, .12))
            def retry_confirm(bs):
                found = dialog(bs)
                return center(found[1][0]) if found and found[0] == 'confirm' else None
            close = self._wait(article, retry=retry_confirm)[0]
            self._settle()
            self._stage = '关闭正文，等待奖励'
            self._click(close['x'], close['y'])
            def reward(bs):
                return any('获得道具' in text(b) for b in bs) and any('空白' in text(b) and '关闭' in text(b) for b in bs)
            def retry_close(bs):
                found = article(bs)
                return (found[0]['x'], found[0]['y']) if found else None
            self._wait(reward, retry=retry_close)
            self._settle()
            self._stage = '关闭奖励，核对解锁进度'
            self._relative(.5, .87)
            now = self._wait(lambda bs: self._progress(bs),
                             retry=lambda bs: (.5*self.width, .87*self.height) if reward(bs) else None)
            if now[1] != progress[1] or now[0] <= progress[0]:
                raise RuntimeError('解锁后进度未增加，已停止')
            self.log_info('解锁进度 %s/%s' % now)
        raise RuntimeError('单项目操作达到上限，已停止')

    def _settle(self):
        """弹窗刚出来：等它弹稳再点（POPUP_SETTLE），中途照样看 F8。"""
        deadline = time.monotonic() + POPUP_SETTLE
        while time.monotonic() < deadline:
            self._check_stop()
            self.sleep(.1)

    def run(self):
        self._stage = '读取机要密报列表'
        self._known_total = None
        visited = set()
        tries = {}
        completed = skipped = still = 0
        previous = None
        try:
            initial = self._wait(lambda bs: bs if self._list(bs) or self._progress(bs) else None)
            if not self._list(initial):
                self._project_name = '当前已打开的项目'
                self.log_info('从当前详情页继续解锁')
                if self._project() is True:
                    completed += 1
                else:
                    skipped += 1
            for _ in range(max(1, int(self.config.get('最多翻页次数', 80)))):
                while True:
                    boxes = self._wait(lambda bs: bs if self._list(bs) else None)
                    cards = [(name, point) for name, point in pending_cards(boxes, self.width, self.height)
                             if name not in visited]
                    if not cards:
                        break
                    name, point = cards[0]
                    self._project_name = name
                    self._stage = '进入项目：' + name
                    tries[name] = tries.get(name, 0) + 1
                    visited.add(name)
                    listed = card_progress(boxes, self.width, self.height, name)
                    self._known_total = listed[1] if listed else None
                    self.log_info('处理机要密报：' + name + ('（第 %d 次）' % tries[name] if tries[name] > 1 else ''))
                    self._click(*point)
                    def retry_card(bs):
                        if self._list(bs):
                            return next((pt for title, pt in pending_cards(bs, self.width, self.height)
                                         if title == name), None)
                    self._wait(lambda bs: self._progress(bs), retry=retry_card)
                    outcome = self._project()
                    if outcome is True:
                        completed += 1
                    elif outcome == 'none' and tries[name] < PROJECT_TRIES:
                        visited.discard(name)          # 没找到可解锁的条目：这一轮里再进一次（多半是刚进去时还没加载完）
                    else:
                        skipped += 1
                    previous, still = None, 0
                thumb = self._thumb()
                still = still + 1 if previous is not None and np.mean(abs(thumb-previous)) < 1.5 else 0
                if still >= 3:
                    self._diagnostic(boxes)
                    self.log_info('列表连续三次未移动，停止扫描（可能已到底，也可能拖动未生效）；完成 %d 项，跳过 %d 项。已保存现场供核对' % (completed, skipped))
                    return
                previous = thumb
                self._scroll()
            self.log_info('已达到翻页上限；完成 %d 项，跳过 %d 项，可从当前位置继续' % (completed, skipped))
        except InterruptedError:
            self.log_info('用户按 F8，机要密报任务已停止')
