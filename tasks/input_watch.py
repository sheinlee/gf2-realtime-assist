# -*- coding: utf-8 -*-
"""分清「你点开游戏看一眼」和「你在操作游戏」（ISSUES.md I-52）。

以前只看 GetLastInputInfo（系统里最近一次键鼠输入是什么时候），分不出是挪了一下鼠标还是点了东西：
游戏在前台、2.5 秒内有输入就当你在操作。借助战途中你点开游戏看进度，这次就不借了、停手等你自己放人
（9/29 ~ 10/12 的日志里 75 次「借助战途中你动了键鼠」，几乎每次都是你自己补完的）。

这里开一个后台线程，每 20ms 看一次：最近一次输入的时刻、光标位置、鼠标键 / 修饰键按没按着、游戏在不在前台、
光标在不在游戏画面（客户区）里。游戏在前台时的新输入分成：
  - 'click'：在游戏画面里按下鼠标键（按住拖也算）；
  - 'key'  ：光标没动、鼠标键没按、修饰键也没按 —— 键盘、滚轮（也可能是一次没采到按下的极快点击）；
  - 'focus'：把游戏切到前台的那一下 —— 按下时游戏还不在前台（点游戏窗口 / 任务栏），或者游戏刚到前台的同一刻；
             这一下本身就落在游戏画面里（10/12 日志里记的光标位置都在画面内），那是「点开游戏看」，不是操作；
  - 'move' ：只挪了鼠标；'mod'：Alt / Ctrl / Shift / Win（Alt+Tab 切走、切回来）；'chrome'：点标题栏拖窗口。
'click' 和 'key' 算你在操作游戏（ops）：滚轮会滚助战列表 / 缩放地图，Esc 会关面板，它接着做就会照着旧画面点错。
别的只算「在看」，宿主等你停手（见 stage_runner._sr_check_user）。

它自己的点击是 PostMessage，不走系统输入，看不到；真键鼠拖章节地图 / 滚轮（SendInput）看得到，
由宿主拿自己动作的时间窗排掉。

线程读不到（非 Windows、出错太多）、刚开 / 停过一阵（那段时间没看着）时，宿主回到老办法：有输入就当你在操作。

「按了键」以前是排除法剩下的（有新输入、光标没动、没采到按键）：轻碰鼠标（开着「提高指针精确度」时不到 1 像素）、
光标顶在屏幕边上、光标在别的窗口上滚滚轮、切走游戏那一下，都会被当成你在操作（I-64，10/13～10/14 借助战里
7 次「按了键」交还有 6 次后面没有任何操作）。现在另开一个 Raw Input 旁听线程（隐藏窗口 + RIDEV_INPUTSINK，只听不改，
不装钩子），看这一下到底是什么：键盘的键 -> 'key'；修饰键 / 组合键 -> 'mod'；媒体 / 音量 / 截图键 -> 'media'；
滚轮：光标在游戏画面里 -> 'key'，不在 -> 'away'（滚轮发给光标底下的窗口）；不到 1 像素的挪动 -> 'move'；
没采到按下的极快点击 -> 'click'。原始输入线程没开起来 / 没看到这一下时，照旧算 'key'。
"""
import ctypes
import threading
import time
from collections import deque
from ctypes import wintypes

FOCUS_PRESS_WINDOW = 0.15    # 游戏到前台之后这么久之内按下的那一下也算「切过来的那一下」（采样先看到前台变了的情况）
SAMPLE_GAP = 0.02            # 采样间隔（秒）
OPS_KEEP = 256               # 记最近这么多次操作的时刻
HEALTHY_LAG = 1.0            # 最近一次采样离现在超过这么久就当线程没在跑（这段没看着）
ERROR_GIVE_UP = 50           # 连续出错这么多次就不采了
VK_MOUSE = (0x01, 0x02, 0x04, 0x05, 0x06)   # 左 / 右 / 中 / X1 / X2
VK_MODS = (0x10, 0x11, 0x12, 0x5B, 0x5C)    # Shift / Ctrl / Alt / 左右 Win
VK_MOD_ALL = VK_MODS + (0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5)    # 原始输入里还可能是左右分开的
VK_NOT_GAME = tuple(range(0xA6, 0xB8)) + (0x2C,)             # 浏览器 / 音量 / 媒体 / 启动键、PrintScreen：不是在操作游戏
OPS = ('click', 'key')
RAW_SETTLE = 0.06            # 原始输入：这一下之后再等这么久，把同一下的几条都收齐了再分类
RAW_WAIT = 0.3               # 原始输入：这么久还没收到这一下，按老办法算 'key'
RAW_TICK_SLOP = 16           # 原始输入的消息时刻比「最近一次输入」晚这么多毫秒以内的也算同一下（GetTickCount 一步 15～16ms）
RAW_DEAF_AFTER = 20          # 连着这么多下原始输入一条都没收到：当它聋了，直接按老办法（不再每下白等 0.3 秒），收到了再恢复


def _dlls():
    """自己的 user32 / kernel32 句柄：不碰进程共用的 ctypes.windll.*（ok-script 的叠加层会改那上面的 argtypes，
    RegisterClassW / GetCursorPos 就会报 ArgumentError，I-64 复查）。"""
    return ctypes.WinDLL('user32', use_last_error=True), ctypes.WinDLL('kernel32', use_last_error=True)


class InputClassifier:
    """采样 -> 分类（纯逻辑，测试直接喂样本）。所有时刻用同一个时钟（实机是 time.time）。"""

    def __init__(self, keep=OPS_KEEP, raw=None):
        self.raw = raw                   # RawInputSink（或测试桩）：分清「光标没动的那一下」是什么；None = 老办法
        self._pending = []               # 等原始输入的那几下：(上一次 tick, 这次 tick, 输入时刻, 采样时刻, 光标在画面里)
        self._raw_used = None            # 已经算给前面那几下的原始输入，到这个消息时刻为止（同一条不算两次）
        self._raw_misses = 0             # 连着几下原始输入一条都没收到
        self.raw_deaf = None             # 当它聋了的时候，原始输入那边收到过几条（再多了就恢复）；None = 没聋
        self.notes = deque(maxlen=keep)  # (时刻, 分类, 依据)：交还时写进日志
        self.focus_t = None              # 游戏最近一次到前台的时刻
        self.ops = deque(maxlen=keep)    # (时刻, 'click' / 'key')：你在操作游戏
        self.last = {}                   # 分类 -> 最近一次的时刻（日志 / 测试用）
        self._prev = None                # (tick, pos, fg, buttons, mods)
        self._press_inside = True        # 这次按下时光标在不在游戏画面里（松开 / 拖动跟着它算）
        self._press_focus = False        # 这次按下是不是「把游戏切到前台的那一下」
        self._lock = threading.Lock()

    def reset(self):
        """停过一阵（采样断了）：上一帧不作数，免得拿很久以前的状态比。"""
        self._prev = None

    def feed(self, t, tick, idle, pos, buttons, fg, inside, mods=False):
        """一次采样。

        :param t: 现在
        :param tick: GetLastInputInfo 的 dwTime（只拿来比有没有新输入）
        :param idle: 最近一次输入是多少秒前
        :param pos: 光标位置（屏幕坐标）
        :param buttons: 鼠标键现在按着没（或者上次采样之后按过）
        :param fg: 游戏在不在前台
        :param inside: 光标在不在游戏画面里
        :param mods: Alt / Ctrl / Shift / Win 按着没
        :return: 这次新输入的分类，没有新输入 / 游戏不在前台返回 None
        """
        prev = self._prev
        self._prev = (tick, pos, fg, buttons, mods)
        self._resolve(t)
        if prev is None:
            return None
        p_tick, p_pos, p_fg, p_buttons, p_mods = prev
        if fg and not p_fg:
            self.focus_t = t
        if buttons and not p_buttons:
            # 按下的那一刻就定下来这一下算什么（之后按着拖、松开都跟着它）：
            # 按下时游戏还不在前台 / 游戏刚到前台 = 点窗口把它切过来的那一下
            self._press_inside = bool(inside)
            self._press_focus = bool(not p_fg or not fg
                                     or (self.focus_t is not None and t - self.focus_t <= FOCUS_PRESS_WINDOW))
        if tick == p_tick or not fg:
            return None
        t_in = t - max(0.0, idle or 0.0)
        if buttons or p_buttons:
            if not self._press_inside:
                kind = 'chrome'          # 标题栏 / 边框（拖窗口）
            elif self._press_focus:
                kind = 'focus'
            else:
                kind = 'click'
        elif not p_fg:
            kind = 'focus'               # 游戏刚到前台的这一下（松开 Alt、任务栏按钮弹起来……）
        elif pos != p_pos:
            kind = 'move'
        elif mods or p_mods:
            kind = 'mod'                 # 修饰键（Alt+Tab 切走之类）
        else:
            raw = self.raw
            if raw is not None and raw.healthy() and not self._deaf(raw):
                # 光标没动、没采到按键：看原始输入这一下到底是什么（收齐了再分，见 _resolve）
                with self._lock:
                    self._pending.append((p_tick, tick, t_in, t, bool(inside)))
                return 'pending'
            kind = 'key'                 # 键盘 / 滚轮（也可能是一次没采到按下的极快点击）
        self._commit(t_in, kind, '点了游戏画面' if kind == 'click' else (
            ('原始输入一直收不到，光标没动、没采到按键（按老办法算）' if self.raw_deaf is not None
             else '没开原始输入，光标没动、没采到按键') if kind == 'key' else kind))
        return kind

    def _deaf(self, raw):
        """原始输入连着 RAW_DEAF_AFTER 下一条都没收到（权限 / 桌面切换……）：当它聋了；它又收到东西了就恢复。"""
        if self.raw_deaf is None:
            return False
        if (getattr(raw, 'count', 0) or 0) > self.raw_deaf:
            self.raw_deaf = None
            self._raw_misses = 0
            return False
        return True

    def _commit(self, t_in, kind, why):
        with self._lock:
            if kind in OPS:
                self.ops.append((t_in, kind))
                self.notes.append((t_in, kind, why))
            self.last[kind] = t_in

    def _resolve(self, now):
        """等原始输入的那几下：收齐了（或者等太久了）就分类。"""
        if not self._pending:
            return
        keep = []
        raw = self.raw
        with self._lock:
            pending = list(self._pending)
        for p_tick, tick, t_in, t0, inside in pending:
            if now - t0 < RAW_SETTLE:
                keep.append((p_tick, tick, t_in, t0, inside))
                continue
            lo = p_tick & 0xFFFFFFFF
            if self._raw_used is not None and _tick_after(self._raw_used, lo):
                lo = self._raw_used                 # 前面那一下已经用掉的不再算（晚到的松开键不会再算一次）
            try:
                evs = raw.events_between(lo, (tick + RAW_TICK_SLOP) & 0xFFFFFFFF)
                last = getattr(raw, 'last_tick', None)
            except Exception:
                evs, last = [], None
            # 收到的还没追上这一下（原始输入线程慢了，只收到前面那半截）：再等等，别拿半截去分
            caught_up = last is not None and not _tick_after(tick & 0xFFFFFFFF, last)
            if (not evs or not caught_up) and now - t0 < RAW_WAIT:
                keep.append((p_tick, tick, t_in, t0, inside))
                continue
            if evs and caught_up:
                kind, why = raw_kind(evs, inside)
                self._raw_misses = 0
                self._raw_used = max((e[0] for e in evs), key=lambda x: (x - lo) & 0xFFFFFFFF)
            elif evs:
                kind, why = 'key', '原始输入跟不上，只收到前半截（按老办法算）'
            else:
                kind, why = 'key', '原始输入没看到这一下（按老办法算）'
                self._raw_misses += 1
                if self._raw_misses >= RAW_DEAF_AFTER and self.raw_deaf is None:
                    self.raw_deaf = getattr(raw, 'count', 0) or 0
            if kind == 'click' and self.focus_t is not None and t_in - self.focus_t <= FOCUS_PRESS_WINDOW:
                kind, why = 'focus', '切到前台的那一下'
            self._commit(t_in, kind, why)
        with self._lock:
            done = set(pending) - set(keep)
            self._pending = [p for p in self._pending if p not in done]

    def pending_since(self, t0):
        """还在等原始输入、没分类的那几下里，输入时刻在 t0 之后的（宿主看「你操作了没有」之前先等它们分完）。"""
        with self._lock:
            return [p[2] for p in self._pending if p[2] >= t0]

    def notes_since(self, t0):
        with self._lock:
            return [n for n in self.notes if n[0] >= t0]

    def ops_since(self, t0):
        with self._lock:
            return [(t, k) for t, k in self.ops if t >= t0]


def raw_kind(evs, inside):
    """光标没动、没采到按键的那一下，按原始输入分类 -> (分类, 依据)。evs 见 RawInputSink.events_between。

    按下的键先看（修饰键按下 = 组合键；别的键 = 按键），再看滚轮 / 鼠标键，最后才看只有松开的键 ——
    前面那一下松开的修饰键不能把这一下的滚轮盖掉（I-64 复查）。"""
    downs = [e[2] for e in evs if e[1] == 'kbd' and e[3]]
    ups = [e[2] for e in evs if e[1] == 'kbd' and not e[3]]
    if any(v in VK_MOD_ALL for v in downs):
        return 'mod', '修饰键 / 组合键'
    real = [v for v in downs if v not in VK_NOT_GAME]
    if real:
        return 'key', '按键 0x%02X' % real[0]
    if any(e[1] == 'wheel' for e in evs):
        return ('key', '滚轮') if inside else ('away', '滚轮（光标不在游戏画面里）')
    if any(e[1] == 'button' for e in evs):
        return ('click', '极快的一次点击') if inside else ('chrome', '点在游戏画面外')
    real_up = [v for v in ups if v not in VK_NOT_GAME and v not in VK_MOD_ALL]
    if real_up:
        return 'key', '松开按键 0x%02X' % real_up[0]
    if any(v in VK_MOD_ALL for v in ups):
        return 'mod', '松开修饰键'
    if downs or ups:
        return 'media', '媒体 / 音量 / 截图键 0x%02X' % (downs or ups)[0]
    if any(e[1] == 'move' for e in evs):
        return 'move', '不到 1 像素的挪动'
    return 'key', '原始输入没看到这一下（按老办法算）'


class _POINT(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long), ('right', ctypes.c_long), ('bottom', ctypes.c_long)]


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [('cbSize', ctypes.c_uint), ('dwTime', ctypes.c_uint32)]


WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RI_KEY_BREAK = 0x01
RI_MOUSE_WHEEL = 0x0400
RI_MOUSE_HWHEEL = 0x0800
RI_MOUSE_BUTTONS = 0x03FF
WS_EX_TOOLWINDOW = 0x00000080
RAW_KEEP = 1024

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class _RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [('usUsagePage', wintypes.USHORT), ('usUsage', wintypes.USHORT), ('dwFlags', wintypes.DWORD),
                ('hwndTarget', wintypes.HWND)]


class _RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [('dwType', wintypes.DWORD), ('dwSize', wintypes.DWORD), ('hDevice', wintypes.HANDLE),
                ('wParam', wintypes.WPARAM)]


class _RAWMOUSE(ctypes.Structure):
    _fields_ = [('usFlags', wintypes.USHORT), ('_pad', wintypes.USHORT), ('usButtonFlags', wintypes.USHORT),
                ('usButtonData', wintypes.USHORT), ('ulRawButtons', wintypes.ULONG), ('lLastX', wintypes.LONG),
                ('lLastY', wintypes.LONG), ('ulExtraInformation', wintypes.ULONG)]


class _RAWKEYBOARD(ctypes.Structure):
    _fields_ = [('MakeCode', wintypes.USHORT), ('Flags', wintypes.USHORT), ('Reserved', wintypes.USHORT),
                ('VKey', wintypes.USHORT), ('Message', wintypes.UINT), ('ExtraInformation', wintypes.ULONG)]


class _WNDCLASSW(ctypes.Structure):
    _fields_ = [('style', wintypes.UINT), ('lpfnWndProc', WNDPROC), ('cbClsExtra', ctypes.c_int),
                ('cbWndExtra', ctypes.c_int), ('hInstance', wintypes.HINSTANCE), ('hIcon', wintypes.HICON),
                ('hCursor', wintypes.HANDLE), ('hbrBackground', wintypes.HBRUSH), ('lpszMenuName', wintypes.LPCWSTR),
                ('lpszClassName', wintypes.LPCWSTR)]


class RawInputSink:
    """后台线程收原始输入。events_between(lo, hi)：消息时刻（GetTickCount 那套毫秒）落在 (lo, hi] 里的
    (tick, 'kbd', vk, down) / (tick, 'wheel') / (tick, 'move') / (tick, 'button')。"""

    def __init__(self):
        self.events = deque(maxlen=RAW_KEEP)
        self.ok = False                  # 注册成功、消息循环在跑
        self.failed = False
        self.last_tick = None            # 最近收到的一条的消息时刻
        self.count = 0                   # 一共收到几条（分类器看它聋了没有）
        self._u32 = None
        self._lock = threading.Lock()
        self._hwnd = None
        self._thread = None
        self._ready = threading.Event()

    def start(self, wait=1.0):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='gf2-raw-input', daemon=True)
            self._thread.start()
            self._ready.wait(wait)
        return self

    def stop(self):
        h, u32 = self._hwnd, self._u32
        if h and u32 is not None:
            try:
                u32.PostMessageW(h, WM_CLOSE, 0, 0)
            except Exception:
                pass

    def healthy(self):
        return self.ok and not self.failed

    def events_between(self, lo, hi):
        with self._lock:
            return [e for e in self.events if _tick_after(e[0], lo) and not _tick_after(e[0], hi)]

    def _record(self, ev):
        with self._lock:
            self.events.append(ev)
            self.last_tick = ev[0]
            self.count += 1

    def _run(self):
        try:
            user32, kernel32 = _dlls()
            self._u32 = user32
            user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            user32.DefWindowProcW.restype = LRESULT
            user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p,
                                               ctypes.POINTER(wintypes.UINT), wintypes.UINT]
            user32.GetRawInputData.restype = wintypes.UINT
            user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
            user32.CreateWindowExW.restype = wintypes.HWND
            user32.GetMessageTime.restype = wintypes.LONG
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            hdr_size = ctypes.sizeof(_RAWINPUTHEADER)
            buf = ctypes.create_string_buffer(256)

            def proc(hwnd, msg, wparam, lparam):
                if msg == WM_INPUT:
                    try:
                        self._on_input(user32, lparam, buf, hdr_size)
                    except Exception:
                        pass
                elif msg == WM_DESTROY:
                    user32.PostQuitMessage(0)
                    return 0
                return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
            self._proc = WNDPROC(proc)          # 留着引用，别被回收
            hinst = kernel32.GetModuleHandleW(None)
            cls = _WNDCLASSW()
            cls.lpfnWndProc = self._proc
            cls.hInstance = hinst
            cls.lpszClassName = 'gf2_raw_input_sink_%d' % id(self)
            if not user32.RegisterClassW(ctypes.byref(cls)):
                raise OSError('RegisterClassW')
            hwnd = user32.CreateWindowExW(WS_EX_TOOLWINDOW, cls.lpszClassName, None, 0, 0, 0, 0, 0, None, None, hinst, None)   # 不显示的普通窗口（INPUTSINK 要真窗口）
            if not hwnd:
                raise OSError('CreateWindowExW')
            self._hwnd = hwnd
            devs = (_RAWINPUTDEVICE * 2)()
            devs[0] = _RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd)     # 键盘
            devs[1] = _RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, hwnd)     # 鼠标
            if not user32.RegisterRawInputDevices(devs, 2, ctypes.sizeof(_RAWINPUTDEVICE)):
                raise OSError('RegisterRawInputDevices')
            self.ok = True
            self._ready.set()
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception:
            self.failed = True
        finally:
            self.ok = False
            self._ready.set()

    def _on_input(self, user32, lparam, buf, hdr_size):
        size = wintypes.UINT(ctypes.sizeof(buf))
        n = user32.GetRawInputData(lparam, RID_INPUT, buf, ctypes.byref(size), hdr_size)
        if n == 0xFFFFFFFF or n < hdr_size:
            return
        tick = user32.GetMessageTime() & 0xFFFFFFFF
        hdr = _RAWINPUTHEADER.from_buffer_copy(buf.raw[:hdr_size])
        body = buf.raw[hdr_size:n]
        if hdr.dwType == RIM_TYPEKEYBOARD and len(body) >= ctypes.sizeof(_RAWKEYBOARD):
            kb = _RAWKEYBOARD.from_buffer_copy(body[:ctypes.sizeof(_RAWKEYBOARD)])
            self._record((tick, 'kbd', int(kb.VKey), not (kb.Flags & RI_KEY_BREAK)))
        elif hdr.dwType == RIM_TYPEMOUSE and len(body) >= ctypes.sizeof(_RAWMOUSE):
            m = _RAWMOUSE.from_buffer_copy(body[:ctypes.sizeof(_RAWMOUSE)])
            if m.usButtonFlags & (RI_MOUSE_WHEEL | RI_MOUSE_HWHEEL):
                self._record((tick, 'wheel'))
            if m.usButtonFlags & RI_MOUSE_BUTTONS:
                self._record((tick, 'button'))
            if m.lLastX or m.lLastY:
                self._record((tick, 'move'))


def _tick_after(a, b):
    """GetTickCount 的毫秒（32 位会回绕）：a 是不是在 b 之后。"""
    x = (a - b) & 0xFFFFFFFF
    return 0 < x < 0x80000000


class InputWatch:
    """后台采样线程。fg_fn() -> 游戏在不在前台；hwnd_fn() -> 游戏窗口句柄（拿客户区，判光标在不在画面里）。"""

    def __init__(self, fg_fn, hwnd_fn, now_fn=time.time, use_raw=True):
        self.fg_fn, self.hwnd_fn, self.now_fn = fg_fn, hwnd_fn, now_fn
        self.use_raw = use_raw
        self.raw = None
        self.cls = InputClassifier()
        self.alive_t = None
        self.covered_from = None         # 从这一刻起一直看着（刚开、停过一阵之后重新算）
        self.failed = False
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            if self.use_raw and self.raw is None:
                try:
                    sink = RawInputSink().start()
                    if sink.healthy():
                        self.raw = self.cls.raw = sink
                except Exception:
                    self.raw = None             # 开不起来：老办法
            self._thread = threading.Thread(target=self._run, name='gf2-input-watch', daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.raw is not None:
            self.raw.stop()

    def notes_since(self, t0):
        """t0 之后算成操作的那几下的依据 [(时刻, 分类, 依据)]（交还时写进日志）。"""
        return self.cls.notes_since(t0)

    def pending_since(self, t0):
        """t0 之后还在等原始输入、没分类的输入时刻（最多等 RAW_WAIT 就分完）。"""
        return self.cls.pending_since(t0)

    def raw_state(self):
        """原始输入旁听现在怎样：'off'（没开 / 开不起来）、'deaf'（一直收不到）、'ok'。"""
        if self.raw is None or not self.raw.healthy():
            return 'off'
        return 'deaf' if self.cls.raw_deaf is not None else 'ok'

    def healthy(self):
        a = self.alive_t
        return bool(not self.failed and a is not None and self.now_fn() - a <= HEALTHY_LAG)

    def ops_since(self, t0):
        """t0 之后你操作游戏的 (时刻, 'click' / 'key')；t0 早于一直看着的起点（刚开 / 停过）返回 None —— 那段没看着。"""
        if not self.healthy() or self.covered_from is None or t0 < self.covered_from:
            return None
        return self.cls.ops_since(t0)

    def sample(self, now, tick, idle, pos, buttons, fg, inside, mods=False):
        """一次采样（_run 拿真值调；测试直接调）。"""
        if self.alive_t is None or now - self.alive_t > HEALTHY_LAG:
            self.cls.reset()               # 刚开 / 停了一阵：这段没看着，从这一刻重新算
            self.covered_from = now
        self.cls.feed(now, tick, idle, pos, buttons, fg, inside, mods)
        self.alive_t = now

    def _inside(self, user32, pt):
        try:
            hwnd = self.hwnd_fn()
            if not hwnd:
                return True                  # 拿不到窗口：按在画面里算（宁可多停）
            rc = _RECT()
            org = _POINT(0, 0)
            if not user32.GetClientRect(hwnd, ctypes.byref(rc)) or not user32.ClientToScreen(hwnd, ctypes.byref(org)):
                return True
            return (org.x <= pt.x < org.x + rc.right - rc.left) and (org.y <= pt.y < org.y + rc.bottom - rc.top)
        except Exception:
            return True

    def _run(self):
        try:
            user32, kernel32 = _dlls()
            kernel32.GetTickCount.restype = ctypes.c_uint32
            user32.GetAsyncKeyState.restype = ctypes.c_short
        except Exception:
            self.failed = True
            return
        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        pt = _POINT()
        errors = 0
        last_tick = None
        while not self._stop.is_set():
            try:
                if not user32.GetLastInputInfo(ctypes.byref(info)) or not user32.GetCursorPos(ctypes.byref(pt)):
                    raise OSError('GetLastInputInfo / GetCursorPos failed')
                idle = ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0
                # 高位：现在按着；低位：上次调用之后按过（不完全可靠，只当补充，接住极快的点击）
                buttons = any(user32.GetAsyncKeyState(vk) & 0x8001 for vk in VK_MOUSE)
                mods = any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in VK_MODS)
                fg = bool(self.fg_fn())
                tick = int(info.dwTime)
                # 光标在不在游戏画面里：按下鼠标键时看（点标题栏 / 画面外），有新输入时也看（滚轮发给光标底下的窗口）
                inside = self._inside(user32, pt) if (fg and (buttons or tick != last_tick)) else True
                last_tick = tick
                self.sample(self.now_fn(), tick, idle, (pt.x, pt.y), buttons, fg, inside, mods)
                errors = 0
            except Exception:
                errors += 1
                if errors >= ERROR_GIVE_UP:
                    self.failed = True
                    return
            time.sleep(SAMPLE_GAP)
