# -*- coding: utf-8 -*-
"""章节地图上挑关卡（ok-gf2 扩展）。

和 support_pick.py 一样，本文件**不定义任何 BaseTask 子类**，只提供一个混入 ——
ok 的任务扫描器会给 ok_import 下每个 .py 找第一个 BaseTask 子类注册成任务，
这里没有，所以不会多出一个空任务。

每个活动的关卡卡片长得都不一样：

    旧章节        深青色卡片，编号下方一条青色底边
    拂晓的回响    白卡 + 青绿边框，左边一个金色「嶄」字，没有青条
    密米尔的返演  深色水墨卡，没有青条

以前的判据（青色底边 = 已解锁、暖色像素 = 已完成）只在第一种上标定过，换到后两种就全错：
一关都不算解锁，金色「嶄」字还把没打过的 ZS-1-2 抬成了「已完成」。

现在只认跨样式都不变的两个图标（见 stage_icons.py）：

    编号左边有白色挂锁        -> 未解锁
    卡片右上角有橙色圆形对勾  -> 已完成
    都没有                    -> 可打
    图标可能在屏幕外          -> 不确定（不点）

编号本身也不能直接用整屏 OCR：编号会和左边的图形并成一个框、读错、读不出（见 stage_codes.py），
所以对每个像编号的片段裁一小块原分辨率重读。

再加两道**不看画面**的保险，防止图像判据在没见过的新样式上翻车：
  - 这次运行里自己点进去打完 / 看完的关卡，不再打第二遍；
  - 点了卡片、下一轮还停在章节地图（没弹出关卡详情）的，连着两次就不再点。
"""

import ctypes
import os
import re
import time

import cv2
import numpy as np
from ok import Box

from stage_codes import _overlap_frac, extract_stage_codes, garbled_codes, has_stage_code, repair_merged, white_text
from stage_icons import classify_card, frame_ref_h
from stage_memory import StageLedger, client_key, ledger_path, norm_title, same_title

try:
    from ok import FinishedException, TaskDisabledException
except ImportError:            # 离线回放截图时 ok 是个桩模块
    class TaskDisabledException(Exception):
        pass

    class FinishedException(Exception):
        pass

# 用户在界面上点了「停止」时 ok 抛的异常（和 support_pick.STOP_EXCEPTIONS 同一对）。
# **绝不能被 except Exception 吞掉**，吞掉之后任务会在用户按了停止之后接着点。
STOP_EXCEPTIONS = (TaskDisabledException, FinishedException)

# 关卡编号，例如 AA-2-4 / AZ-2-1 / AS-2-4。**只用于判断「这是不是章节地图」**时的旧接口；
# 真正提取编号用 stage_codes.extract_stage_codes（它能从「学ZS-1-2」这种并框里认出编号）。
STAGE_CODE_RE = re.compile(r'[A-Z]{2}-\d+-\d+')

STATE_NAMES = {'completed': '已完成', 'available': '可打', 'locked': '未解锁', 'unknown': '看不全'}

# 点了同一张卡几次都没打开关卡详情，就这次运行里不再点它
STAGE_CLICK_GIVE_UP = 2
# 章节地图没变的话，多久之内直接复用上次的识别结果（整套识别约 0.5~0.7s，轮询是 2s 一次）
STAGE_CACHE_SECONDS = 10
# 一个可打的都没有时会深扫一次（约 1s）；同样的画面多久之内不重复深扫
DEEP_RETRY_SECONDS = 30

DEBUG_DIR = 'debug_tiles'
DEBUG_KEEP = 120
# 「没有可推的」存的章节地图单独限量（原图 + 标注图算两个），不和别的现场图抢那 120 个名额：
# 9/29 18:33~18:45 停在章节最左边、边上半张卡的编号每次读得不一样，每 2 秒存一对图，
# 把部署失败、点不动作战开始这些现场图全挤掉了
CHAPTER_DEBUG_KEEP = 40
# 「没有可推的」这句话 / 这张图：和上次说过的那屏比，(编号, 状态) 的重合度不低于它就算同一屏
# （边上那张半露的卡时有时无、编号认成别的，都还是同一屏）
IDLE_SAME_SCREEN = 0.5

# ---- 下一关在屏幕外：拖动章节地图去找（2026-09-29 停住的 11 次里有 4 次是这个）----
# 在卡片**下方**的空白处横向拖：卡片都在 25%~72% 高度，80% 高度那一条只有地图底图；
# 左下角的收集进度、右下角的「普通 / 困难」都避开了。就算拖动没被当成拖动、成了一次点击，
# 也只会点在空白地图上。
CHAPTER_DRAG_Y = 0.80
CHAPTER_DRAG_X = (0.72, 0.28)      # 从右往左拖 = 看右边
# 先往右最多拖几下（章节往右推进，下一关多半在右边）。以前是 3：9/30「密米尔的返演·下篇」活动地图很长，
# 每打完一关回到地图镜头都回到最左边，打到后面下一关在第 4 屏以后，往右 3 下够不着，停在那里等你（21:11、21:15）。
# 现在一直往右拖，直到：找到可打的 / 一整屏都是没解锁的（已经越过进度了）/ 拖到头（拖不动）
CHAPTER_SCROLL_RIGHT = 12
CHAPTER_SCROLL_LEFT = 3            # （旧）往左那趟：先回到起点，再往左最多看这么多下 —— 现在往左一直拖到头，见下
CHAPTER_LEFT_MIN = 6               # 往左那趟最少拖几下（往右没挪动时，起点左边也看 6 屏）
# 往左那趟以前只拖「回起点 + 3 下」（至少 6 下），10/2 17:10「错位锚定·下篇」：往右 3 下到头、往左 6 下还在动就不拖了，
# 说「没有可推的」；其实再往左还有一关支线没打（63/66），你 17:16 自己拖过去打的（I-25）。
# 现在往左和往右一样一直拖到头（拖不动才算），只设一个防呆上限
CHAPTER_SCROLL_LEFT_MAX = 16
CHAPTER_DRAG_GIVE_UP = 2           # 连着几个章节地图都拖不动，这次运行就不再试
# 往一个方向拖的第一下「画面一点没变」：拖到头时游戏一般会弹一下（「卡片位移 +0 px」），一点没变多半是这一下游戏没收到。
# 10/12 01:30「逆序矩阵」：往右 +0 px 到头，往左第一下「画面没变」—— 左边的 DF-4-1 / 4-2 一眼没看，整趟算「一下都没挪动」，
# 两头都判不出、不换章，停在那里（I-45）。等一下再拖一次，还是没变才算到头
CHAPTER_STILL_RETRY_WAIT = 1.0
# 章节地图上挡着的弹窗（章节收集奖励）：这时候别拖别点
POPUP_RE = re.compile('收集数')

# ---- 用什么让章节地图动起来 ----
# 历史日志里，后台 PostMessage 的输入在部署地图上一次都没挪动过镜头（9/25 右键拖拽三个中点都拖不动，
# 9/29 滚轮 ±4 格没反应），章节地图上不能只赌一种。按顺序试，**记住这次运行里真正有效的那种**：
#   post_drag  后台左键拖拽（support_pick._drag：一次 operate() 里 PostMessage + 挪真光标）
#   wheel      后台滚轮（support_pick._wheel）。Unity UGUI 只能横向滚的 ScrollRect 会把竖直滚轮量
#              当横向用（OnScroll 里 delta.y 取反后赋给 delta.x），照源码推「往上滚 = 内容往左 = 看右边」；
#              推不准也没关系，第一次动了就按实际方向把符号学下来
#   real_drag  真键鼠左键拖拽（SendInput 按下 / 分步 SetCursorPos / SendInput 松开）。
#              **只在游戏窗口就是前台窗口时做**：真输入落在前台窗口上，游戏在后台时会点到你正在用的窗口
# 拖不动和拖到头从画面上分不出来，所以已经证明有效的方式某一下没动，就当作拖到头了，
# 不再把别的方式轮一遍（否则每到边界都要白试好几种）。**只在前后台状态和学到它时一样时才这么算**：
# 前台学会的方式到了后台（或反过来）可能就不灵了，那时它没动还是要把别的方式试一遍。
CHAPTER_SCROLL_METHODS = ('post_drag', 'wheel', 'real_drag')
CHAPTER_METHOD_NAMES = {'post_drag': '后台左键拖拽', 'wheel': '后台滚轮', 'real_drag': '真键鼠拖拽'}
CHAPTER_WHEEL_X = 0.50             # 滚轮在 80% 高度那条空白底图的正中滚
# 一下滚几格。UGUI 默认一格只挪 1 个单位，游戏一般会调大；10 格既够看出动没动，又不至于一下滚过一屏
CHAPTER_WHEEL_NOTCHES = 10
REAL_DRAG_STEPS = 16               # 真拖拽分几小步挪光标（一步跳到底游戏认不出是拖）
REAL_DRAG_STEP_SLEEP = 0.02
# 松手前停住多久：ScrollRect 松手时按最后的速度惯性滑行，停够了速度衰减完，挪多少就是手走了多少
REAL_DRAG_HOLD = 0.3
REAL_DRAG_CURSOR_TOLERANCE = 3     # 拖的过程中光标被挪开超过几像素，就是你在动鼠标
# 没做 / 做到一半放弃真拖拽的原因。前两种是「游戏在后台」，切到前台之后值得再试
REAL_BG = '游戏不在前台，真输入会落到你正在用的窗口上'
REAL_LOST_FG = '拖到一半游戏不在前台了，已松开左键'
REAL_COVERED = '拖动起点上盖着别的窗口'
REAL_USER_MOUSE = '拖到一半你动了鼠标，已松开左键'
REAL_REJECTED = '系统没收下模拟的鼠标按键'
REAL_USER_BUTTON = '你正按着鼠标键，没去按（免得把你的拖动截断）'
REAL_USER_MOVING = '你正在动鼠标，没去按'
REAL_BG_REASONS = (REAL_BG, REAL_LOST_FG)
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_VK_LBUTTON = 0x01
_VK_RBUTTON = 0x02
_SM_SWAPBUTTON = 23
_GA_ROOT = 2

# ---- 拖完之后，章节地图到底往哪边动了 ----
# 以前拿整屏缩略图做 cv2.phaseCorrelate(a, b)。它的 dx 是 b 里的内容相对 a 挪了多少：
# **看右边（从右往左拖、内容往左移）时 dx < 0**。2026-09-29 拿 13 张真章节地图（debug_tiles/20260929）
# 拼接平移合成验证过：平移 ≤1000px（3200 宽的 31%）次次精确；1100~1400px 大小偶尔读错（报成 20~50）；
# 过了半屏（≥1600px）就绕回去报成反号（320 宽的缩略图上 +180 和 -140 分不开）。
# 更要命的是**底图有视差**：同一章节不同位置的真截图（101324->101330、094950->101340）里，
# 卡片挪了 +1279 / +765px，没有卡片的顶部那条底图只挪了 +53 / +35px（约 4.4%）。
# 一拖整屏里就有两层位移在抢相位相关的峰：那两对真图碰巧锁在了卡片上，可把卡片抠出来平移、
# 底图只挪 4.4% 合成的图上，卡片挪 ≥700px 的 21 次全被判成了「没动」（锁在了底图上）。
# 所以现在直接看**卡片上的字**：同一段只出现一次的文字（关卡编号等）拖动前后在 OCR 里的 x 差，
# 符号约定和上面一样（< 0 = 卡片往左挪 = 看到了右边）。
# 缩略图只用来省 OCR：前后几乎一模一样就肯定没动，不用再 OCR。
CHAPTER_STILL_RATIO = 0.02         # 缩略图里变了（灰度差 >12）的像素不到这个比例 = 画面没动
CHAPTER_MIN_SHIFT = 0.03           # 卡片横向挪了屏幕宽的这个比例才算动了（3200 宽约 96px）
# 卡片连同上面的「普通战斗」、下面的编号所在的高度范围。顶栏标题 / 资源数、底下的收集进度 /
# 「普通 困难」都是不跟着地图动的固定 UI，不能算进来
CHAPTER_CARD_BAND = (0.18, 0.80)


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', ctypes.c_long), ('dy', ctypes.c_long), ('mouseData', ctypes.c_ulong),
                ('dwFlags', ctypes.c_ulong), ('time', ctypes.c_ulong), ('dwExtraInfo', ctypes.c_size_t)]


class _INPUT(ctypes.Structure):
    # 联合体里 MOUSEINPUT 最大，只放它，大小和系统的 INPUT 一致（64 位 40 字节）
    _fields_ = [('type', ctypes.c_ulong), ('mi', _MOUSEINPUT)]


def _send_mouse(flags):
    """真鼠标按键事件：SendInput，和物理鼠标一样进系统输入队列，落在光标底下的窗口上。

    :return: 系统收下了没有（被 UIPI 拦下时是 False）
    """
    inp = _INPUT(0, _MOUSEINPUT(0, 0, 0, flags, 0, 0))
    return ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT)) == 1


def _buttons_swapped():
    """系统设置里左右键互换了没有（左手鼠标）。读不到当没换。"""
    try:
        return bool(ctypes.windll.user32.GetSystemMetrics(_SM_SWAPBUTTON))
    except STOP_EXCEPTIONS:
        raise
    except Exception:
        return False


def _primary_button_flags():
    """真鼠标「主键」（游戏眼里的左键）按下 / 松开用的 SendInput 标志。

    SendInput 发的是物理按键，左右键互换过的系统上 LEFTDOWN 到了游戏那里是右键 ——
    pywinauto / AutoHotkey 都按 SM_SWAPBUTTON 反过来发，这里也一样。
    """
    if _buttons_swapped():
        return _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP
    return _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP


def _cursor_moved(a, b):
    """两次读到的光标位置差得超过容差 = 有人在动鼠标。"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > REAL_DRAG_CURSOR_TOLERANCE


def card_shift(before, after, width):
    """两屏之间卡片横向挪了多少：同一段文字拖动前后 x 的差，取中位数。

    :param before, after: {文字: x 中心}（见 StageSelectMixin._card_tokens）
    :return: (位移px, 对上了几段, 方向可信吗)；一段都对不上返回 None。
             **位移 < 0 = 卡片往左挪 = 看到了右边**
    """
    d = sorted(after[k] - before[k] for k in before if k in after)
    if not d:
        return None
    med = d[len(d) // 2]
    agree = sum(1 for v in d if abs(v - med) <= 0.05 * width)
    return med, len(d), agree * 2 > len(d)


def is_chapter_map(boxes, W=None, H=None):
    """这一帧是不是章节地图：整屏 OCR 结果里有关卡编号（并框的「学ZS-1-2」、断成两截的「LA」+「-1-8」也算）；
    给了 W / H 时，编号一个都没读全也认版式（map_layout）。

    10/13 11:11「瑞骋迎春至」：整屏 OCR 把每张卡的编号都读成两截（「LA」「-1-8」「S」），拖完一下就说
    「拖完之后不在章节地图上了」，轮询也当成「关卡外」，停了 6 分钟，下一关 LA-1-8 明明可打（I-55）。
    选关时的裁剪重读（extract_stage_codes）读得出整串编号 —— 只是这道门先没让它进去。"""
    if has_stage_code(boxes):
        return True
    return bool(W and H) and map_layout(boxes, W, H)


def map_layout(boxes, W, H):
    """章节地图的版式（不看关卡编号）：活动地图 = 顶上「活动剧情」+ 右下角「剧情」「物资」两个页签（chapter_nav.event_map_info）；
    主线地图 = 顶上「/6.5 驻行者·上篇」+ 右下角「普通」「困难」。关卡详情面板打开时右下角是「挑战」、页签被盖住，认不成这个。"""
    try:
        import chapter_nav as CN
    except ImportError:                # 打包漏了它：只认编号
        return False
    if CN.event_map_info(boxes, W, H) is not None:
        return True
    if CN.story_map_title(boxes, W, H):
        tabs = {b.name for b in boxes if getattr(b, 'name', None) and CN._in_band(b, CN.TAB_BAND, W, H)}
        return {'普通', '困难'} <= tabs
    return False


# ---- 关卡详情面板上看「打过没有」（通用关卡识别第 2 层，10/10）----
# 卡片样式每期活动都在换（I-28 的勾就没认出来），关卡详情面板却是同一套界面：「挑战目标」下面每一条前面有个六边形图标，
# 达成过就是橙色实心，没达成是灰色空心。达成任何一条都得先打赢，所以亮着一个 = 这关打过了。
# 你 10/10 截的 LA-1-5（打过）：三条的橙色像素占比 0.12～0.18；存下来的 6 张没打过的面板（7-5、7-9、8-10、6-9、定向精研、
# 113119）每一条都是 0.000（灰的、加上补充框的暗罩也一样）。OCR 的框从图标开始，图标那一列和表头「挑战目标」的左边对齐
PANEL_GOALS_HEAD = '挑战目标'
PANEL_GOALS_STOP = ('敌方单位', '特殊奖励', '作战奖励')
PANEL_X_MIN = 0.6                  # 关卡详情面板在屏幕右边
GOAL_ICON_WIN = (-0.6, 1.2)        # 图标那一列：表头左边缘往左 0.6h ~ 往右 1.2h（h = 那一行字高；伸进字里不要紧，字不是橙色）
GOAL_LIT_HUE = (8, 30)             # OpenCV 色相（0~180）：实测 21
GOAL_LIT_S = 110
GOAL_LIT_V = 120
GOAL_LIT_MIN = 0.04                # 橙色像素占比：亮着的 0.12~0.18，没亮的 0.000


def panel_goals(frame, boxes, W, H):
    """关卡详情面板「挑战目标」下面每一条前面的图标亮没亮。

    :param boxes: 整屏 OCR（W x H 坐标）
    :return: dict(cleared=True 有一条亮着（打过了）/ False 都没亮 / None 没找到「挑战目标」或下面没有条目（剧情点、收起来了）,
                  rows=[(文字, 橙色占比)])
    """
    out = dict(cleared=None, rows=[])
    if frame is None or not W or not H:
        return out
    named = [b for b in boxes or () if getattr(b, 'name', None)]
    heads = [b for b in named if PANEL_GOALS_HEAD in b.name and b.x > PANEL_X_MIN * W]
    if not heads:
        return out
    hd = min(heads, key=lambda b: b.y)
    stops = [b.y for b in named if b.x > PANEL_X_MIN * W and b.y > hd.y + hd.height
             and any(s in b.name for s in PANEL_GOALS_STOP)]
    y_lim = min(stops) if stops else hd.y + 6 * hd.height
    rows = [b for b in named if b is not hd and hd.y + 0.6 * hd.height < b.y < y_lim
            and abs(b.x - hd.x) < 2.5 * hd.height]
    if not rows:
        return out
    fh, fw = frame.shape[:2]
    sx, sy = fw / float(W), fh / float(H)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lit = False
    for b in sorted(rows, key=lambda b: b.y):
        h = b.height
        x0, x1 = int((hd.x + GOAL_ICON_WIN[0] * h) * sx), int((hd.x + GOAL_ICON_WIN[1] * h) * sx)
        y0, y1 = int(b.y * sy), int((b.y + h) * sy)
        roi = hsv[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if roi.size == 0:
            continue
        hue, sat, val = roi[..., 0], roi[..., 1], roi[..., 2]
        frac = float(((hue >= GOAL_LIT_HUE[0]) & (hue <= GOAL_LIT_HUE[1]) & (sat >= GOAL_LIT_S)
                      & (val >= GOAL_LIT_V)).mean())
        out['rows'].append((b.name.strip(), round(frac, 3)))
        lit = lit or frac >= GOAL_LIT_MIN
    if out['rows']:
        out['cleared'] = lit
    return out


def same_idle_screen(a, b):
    """两次「没有可推的」看到的是不是同一屏：(编号, 状态) 集合一个包含另一个，或者重合度 ≥ IDLE_SAME_SCREEN。"""
    if a is None or b is None:
        return False
    if a <= b or b <= a:
        return True
    return len(a & b) / float(len(a | b)) >= IDLE_SAME_SCREEN


def is_chapter_image(name):
    """「没有可推的」存的章节地图（原图 / 标注图 / 记录）。「点了没打开」的 _clickfail 不算，那是要留的证据。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem.endswith('_chapter') or stem.endswith('_chapter_ann')


def prune_chapter_images(root, keep):
    """章节地图图只留最近 keep 个文件（.json 记录跟着图删），不碰别的现场图。"""
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if is_chapter_image(f))
    images = [f for f in files if not f.endswith('.json')]
    for old in images[:-keep]:
        os.remove(old)
        note = os.path.splitext(old)[0] + '.json'
        if os.path.exists(note):
            os.remove(note)


class StageSelectMixin:
    """挂在 BaseGfTask 子类上，提供「在章节地图上挑一个没打过的关卡点进去」。"""

    # ---------------------------------------------------------------- 识别

    def _ocr_crop(self, img):
        """用 ok 自己的 OCR 读一小块图（和整屏那次同一个引擎、同样的错字修正），坐标是这块图里的。"""
        return self.ocr(frame=img)

    @staticmethod
    def _token_sig(boxes):
        """整屏 OCR 结果的粗略指纹：文字 + 大致位置。画面没变时它也不变。"""
        return tuple(sorted((b.name or '', int(b.x) // 24, int(b.y) // 24)
                            for b in boxes if getattr(b, 'name', None)))

    def read_stage_cards(self, boxes=None, deep='auto'):
        """读出当前章节地图上看得见的关卡卡片。

        :param boxes: 已经 OCR 过的整屏结果（实时辅助每轮只 OCR 一次，传进来复用）
        :param deep: 'auto' = 有迹象时才整屏分块深扫；True = 一定深扫（找不到可打的关卡时用）
        :return: [{'code', 'box', 'center', 'state', 'completed', 'unlocked', ...}]，
                 从左到右（同一竖列里从上到下）
        """
        frame = self.frame
        if frame is None:
            return []
        if boxes is None:
            boxes = self.ocr()
        _, accepted, rejected, info = extract_stage_codes(frame, boxes, self._ocr_crop,
                                                          deep=deep, details=True)
        if accepted:
            accepted, _ = repair_merged(frame, accepted, info['h_ref'], self._ocr_crop)
        accepted = self._white_pass(frame, boxes, accepted, info)
        self._stage_info = info
        for c in rejected:
            self.log_debug('关卡编号 %s 不算：%s' % (c.get('code'), c.get('reason')))
        ref_h = frame_ref_h([c['box'] for c in accepted]) if accepted else None
        title = getattr(self, '_map_title', None)
        run = self._run_mem(title)      # 这次运行的记忆也只取这张图的（I-36）
        cleared, dead, lost = run['cleared'], run['dead'], run['lost']
        # 点开面板核实过、其实没打过的（I-53）：记录不算数
        saved = self._saved_cleared(title) - run.setdefault('ledger_wrong', set())
        cards = []
        for c in accepted:
            code = c['code']
            r = classify_card(frame, c['box'], ref_h=ref_h, code=code)
            state, memo, ledger_only = r['state'], '', False
            seen_locked = state == 'locked'
            if state == 'available' and (c.get('left_cut') or c.get('right_cut')):
                state = 'unknown'          # 卡片伸出屏幕外，挂锁 / 对勾可能在外面
            if state != 'completed' and code in cleared:
                state, memo = 'completed', '（这次运行里打过）'
            elif state != 'completed' and code in saved:
                # 以前打赢过（重开任务之前 / 前几天），卡片上没认出勾（I-28）：先算打过。
                # 但卡片看着可打的记一笔（ledger_only）：这张图别的都没得打时点开面板核实一下 —— 记录可能是别的账号 /
                # 别的难度的（10/13 困难「驻行者」0/15，记录却是前一天打完的，被判成「整张地图打完」停下，I-53）
                ledger_only = state == 'available' and code not in lost and dead.get(code, 0) < STAGE_CLICK_GIVE_UP
                state, memo = 'completed', '（记录里打过）'
            elif state == 'available' and dead.get(code, 0) >= STAGE_CLICK_GIVE_UP:
                # 两种都是「这次运行里不再自动选」，但原因要写对（9/29 19:56 ZF-1-5 打输了，清单上却写「点了没反应」）
                state, memo = 'locked', ('（这次运行里不再自动选）' if code in lost else '（点了没反应）')
            x, y, w, h = [int(v) for v in c['box']]
            cards.append({
                'code': code,
                'box': Box(x, y, w, h, 1.0, code),
                # 点编号文字本身：它一定在卡片上（卡片的其余部分在各样式里朝哪边伸不一样）
                'center': (x + w // 2, y + h // 2),
                'state': state,
                'completed': state == 'completed',
                'unlocked': state in ('completed', 'available'),
                'lock': r['lock_score'],
                'check': r['check_score'],
                'certain': r['certain'],
                'disagree': r['disagree'],
                'memo': memo,
                'ledger_only': ledger_only,
                # 清单上显示的状态名：打输过的关卡画面上是解锁的，只是这次不再自动选，别显示成「未解锁」
                'shown': ('打输过' if (code in lost and memo)
                          else STATE_NAMES['locked'] if (seen_locked and memo == '（记录里打过）')   # 画面上锁着（I-60）
                          else STATE_NAMES[state]),
            })
        return cards

    def _white_pass(self, frame, boxes, accepted, info):
        """编号被盖住时（I-56）只留白字再读一遍，把普通那遍漏掉的编号补上（卡片状态照旧在原图上判）。

        什么时候读：普通那遍一个编号都没读出来 / 有读不通的碎片 / 卡片带里有「τ1=1」这种被盖住的字块；
        这张图上补出过编号的话，这次运行里每次都读（多一次整屏 OCR，约 1 秒）。"""
        run = self._run_mem(getattr(self, '_map_title', None))
        need = (run.get('white') or not accepted or bool(info.get('unresolved'))
                or bool(garbled_codes(boxes, frame.shape[0])))
        if not need:
            return accepted
        try:
            wf = white_text(frame)
            toks = self.ocr(frame=wf)
            _, extra, _, _ = extract_stage_codes(wf, toks, self._ocr_crop, deep=False, verify='suspicious',
                                                 details=True, max_calls=8)
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            self.log_debug('只留白字再读出错了：%r' % (e,))
            return accepted
        new, fixed = [], []
        for c in extra:
            same = [a for a in accepted if _overlap_frac(c['box'], a['box']) > 0.3]
            if not same:
                if c['code'] not in {a['code'] for a in accepted}:
                    new.append(c)
            elif c.get('exact') and all(a['code'] != c['code'] for a in same):
                # 同一个位置两遍读得不一样（被盖住时普通那遍会把 VA 读成 YA）：信只留白字的那遍
                fixed.append('%s->%s' % ('/'.join(a['code'] for a in same), c['code']))
                accepted = [a for a in accepted if a not in same]
                new.append(c)
        if new:
            if not run.get('white'):
                self.log_info('卡片上的编号被盖住了（%s），只留白字再读一遍，补上了 %s%s'
                              % (' / '.join((garbled_codes(boxes, frame.shape[0]) or ['普通那遍没读全'])[:3]),
                                 '、'.join(c['code'] for c in new), ('（改正 %s）' % '、'.join(fixed)) if fixed else ''))
            run['white'] = True
            accepted = sorted(accepted + new, key=lambda c: c['box'][0])
        return accepted

    def log_stage_cards(self, cards):
        for c in cards:
            self.log_info('  %-10s %s%s (挂锁%+.2f 对勾%s)%s' % (
                c['code'], c.get('shown') or STATE_NAMES[c['state']], c['memo'], c['lock'],
                '--' if c['check'] is None else '%+.2f' % c['check'],
                ('  两套检测不一致：' + ', '.join(c['disagree'])) if c['disagree'] else ''))

    # ---------------------------------------------------------------- 记忆

    def _run_mem(self, title=None):
        """这次运行的记忆，按地图标题分开：cleared 打过的编号 / dead 点了没反应的次数 / lost 打输过的编号。

        以前三样都只按编号记。可不同章节会有同一个编号：10/11「11.0 错位锚定·上篇」和「12.5 交织突进」都有 LS-1-1、LA-1-1，
        前一章打过的在后一章也算打过 —— 12.5 卡片上明明没勾 / 还锁着，却被判成整张地图打完，去下一章又锁着，停了 95 分钟（I-36）。
        标题按 same_title 比（容 OCR 丢一个字、间隔号「·」「•」不分）；读不出标题的单独一份，不和读得出的混。
        """
        mem = getattr(self, '_run_memory', None)
        client = self._game_client()
        if client and client != getattr(self, '_run_client', None):
            # 开着任务换了客户端（关了国际服、开了国服）：上一个客户端这次运行打过的 / 点了没反应的都不算（I-54 复查）
            if getattr(self, '_run_client', None) is not None:
                mem = None
                self._pick_verify = None
            self._run_client = client
        if mem is None:
            mem = self._run_memory = {}
        key = ''
        if title:
            key = next((k for k in mem if k and same_title(k, title)), None) or norm_title(title)
        run = mem.get(key)
        if run is None:
            run = mem[key] = {'cleared': set(), 'dead': {}, 'lost': set()}
        return run

    def mark_stage_cleared(self, code, title=None, how=None, why=''):
        """自己点进去并打完（或看完）的关卡：这次运行里不再打。

        :param title: 这关所在的活动名 / 主线章节名（选关时从地图顶上读的）
        :param how: 'won' 打赢的 / 'chain' 「继续前进」连过去的（编号是推算的）—— 同时记进文件（stage_memory），
                    重开也记得；None（剧情点）或读不出 title：只记这次运行
        :param why: 日志里接在后面的说明
        """
        s = self._run_mem(title)['cleared']
        new = code not in s
        s.add(code)
        saved = False
        led = self._ledger() if (how and title) else None
        if led is not None:
            try:
                saved = led.add(title, code, how, self._game_client())
            except Exception as e:
                self._ledger_warn('写', e)
        if saved:
            self.log_info('记下「%s」已经打完%s（记进了「%s」的记录，重开也记得）' % (code, why, title))
        elif new:
            self.log_info('记下「%s」这次运行里已经打完%s' % (code, why))

    def _ledger(self):
        """打赢的关卡记录（stage_memory）。设置里关掉 / 测试里（GF2_STAGE_MEMORY=off）返回 None。"""
        cfg = getattr(self, 'config', None) or {}
        if not cfg.get('记住打过的关卡', True):
            return None
        path = ledger_path()
        if path is None:
            return None
        led = getattr(self, '_ledger_obj', None)
        if led is None or led.path != path:
            led = self._ledger_obj = StageLedger(path)
        return led

    def _saved_cleared(self, title):
        """记录里这个活动 / 这一章打赢过的编号；没有活动名、记录关掉、文件读不了都是空集。"""
        led = self._ledger() if title else None
        if led is None:
            return set()
        try:
            return led.codes(title, self._game_client())
        except Exception as e:
            self._ledger_warn('读', e)
            return set()

    def _game_client(self):
        """连着的是哪个游戏客户端（exe 路径，见 stage_memory.client_key）：两个服 / 两个号打过的关分开记（I-54）。
        拿不到（插件版没有窗口对象、测试）返回 ''，记在共用那份。"""
        wins = (getattr(getattr(getattr(self, 'executor', None), 'interaction', None), 'hwnd_window', None),
                getattr(self, 'hwnd', None))
        for win in wins:
            try:
                key = client_key(getattr(win, 'exe_full_path', None))
            except Exception:
                key = ''
            if key:
                return key
        return ''

    def _ledger_warn(self, what, e):
        """记录文件读写出错：只说一次，不影响选关（照旧按卡片上的勾 + 这次运行的记忆）。"""
        if getattr(self, '_ledger_warned', False):
            return
        self._ledger_warned = True
        self.log_info('打过的关卡记录%s不了（%r），这次运行先只靠卡片上的勾和这次运行的记忆' % (what, e))

    def note_stage_click_failed(self, code):
        """点了卡片、下一轮还停在章节地图 —— 没打开关卡详情（多半是没解锁）。"""
        d = self._run_mem(getattr(self, '_map_title', None))['dead']      # 点的是这张图上的卡（I-36）
        d[code] = d.get(code, 0) + 1
        last = getattr(self, '_last_stage_click', None)
        if d[code] == 1 and last and last[0] == code:
            # 取证：9/29 SZ-1-2、SZ-1-6 都是「看着可打、点了没反应」（挂锁分数 -0.5~-0.6，比别的卡都接近有锁），
            # 当时没存图，分不清是锁没认出来还是点偏了。点之前那一帧 + 点的位置存下来
            self._save_stage_debug(last[1], frame=last[3], tag='clickfail', mark=last[2])
        if d[code] == STAGE_CLICK_GIVE_UP:
            self.log_info('点了 %d 次「%s」都没打开关卡详情，这次运行里不再点它'
                          % (STAGE_CLICK_GIVE_UP, code), notify=True)
        else:
            self.log_info('点了「%s」没打开关卡详情，下一轮再试一次' % code)

    # ---------------------------------------------------------------- 选关

    def pick_next_stage(self, boxes=None):
        """挑一个「已解锁且没打过」的关卡点开。

        从左往右挑第一个 —— 章节地图是往右推进的，最左边没打的通常就是下一关。

        :return: 点开的关卡编号；没有可选的返回 None
        """
        if boxes is None:
            boxes = self.ocr()
        sig = self._token_sig(boxes)
        now = time.time()
        cache = getattr(self, '_stage_cache', None)
        if cache and cache[0] == sig and now - cache[1] < STAGE_CACHE_SECONDS:
            return None            # 画面没变、上次也没东西可打：不重复做整套识别
        cards, todo = self._read_todo(boxes, sig, now)
        if not cards:
            return None
        if not todo:
            # 当前视野里没有 —— 下一关常常就在屏幕外，拖动章节地图往两边找一找
            code, boxes2, cards2 = self._search_by_scrolling(boxes, cards)
            if code:
                return code
            if getattr(self, '_scroll_yielded', False) or getattr(self, '_scroll_left_map', False):
                # 拖到一半你接手了 / 拖完画面已经不是章节地图了：这趟没找完，不下「没有可推的」的结论
                # （以前后者照样下结论，自动换章就拿这一轮开头那一屏的字去点现在这一屏，复查 F1）
                return None
            if boxes2 is not None:
                boxes, cards, sig = boxes2, cards2, self._token_sig(boxes2)
        if not todo:
            self._stage_cache = (sig, now)
            # 给「章节打完自动进下一章」（chapter_nav）看：这一轮刚下的结论、最后这一屏的卡
            self._no_todo_t = time.time()
            self._no_todo_cards = cards
            state_sig = frozenset((c['code'], c['state']) for c in cards)
            if not same_idle_screen(state_sig, getattr(self, '_stage_idle_sig', None)):
                # 同样的结论只说一次（以前每 2 秒刷一遍同样的 4 行）。边上半露的卡编号读得不稳，
                # 按重合度判断是不是同一屏，不按一模一样（见 IDLE_SAME_SCREEN）
                self._stage_idle_sig = state_sig
                self.log_info('这一屏 %d 个关卡都打过了 / 没解锁 / 看不全，没有可推的' % len(cards))
                self.log_stage_cards(cards)
                self._save_stage_debug(cards)
            return None
        return self._click_stage(cards, todo)

    def _read_todo(self, boxes, sig, now):
        """识别当前画面的卡片；一个可打的都没有时整屏深扫一次再下结论。-> (cards, todo)"""
        cards = self.read_stage_cards(boxes)
        todo = [c for c in cards if c['state'] == 'available']
        if not todo and not getattr(self, '_stage_info', {}).get('deep'):
            # 暗色的编号整屏那次可能根本没读出来（密米尔那张图的 SA-2-2 / SA-2-3）——
            # 下结论之前整屏分块深扫一次。同样的画面 30s 内不重复（深扫约 1s）。
            last = getattr(self, '_stage_deep_at', None)
            if not (last and last[0] == sig and now - last[1] < DEEP_RETRY_SECONDS):
                self._stage_deep_at = (sig, now)
                cards = self.read_stage_cards(boxes, deep=True)
                todo = [c for c in cards if c['state'] == 'available']
        return cards, todo or self._todo_of(cards)

    @staticmethod
    def _todo_of(cards):
        """这一屏挑哪张：可打的；一张都没有时，「只有记录说打过」（卡片看着可打，ledger_only）的第一张，
        点开看一眼关卡详情面板 —— 挑战目标亮着就关掉（RealtimeAssistTask._panel_says_cleared），没亮就照常打（I-53）。
        这样记录只省掉点开面板的那几秒，不会再把一张图判成「打完了」、停在下一章门口。"""
        todo = [c for c in cards if c['state'] == 'available']
        if todo:
            return todo
        verify = [c for c in cards if c.get('ledger_only')]
        return [dict(verify[0], verify=True)] if verify else []

    def _click_stage(self, cards, todo):
        self._stage_cache = None
        self._stage_idle_sig = None
        self._scroll_seen = set()
        self._sweep_edges = set()
        self._sweep_moved = False
        self._no_todo_t = None
        target = todo[0]
        # 「只有记录说打过」的那张：点开面板核实（见 _todo_of）。_panel_says_cleared 看到是它，面板没亮就说一声
        self._pick_verify = target['code'] if target.get('verify') else None
        if target.get('verify'):
            self.log_info('章节地图上找到 %d 个关卡，没有可打的；「%s」记录里打过、卡片上却没认出勾 —— '
                          '点开看一眼关卡详情（打过就关掉）' % (len(cards), target['code']), notify=True)
        else:
            self.log_info('章节地图上找到 %d 个关卡，%d 个待打，选「%s」'
                          % (len(cards), len(todo), target['code']), notify=True)
        self.log_stage_cards(cards)
        # 记下点之前那一帧：要是没点开，存图时用它（下一轮的画面可能已经被弹窗 / 提示挡住）
        frame = self.frame
        self._last_stage_click = (target['code'], cards, target['center'],
                                  frame.copy() if frame is not None else None)
        # 取证：这次运行里每种编号前缀（PA / PZ / 4 / BH…）第一次选关时存一张图（和「没有可推的」的图一起限量）。
        # 新活动换卡片样式时，以前只有「没有可推的」才存图 —— 拿不到可打 / 没解锁的卡长什么样（9/30 绿色勾）
        prefix = target['code'].rsplit('-', 2)[0]
        shot = getattr(self, '_pick_shot_prefixes', None)
        if shot is None:
            shot = self._pick_shot_prefixes = set()
        if prefix not in shot:
            shot.add(prefix)
            self._save_stage_debug(cards, frame=frame, tag='pick_chapter', mark=target['center'])
        self.click(*target['center'], after_sleep=1.5 + self.ui_delay,
                   name='stage_' + target['code'])
        self.next_frame()
        return target['code']

    # ---------------------------------------------------------------- 拖动章节地图

    def _chapter_thumb(self):
        """章节地图的灰度缩略图（去掉顶栏和底部按钮条），用来判断拖动之后画面变了没有。"""
        self.next_frame()
        frame = self.frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        sub = frame[int(0.12 * h):int(0.86 * h), :]
        return np.float32(cv2.cvtColor(cv2.resize(sub, (320, 150)), cv2.COLOR_BGR2GRAY))

    @staticmethod
    def _chapter_still(a, b):
        """两张缩略图几乎一模一样 -> 肯定没挪（章节地图静止时连底图都不动），不用再 OCR。"""
        if a is None or b is None or a.shape != b.shape:
            return False
        return float((cv2.absdiff(a, b) > 12).mean()) < CHAPTER_STILL_RATIO

    def _card_tokens(self, boxes):
        """整屏 OCR 里卡片那一层的文字 -> {文字: x 中心}。

        只要卡片带（CHAPTER_CARD_BAND）里**整屏只出现一次**的文字（关卡编号、卡片上的数字等）：
        「普通战斗」「剧情点」这种一屏好几个的，拖完之后对不上是哪一张。
        """
        h = self.height
        pos = {}
        for b in boxes or ():
            name = (getattr(b, 'name', None) or '').strip()
            if len(name) < 2:
                continue
            cy = b.y + b.height / 2.0
            if CHAPTER_CARD_BAND[0] * h <= cy <= CHAPTER_CARD_BAND[1] * h:
                pos.setdefault(name, []).append(b.x + b.width / 2.0)
        return {k: v[0] for k, v in pos.items() if len(v) == 1}

    def _chapter_foreground(self, it=None):
        """游戏窗口是不是前台窗口 —— 真键鼠输入只会落在前台窗口上。

        **不能叫 _game_foreground**：StageRunnerMixin 有同名方法给 user_busy()（你操作时不点）用，
        RealtimeAssistTask 的 MRO 里本混入排在它前面，同名就把它盖掉了（2026-09-29 审查时发现）。
        """
        if it is None:
            it = self._genshin() if hasattr(self, '_genshin') else None
        hwnd = getattr(it, 'hwnd', None) if it is not None else None
        if not hwnd:
            return False
        try:
            import win32gui
            return win32gui.GetForegroundWindow() == hwnd
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return False

    def _real_drag(self, it, fx, fy, tx, ty):
        """真键鼠左键从 (fx,fy) 拖到 (tx,ty)（相对坐标）。

        开始前、按下前、每挪一小步之前都确认游戏窗口还是前台窗口，一变就立刻松手放弃；
        你手上正按着鼠标键就压根不按；光标被别人挪开了（你在动鼠标）—— 按下前就不按，
        拖到一半就松手放弃。左右键互换过的系统上发物理右键（游戏眼里还是左键）。
        松开左键放在 finally 里 —— 不管出什么事（包括你按了停止）都一定松开，
        不然系统一直当左键按着，你点哪儿都成了拖。最后光标放回原处。
        全程用 time.sleep：ok 的 sleep 在暂停时会一直等下去，那时左键还按着。

        :return: None = 拖完了；否则是没拖 / 拖到一半放弃的原因（REAL_*）
        """
        import win32api
        import win32gui
        hwnd = getattr(it, 'hwnd', None)
        if not hwnd or win32gui.GetForegroundWindow() != hwnd:
            return REAL_BG
        pts = []
        for i in range(REAL_DRAG_STEPS + 1):
            x = int((fx + (tx - fx) * i / REAL_DRAG_STEPS) * self.width)
            y = int((fy + (ty - fy) * i / REAL_DRAG_STEPS) * self.height)
            ax, ay = it.capture.get_abs_cords(x, y)
            pts.append((int(ax), int(ay)))
        # 起点上盖着置顶的小窗（比如别的工具的悬浮窗）时，真按下去按的是它
        hit = win32gui.WindowFromPoint(pts[0])
        if not hit or win32gui.GetAncestor(hit, _GA_ROOT) != hwnd:
            return REAL_COVERED
        # 你手上正按着鼠标键（比如正在自己拖地图）：这时候再按一下、松一下，会把你的拖动截断。
        # GetAsyncKeyState 看的是物理键，左右键互换过也一样，两个都看
        if any(win32api.GetAsyncKeyState(vk) & 0x8000 for vk in (_VK_LBUTTON, _VK_RBUTTON)):
            return REAL_USER_BUTTON
        down, up = _primary_button_flags()
        saved = win32api.GetCursorPos()
        pressed = False
        try:
            win32api.SetCursorPos(pts[0])
            first = win32api.GetCursorPos()
            time.sleep(0.05)
            last = win32api.GetCursorPos()
            if win32gui.GetForegroundWindow() != hwnd:
                return REAL_LOST_FG
            # 光标放到起点之后这 50ms 里被挪开了 = 你在动鼠标：还没按下去，不拖了。
            # 拿两次读回来的值比、不拿目标点比：高 DPI 下 SetCursorPos 再读回来可能差一两像素
            if _cursor_moved(first, last):
                return REAL_USER_MOVING
            clock = getattr(self, '_now', None) or time.time
            t_down = clock()
            pressed = True              # 先记上再按：按键一旦发出去，后面哪一步出错 finally 都会松开
            if not _send_mouse(down):
                return REAL_REJECTED
            time.sleep(0.08)
            for p in pts[1:]:
                if win32gui.GetForegroundWindow() != hwnd:
                    return REAL_LOST_FG
                if _cursor_moved(win32api.GetCursorPos(), last):
                    return REAL_USER_MOUSE
                win32api.SetCursorPos(p)
                last = win32api.GetCursorPos()
                time.sleep(REAL_DRAG_STEP_SLEEP)
            # 停住等惯性消掉；这时候前台变了、或者你动了鼠标，就不等了马上松手（手已经走完，这一下还算数）
            for _ in range(3):
                if win32gui.GetForegroundWindow() != hwnd or _cursor_moved(win32api.GetCursorPos(), last):
                    break
                time.sleep(REAL_DRAG_HOLD / 3)
            return None
        finally:
            try:
                if pressed:
                    _send_mouse(up)
                    real = getattr(self, '_sr_note_real', None)
                    if real is not None:
                        real(t_down, clock())        # 真按下 / 松开：input_watch 看得到，别当成你点的（I-52）
            finally:
                try:
                    win32api.SetCursorPos(saved)
                except STOP_EXCEPTIONS:
                    raise
                except Exception:
                    pass                # 还原光标失败（比如锁屏了）不能盖掉上面真正的异常

    def _chapter_try(self, it, method, direction, before):
        """用 method 往 direction 那边（+1 = 想看右边）挪一下章节地图，再量卡片实际挪了多少。

        :param before: 挪之前这一屏 {'thumb', 'tokens', 'boxes'}
        :return: None = 这种方式现在用不了；否则 (动了吗, 卡片位移px, 方向可信吗, 挪完这一屏)
        """
        name = CHAPTER_METHOD_NAMES[method]
        d = direction * self._chapter_scroll_sign[method]
        x0, x1 = CHAPTER_DRAG_X if d > 0 else CHAPTER_DRAG_X[::-1]
        try:
            if method == 'post_drag':
                self._drag(x0, CHAPTER_DRAG_Y, x1, CHAPTER_DRAG_Y, key='left')
            elif method == 'wheel':
                if not hasattr(self, '_wheel'):
                    return None
                self._wheel(int(CHAPTER_WHEEL_X * self.width), int(CHAPTER_DRAG_Y * self.height),
                            d * CHAPTER_WHEEL_NOTCHES)
            else:
                if getattr(self, '_chapter_real_blocked', None):
                    self._chapter_drag_blocked = True    # 这一下也算被挡（判「拖到头」用，见 _search_by_scrolling）
                    return None         # 这一趟已经因为这个没用上了，别每一下都再试、再说一遍
                why = self._real_drag(it, x0, CHAPTER_DRAG_Y, x1, CHAPTER_DRAG_Y)
                if why:
                    self._chapter_drag_blocked = True
                    self._chapter_real_blocked = why
                    self.log_info('没用%s：%s' % (name, why))
                    return None
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            if method == 'real_drag':
                self._chapter_drag_blocked = True
                self._chapter_real_blocked = '出错了'
            self.log_info('「%s」出错了，这一下跳过它：%r' % (name, e))
            return None
        # support_pick._drag 自己已经等过 0.4 + 界面等待，这里只补 0.3（和改之前一样）；
        # 再加一遍界面等待，默认 1.5s 的配置下每拖一下平白多等 1.5s
        self.sleep(0.3 if method == 'post_drag' else 0.3 + self.ui_delay)
        return self._chapter_measure(before, name, direction)

    def _chapter_measure(self, before, name, direction):
        """挪完之后看卡片横向挪了多少。:return: (动了吗, 卡片位移px, 方向可信吗, 挪完这一屏)"""
        side = '右' if direction > 0 else '左'
        thumb = self._chapter_thumb()
        if self._chapter_still(before['thumb'], thumb):
            self._chapter_last_still = True
            self.log_info('%s（想看%s边）：画面没变 -> 没动' % (name, side))
            return False, 0, True, dict(before, thumb=thumb)
        boxes = self._chapter_view_boxes = self.ocr()
        after = {'thumb': thumb, 'tokens': self._card_tokens(boxes), 'boxes': boxes}
        r = card_shift(before['tokens'], after['tokens'], self.width)
        if r is None:
            # 画面变了，但前后对不上同一段文字（一下挪出去超过一屏？界面变了？）——
            # 当作动了，交给调用方重新认界面和卡片；方向不知道，不拿它学
            self.log_info('%s（想看%s边）：画面变了，但前后对不上同一张卡片 -> 当作动了，方向看不出来'
                          % (name, side))
            return True, 0, False, after
        dx, n, sure = r
        self._chapter_last_n = n
        moved = abs(dx) >= CHAPTER_MIN_SHIFT * self.width
        if not moved and not sure:
            # 画面变了、对上的几段文字却彼此不一致（有的没动、有的挪了一大截）：说明不了「拖到了头」。
            # 10/10 20:08「素银匿信」往右拖一下，2 段文字一段 0、一段挪了 → 中位数 0，当成拖到头，
            # 右边的 YZ-1-3 / YZ-1-5 没看就判「剧情页签打完了」（I-28）。当作动了、方向看不准，让调用方重新认这一屏
            self.log_info('%s（想看%s边）：画面变了，对上的 %d 段文字彼此不一致 -> 当作动了，方向看不出来'
                          % (name, side, n))
            return True, 0, False, after
        self.log_info('%s（想看%s边）：卡片横向位移 %+d px（%d 段文字对上%s）-> %s'
                      % (name, side, dx, n, '' if sure else '，彼此不一致',
                         ('看到了%s边' % ('右' if dx < 0 else '左')) if moved else '没动'))
        return moved, dx, sure, after

    def _learn_chapter_method(self, method, fg):
        # 连同「学到时游戏在不在前台」一起记：后台消息在前台 / 后台时游戏的反应可能不一样
        # （ok 的 operate() 只在后台时才先发激活消息、锁键鼠），真键鼠拖拽更是只有前台能用
        self._chapter_scroll_fg = fg
        prev = getattr(self, '_chapter_scroll_method', None)
        if prev != method:
            self._chapter_scroll_method = method
            # 只有头一回学到时弹通知：之后游戏前后台来回切、换来换去的只记日志，不刷通知
            self.log_info('章节地图用「%s」拖得动（游戏在%s），后面先用它'
                          % (CHAPTER_METHOD_NAMES[method], '前台' if fg else '后台'), notify=prev is None)

    def _drag_chapter(self, direction):
        """横向挪一下章节地图。direction=+1 看右边，-1 看左边。

        按 CHAPTER_SCROLL_METHODS 的顺序试，一种没动就换下一种；学到有效的方式之后只用它
        （它没动 = 拖到头了）。挪之前这一屏的 OCR 取 self._chapter_view_boxes（没有就现 OCR）。

        :return: 章节地图真的往这边挪了吗。挪了的话，挪完那一屏的 OCR 在 self._chapter_after_boxes
        """
        self._chapter_after_boxes = None
        self._chapter_last_still = False    # 这一下是不是「画面一点没变」（_chapter_measure 记，见 CHAPTER_STILL_RETRY_WAIT）
        if not isinstance(getattr(self, '_chapter_scroll_sign', None), dict):
            self._chapter_scroll_sign = {m: 1 for m in CHAPTER_SCROLL_METHODS}
        learned = getattr(self, '_chapter_scroll_method', None)
        it = self._genshin() if hasattr(self, '_genshin') else None
        fg = it is not None and self._chapter_foreground(it)
        if it is None:
            methods = ('post_drag',)    # 插件版 / 不是 Genshin 交互：只有老办法
        elif learned and getattr(self, '_chapter_scroll_fg', fg) == fg:
            methods = (learned,)        # 有效的方式这一下没动 = 拖到头了，不再试别的
        elif learned:
            # 学到它的时候游戏在前台、现在在后台（或者反过来）：它这一下没动说明不了是拖到头了 ——
            # 以前这里直接当拖到头，前台学会的方式到了后台拖不动，连着两屏就「这次运行里不再试」，
            # 切回前台也不会再拖。先试它，不动再按老顺序把别的试一遍
            methods = (learned,) + tuple(m for m in CHAPTER_SCROLL_METHODS if m != learned)
        else:
            methods = CHAPTER_SCROLL_METHODS
        thumb = self._chapter_thumb()
        boxes = getattr(self, '_chapter_view_boxes', None)
        if boxes is None:
            boxes = self._chapter_view_boxes = self.ocr()
        view = {'thumb': thumb, 'tokens': self._card_tokens(boxes), 'boxes': boxes}
        for method in methods:
            r = self._chapter_try(it, method, direction, view)
            if r is None:
                continue                # 这种方式现在用不了（真键鼠：游戏在后台）
            moved, dx, sure, view = r
            if not moved:
                continue
            self._chapter_after_boxes = view['boxes']
            if not sure:
                return True             # 动了但看不准往哪边：让调用方重新认卡片，不拿它学
            flip_left = getattr(self, '_chapter_flip_left', None)
            if not isinstance(flip_left, dict):
                flip_left = self._chapter_flip_left = {}
            recent_flip = flip_left.get(method, 0) > 0
            flip_left[method] = max(0, flip_left.get(method, 0) - 1)
            if (dx < 0) != (direction > 0) and learned == method and getattr(self, '_chapter_last_n', 2) < 2 \
                    and not recent_flip:
                # （刚翻过方向之后的几下不拦：那次翻转本身可能就是错的，得让 1 段的结果把它翻回来 —— I-60 复查）
                # 这种方式的方向已经学过了，这一下只对上 1 段文字就说反了：多半是对错了卡（10/14 08:56「逃离赛博格」
                # 被水印盖住的卡上「VA」「c1-1」这种碎片好几张卡上都有，方向被它翻过去又翻回来，一趟拖了 13 下，I-60）
                self.log_info('「%s」这一下看着是反方向，可只对上 1 段文字 —— 不改学过的方向' % CHAPTER_METHOD_NAMES[method])
                return True
            if (dx < 0) != (direction > 0):
                # 往反方向动了：这种方式的方向符号翻过来记住。刚才反着挪了一下，
                # 再按翻过来的方向挪两下 —— 先回原位，再往要的方向走一步
                self._chapter_scroll_sign[method] *= -1
                flip_left[method] = 3           # 之后 3 下里，1 段文字的反方向也能把它翻回来
                self._chapter_flipped = True    # 给 _search_by_scrolling 看：往左找时才发现反了，右边还没找
                self.log_info('「%s」的方向和猜的相反，以后反过来用' % CHAPTER_METHOD_NAMES[method])
                self._learn_chapter_method(method, fg)
                back = self._chapter_try(it, method, direction, view)
                if not back or not back[0]:
                    return True         # 回不去了：至少画面挪过，交给调用方认这一屏
                step = self._chapter_try(it, method, direction, back[3])
                if not step or not step[0]:
                    return False        # 回到原位了，原位就是这一侧的尽头
                self._chapter_after_boxes = step[3]['boxes']
                return True
            self._learn_chapter_method(method, fg)
            return True
        return False

    def _scroll_user_took_over(self):
        """拖章节地图途中看你一眼：你操作了游戏（或者分不出来时你在动键鼠）-> 原因，这趟不找了；
        只是点开游戏看 / 挪鼠标 -> 等你停手，None 接着拖（I-52，见 stage_runner._sr_check_user）。"""
        check = getattr(self, '_sr_check_user', None)
        if check is not None:
            return check('拖章节地图')
        busy = getattr(self, 'user_busy', None)
        return '动了键鼠' if (busy is not None and busy()) else None

    def _drag_chapter_noted(self, direction):
        """拖一下章节地图，记一笔「这段时间的键鼠是自己的」：拖拽不走 _sr_input（一次 operate() 里自己发的），
        真键鼠拖拽会刷新系统的「最后一次输入」，不记的话下一下之前会把自己当成你。
        每一下单独记、紧挨着这一下 —— 以前两下连同中间等你停手的那段一起记，你在那段里点的就被当成了它自己的（I-52 复查）"""
        note = getattr(self, '_sr_note_own', None)
        t0 = self._now() if note else None
        try:
            return self._drag_chapter(direction)
        finally:
            if note:
                note(t0, self._now())

    def _search_by_scrolling(self, boxes, cards):
        """当前视野里没有可打的关卡时，拖动章节地图往右、再往左找。

        :return: (点开的编号, 最后一屏的 OCR, 最后一屏的卡片)；没找到时编号是 None；
                 压根没拖（拖不动 / 有弹窗 / 这片区域刚找过）时后两个也是 None
        """
        self._scroll_yielded = False
        self._scroll_left_map = False
        if not hasattr(self, '_drag'):
            return None, None, None
        if self.find_boxes(boxes, match=POPUP_RE):
            if not getattr(self, '_popup_warned', False):
                self._popup_warned = True
                msg = '章节地图上有弹窗（章节收集奖励）挡着，先不拖地图 —— 关掉弹窗之后接着找'
                alert = getattr(self, 'alert_user', None)     # 要你处理（stuck_alert），测试桩没有就照旧 notify
                if alert is not None:
                    alert(msg)
                else:
                    self.log_info(msg, notify=True)
            return None, None, None
        self._popup_warned = False
        # 上一趟只是因为游戏在后台才没用上真键鼠拖拽，现在游戏到前台了：
        # 不管之前放没放弃、这片找没找过，都再找一趟
        fg_retry = getattr(self, '_chapter_retry_fg', False) and self._chapter_foreground()
        if getattr(self, '_chapter_drag_fail', 0) >= CHAPTER_DRAG_GIVE_UP and not fg_retry:
            return None, None, None
        # 按「编号 + 状态」记：同一片区域刚刚来回找过一遍、状态也没变，就别每轮都拖；
        # 你手动打完一关回来（状态变了）会重新找
        now_seen = {(c['code'], c['state']) for c in cards}
        seen = getattr(self, '_scroll_seen', None) or set()
        if now_seen and now_seen <= seen and not fg_retry:
            return None, None, None
        self.log_info('这一屏没有可打的关卡，拖动章节地图找一找下一关'
                      + ('（游戏到前台了，这回能用真键鼠拖）' if fg_retry else ''))
        # 这一趟往哪边拖到了头（拖不动了）：两头都到了，才算整张地图都看过（chapter_nav 判断「这页打完了」用）
        self._sweep_edges = set()
        self._chapter_retry_fg = False
        self._chapter_real_blocked = None
        self._chapter_view_boxes = boxes
        seen = set(now_seen)
        moved_any = moved_right = False
        last = (None, None)
        passes = [(1, CHAPTER_SCROLL_RIGHT), (-1, CHAPTER_SCROLL_LEFT)]
        method_at_start = getattr(self, '_chapter_scroll_method', None)
        right_retried = False
        right_moves = 0
        i = 0
        while i < len(passes):
            direction, steps = passes[i]
            i += 1
            self._chapter_flipped = False
            if direction < 0:
                # 往右挪了几下，往左就先挪回来这么多下；镜头本来就在右端附近（往右挪不了几下）时，
                # 往左至少一共拖 CHAPTER_LEFT_MIN 下 —— 和以前一样够得着起点左边 6 屏（复查 CH-1）
                steps = max(steps + right_moves, CHAPTER_LEFT_MIN, CHAPTER_SCROLL_LEFT_MAX + right_moves)
            for step in range(steps):
                # 一趟最多拖二十几下（I-15 之后），中途你接手了就马上停，别在你手底下拖地图（复查 CH-2）。
                # 这片没找完：不记进 _scroll_seen，你停手之后下一轮从头再找
                why = self._scroll_user_took_over()
                if why:
                    self.log_info('拖章节地图找关卡途中你%s，这趟先不找了' % why)
                    self._scroll_yielded = True
                    return None, None, None
                # 「真键鼠被挡」按每一下单独算（_chapter_drag_blocked）：10/10 18:33 往右第一下拖法还没学会、试真键鼠被挡
                # （游戏在后台），整趟的记号 _chapter_real_blocked 一直留到这一趟结束，以前拿它判，后面用学会的后台拖法
                # 拖到头也不算到头 —— 两头都判不出，不换章（I-27）
                self._chapter_drag_blocked = False
                dragged = self._drag_chapter_noted(direction)
                if (not dragged and step == 0 and getattr(self, '_chapter_last_still', False)
                        and getattr(self, '_chapter_scroll_method', None)
                        and not getattr(self, '_chapter_drag_blocked', False)):
                    # 往这边的第一下画面一点没变：多半是这一下游戏没收到，等一下再拖一次（见 CHAPTER_STILL_RETRY_WAIT）
                    why = self._scroll_user_took_over()
                    if why:
                        self.log_info('拖章节地图找关卡途中你%s，这趟先不找了' % why)
                        self._scroll_yielded = True
                        return None, None, None
                    self.log_info('往%s边拖的第一下画面一点没变（拖到头时一般会弹一下），再拖一次确认'
                                  % ('右' if direction > 0 else '左'))
                    self.sleep(CHAPTER_STILL_RETRY_WAIT)
                    dragged = self._drag_chapter_noted(direction)
                blocked_now = getattr(self, '_chapter_drag_blocked', False)
                if not dragged:
                    # 只有「拖法学到过（真拖得动）、这一下也没被挡（后台 / 拖到一半游戏不在前台）」时没动才算拖到了头；
                    # 拖法根本不灵时两头「没动」说明不了什么 —— 拿它判「整张地图都看过了」会漏掉屏幕外的关
                    if getattr(self, '_chapter_scroll_method', None) and not blocked_now:
                        self._sweep_edges.add('right' if direction > 0 else 'left')
                    break                  # 拖到头了（或者拖不动），换个方向
                moved_any = True
                moved_right = moved_right or direction > 0
                if direction > 0:
                    right_moves += 1
                # 动了就清零「连着拖不动」—— 要在这里清：找到关卡是从循环里直接 return 的，
                # 以前只在最后清，找到了计数也不清，攒到上限就再也不拖了
                self._chapter_drag_fail = 0
                # 判断动没动的时候已经 OCR 过挪完的这一屏，直接用
                boxes2 = getattr(self, '_chapter_after_boxes', None) or self.ocr()
                self._chapter_view_boxes = boxes2
                if not is_chapter_map(boxes2, getattr(self, 'width', None), getattr(self, 'height', None)):
                    # 界面变了（拖动被当成点击点开了什么？）—— 交给下一轮重新认界面
                    self.log_info('拖完之后不在章节地图上了，交给下一轮重新认界面')
                    self._scroll_left_map = True
                    return None, None, None
                cards2 = self.read_stage_cards(boxes2)
                seen |= {(c['code'], c['state']) for c in cards2}
                last = (boxes2, cards2)
                todo = self._todo_of(cards2)
                if todo:
                    return self._click_stage(cards2, todo), None, None
                if direction > 0 and cards2 and all(c['state'] == 'locked' for c in cards2):
                    self._sweep_edges.add('right')     # 进度的尽头（复查 F5：以前没记，剩下全锁着的页签永远判不出来）
                    break                  # 一整屏都没解锁：已经越过进度了，再往右只会更远
            else:
                if direction < 0 and moved_any:
                    self.log_info('往左拖了 %d 下还没到头，先不拖了（上限）' % steps)
            if (direction < 0 and not right_retried and not moved_right and 'right' not in self._sweep_edges
                    and method_at_start is None and getattr(self, '_chapter_scroll_method', None)
                    and not self._chapter_flipped):
                # 往右那趟拖法还没学会、一下都没挪动，往左时才学会：右边其实一眼都没看（I-27）。回头往右再找一趟
                right_retried = True
                self.log_info('拖法刚学会，回头往右边再找一趟')
                passes.append((1, CHAPTER_SCROLL_RIGHT))
            if direction < 0 and self._chapter_flipped and not moved_right and len(passes) == 2:
                # 往右那趟一下都没挪动，往左找时才发现这种方式方向反了（滚轮方向猜反、又正好停在最左边：
                # 往右那趟其实一直往左边顶，顶不动）。_drag_chapter 已经把它挪回原位、方向也学对了，
                # 右边却一眼都没看 —— 补一趟往右的，不然这片记进 _scroll_seen 之后就再也不找了
                self.log_info('方向学对了，回头往右边再找一趟')
                passes.append((1, CHAPTER_SCROLL_RIGHT))
        self._scroll_seen = seen
        # 这一趟真的挪动过：拖法现在是灵的，两头「没动」才算拖到了头。一下都没挪动的一趟分不清是一屏放得下、还是拖法
        # 悄悄不灵了（10/1 10:10 纪念日礼赞 剧情：学到的拖法两边都 +0px，其实地图更长），chapter_nav 不拿它判「看完了」
        self._sweep_moved = moved_any
        # 真键鼠拖拽只是因为游戏在后台才没用上（没拖完这一趟）：切到前台之后再找一趟
        self._chapter_retry_fg = getattr(self, '_chapter_real_blocked', None) in REAL_BG_REASONS
        if moved_any:
            self._chapter_drag_fail = 0
        else:
            self._chapter_drag_fail = getattr(self, '_chapter_drag_fail', 0) + 1
            if self._chapter_drag_fail >= CHAPTER_DRAG_GIVE_UP:
                if self._chapter_retry_fg:
                    msg = ('连着 %d 次都拖不动章节地图 —— 游戏在后台时只能发后台消息，拖不动的话把游戏切到前台'
                           '（切过去之后会用真键鼠再拖一趟）；在那之前，下一关在屏幕外时需要你拖一下')
                else:
                    msg = ('连着 %d 次都拖不动章节地图，这次运行里不再试 —— 下一关在屏幕外时需要你拖一下'
                           '（游戏在后台时只能发后台消息，拖不动的话把游戏切到前台）')
                self.log_info(msg % CHAPTER_DRAG_GIVE_UP, notify=True)
        return None, last[0], last[1]

    def _save_stage_debug(self, cards, frame=None, tag='chapter', mark=None):
        """「没有可推的」/「点了没打开」时把章节地图存下来（原图 + 标了判定的缩略图）。

        新活动换了卡片样式、判错了的时候，拿这两张图就能对着查、也能拿来补标定。存图失败不影响流程。

        :param frame: 用哪一帧（默认当前帧）；「点了没打开」用的是点之前那一帧
        :param mark: 在标注图上画个十字的点（点下去的位置）
        """
        try:
            frame = self.frame if frame is None else frame
            if frame is None:
                return
            folder = os.path.join(DEBUG_DIR, time.strftime('%Y%m%d'))
            os.makedirs(folder, exist_ok=True)
            stem = os.path.join(folder, time.strftime('%H%M%S') + '_' + tag)
            cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(stem + '.jpg')
            vis = frame.copy()
            colors = {'completed': (0, 165, 255), 'available': (0, 255, 0),
                      'locked': (0, 0, 255), 'unknown': (200, 200, 200)}
            for c in cards:
                b = c['box']
                cv2.rectangle(vis, (b.x, b.y), (b.x + b.width, b.y + b.height), colors[c['state']], 3)
                cv2.putText(vis, '%s %s' % (c['code'], c['state']), (b.x, max(0, b.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, colors[c['state']], 3)
            if mark:
                cv2.drawMarker(vis, (int(mark[0]), int(mark[1])), (255, 0, 255), cv2.MARKER_CROSS, 60, 5)
            h, w = vis.shape[:2]
            cv2.imencode('.jpg', cv2.resize(vis, (w // 2, h // 2)),
                         [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(stem + '_ann.jpg')
            self.log_info('已存章节地图现场图：%s.jpg' % stem)
            note = getattr(self, '_save_case_note', None)       # support_pick 的现场记录（.json）
            if note:
                run = self._run_mem(getattr(self, '_map_title', None))
                note(stem + '.jpg', tag, extra={
                    'cards': [[c['code'], c['state'], c['box'].x, c['box'].y, c['box'].width, c['box'].height]
                              for c in cards],
                    'click': list(map(int, mark)) if mark else None,
                    'map_title': getattr(self, '_map_title', None),
                    'dead_hits': dict(run['dead']), 'lost': sorted(run['lost']), 'cleared': sorted(run['cleared'])})
            # 只清章节地图自己的图（单独限量，见 CHAPTER_DEBUG_KEEP）；别的现场图归 support_pick.prune_debug_dir 管。
            # 以前这几行在 return 后面，一次都没执行过
            prune_chapter_images(DEBUG_DIR, CHAPTER_DEBUG_KEEP)
            return stem + '.jpg'
        except STOP_EXCEPTIONS:
            raise                       # self.frame 在暂停时会 sleep，点了停止会从这里抛出来
        except Exception as e:
            self.log_info('存章节地图现场图失败（不影响流程）：%r' % (e,))
