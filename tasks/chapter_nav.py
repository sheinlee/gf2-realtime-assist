# -*- coding: utf-8 -*-
"""章节打完自动进下一章 —— 只管「战役推进 · 往期活动」里的活动剧情（ISSUES.md I-17）。

10/1 你演示了两遍（界面记录 debug_tiles/20261001/101852～103600_page）：
  活动的章节地图：顶上正中一行小字是活动名（纪念日礼赞 / 共诉隐匿时…），下面一行大字「活动剧情」；
  右下角两个页签「剧情」「物资」，亮着的（金色 / 红色底）是当前这页；左下角收集数「15/15 可领取」。
    1. 一个页签的关卡都打完了：切到另一个页签接着打（09:29 你从「剧情」切到「物资」）
    2. 两个页签都打完：左下角「可领取」点开「收集数」面板，把「领取」都点掉（每点一下弹「获得道具」，
       点「点击空白处关闭」），再关掉面板
    3. 点左上角「<」回到「往期活动」：顶上一排 剧情战役 / 往期活动 / 补给作战 / 模拟作战，左边 活动剧情 /
       悖论操演 / 模拟习练，中间一排活动卡片从左到右 01 02 03…，卡片下面是名字和「完成度N%」。
       这一页有时停在刚打的那张附近（10:19），有时停在最左边（10:35），要左右拖着找
    4. 点刚打完那张卡**右边**的下一张（完成度 100% 的跳过），进到它的章节地图，接着自动选关。
       不能挑「最左边没打完的」：03「密米尔的返演·上篇」一直是 0%，你是跳过它往后打的
主线（「剧情战役」）一章打完是游戏自己退回章节列表的（10/1 12:07 打完 5.0「异频共振」最后一关 5-10，
放完「第6.0章 简谐周期」解锁动画就回到了列表，它停在那里没动 —— 你说「打完一章回到上级目录，停了」）：
  列表左边一列章节，选中的那行左边是深色条（章节号前带「/」）；解锁的章节名是黑字、有大字章节号「6.0」，
  锁着的是灰字、只有小字「C-6.5.」和锁；右边「章节进度」，右下角「确认」。11:26 你从这一页点「异频共振」进了 5.0 的地图
  （界面记录 112614 -> 112617，3 秒）。
    5. 刚自动打完一关（90 秒内）就到了这一页、你没动过键鼠、选中的正是刚才在打的那一章：点它下面那一章
       （要是解锁的），看它被选中了再点「确认」，到了它的章节地图接着自动选关。下一章锁着 / 认不出就停手提醒
    6. 停在主线章节地图上、整张地图都打完了（12:18 你又进了打完的 5.0，它找了一圈「没有可推的」就停了）：
       点左上角「<」回章节列表（14:31 你就是这样回去的），再按 5 进下一章。
       剩下的关卡锁着（14:30 6.0 打完 6-5，6-6 及以后都锁着 —— 多半是指挥官等级不够）不是打完了：不走，说清楚交给你
       左下角收集数「可领取」先领了再说（10/11 你说「把主线领奖励加上」；收集数面板和往期活动的一样，见 2）
只点认得出的东西：页签文字、「可领取」「领取」「点击空白处关闭」、卡片名字、章节名、「确认」、面板右上角认出来的叉。
唯一按位置点的是左上角「<」—— 点之前要确认画面是活动章节地图、而且那个位置真有「<」的形状
（back_arrow_visible，在所有存过的图上量过）。哪一步对不上就停手提醒，不瞎点；同一章连着失败两次，
这次运行里就不再试。你一动键鼠就停（和借助战一样的关口），界面停在哪下一轮从那接着认。
"""
import re
import time

import cv2
import numpy as np

from support_pick import STOP_EXCEPTIONS, UserTakeoverError
from stage_memory import norm_title, same_title  # noqa: F401  活动名比较（放在 stage_memory：选关、记账都要用，它不依赖别的模块）

try:
    import close_x
except ImportError:          # 测试里单独加载时可能没有
    close_x = None

# ---- 活动章节地图 ----
EVENT_MAP_TITLE = '活动剧情'
TAB_STORY = '剧情'
TAB_SUPPLY = '物资'
TITLE_BAND = (0.30, 0.0, 0.70, 0.11)       # 顶上正中：活动名 +「活动剧情」
TAB_BAND = (0.75, 0.88, 1.0, 1.0)          # 右下角两个页签
COUNTER_BAND = (0.0, 0.84, 0.22, 1.0)      # 左下角收集数「15/15」「可领取」/「再收集5可领取」
COUNTER_RE = re.compile(r'^(\d{1,3})/(\d{1,3})$')
# 「可领取」时收集数右上角有个红色「!」角标，OCR 会把它粘在总数后面：10/12 01:30「15/15」读成「15/150」（I-45）。
# 一章最多三十来关、每关 3 颗，总数到不了 100：三位数的总数去掉最后一位；粘上的是个非数字（「15/15!」）也认
COUNTER_BADGE_RE = re.compile(r'^(\d{1,3})/(\d{1,2})[^\d/]$')
COUNTER_TOTAL_MAX = 99
TAB_SAT_MARGIN = 25.0                      # 亮着的页签底色饱和度要比另一个高出这么多（实测 >100）

# ---- 往期活动 ----
HUB_MARK = '往期活动'
HUB_TABS = ('剧情战役', '补给作战', '模拟作战')
HUB_TOP_BAND = (0.45, 0.0, 1.0, 0.10)
HUB_PCT_RE = re.compile(r'^完成度(\d{1,3})')
HUB_CARD_Y = (0.70, 0.86)                  # 卡片名字（0.77H）和「完成度」（0.81H）
HUB_TITLE_GAP = 0.06                       # 名字在「完成度」正上方这么近以内（卡片图上的「心智洄游」在 0.11H 外）
HUB_TITLE_DX = 0.08
HUB_DRAG_Y = 0.55
HUB_DRAG_X = (0.72, 0.28)                  # 从右往左拖 = 看右边（更新的活动）
HUB_DRAG_MAX = 8                           # 往一个方向最多拖几下
HUB_MOVED = 0.05                           # 同一张卡横向挪了这么多（屏宽）算拖动了

# ---- 左上角「<」----
BACK_ARROW = (0.029, 0.048)                # 实测 (93,86)@3200x1800，所有活动章节地图都一样

# ---- 收集奖励 ----
REWARD_PANEL_MARK = '收集数'
REWARD_BTN = '领取'
REWARD_CLAIMABLE = '可领取'                # 只认这三个字；「再收集5可领取」是还没满
REWARD_BTN_BAND = (0.55, 0.20, 0.85, 0.80)
REWARD_X_BAND = (0.70, 0.18, 0.86, 0.36)   # 面板右上角的叉（实测 (2512,472)@3200x1800）
POPUP_CLOSE = '点击空白处关闭'
REWARD_MAX = 6
# 关「获得道具」（I-57）：10/13 21:35 刚弹出来就点了「点击空白处关闭」，游戏没收到，弹窗一直开着，它说「没回到收集数面板」就停了 35 分钟。
# 先等它弹稳再点；点完这么久还开着就再点，最多点这么多次
POPUP_SETTLE = 0.6
POPUP_CLOSE_WAIT = 4.0
POPUP_CLOSE_TRIES = 3
# 点「可领取」打开收集数面板（I-63）：10/14 11:42 第一下游戏没收到，以前只点一下、12 秒后整段换章失败。
# 正常 3 秒内就开（今天 16 次都是），这么久没开、画面还是那张地图就再点，最多点这么多次
CLAIM_OPEN_WAIT = 4.0
CLAIM_OPEN_TRIES = 3
# 领奖励领到一半你接手了（I-63）：面板 / 「获得道具」是它自己点开的。你这么多秒没操作游戏就接着领完
# （和借助战接手之后接着借同一个规则，RealtimeAssistTask.TAKEOVER_RESUME_IDLE）；停下超过 CLAIM_RESUME_MAX 秒就不管了
CLAIM_RESUME_IDLE = 30
CLAIM_RESUME_MAX = 600
CLAIM_RESUME_TIMES = 2

# ---- 节奏 / 次数 ----
NAV_POLL = 1.0
NAV_WAIT = 12.0
NAV_ENTER_WAIT = 20.0                      # 进新活动可能先加载一会
NAV_TAB_SWITCH_MAX = 3                     # 每章最多切几次页签（防来回切）
NAV_FAIL_MAX = 2                           # 同一章自动换章失败这么多次，这次运行里不再试
NAV_FRESH = 5.0                            # 「没有可推的」是这么多秒内刚下的结论才算数
# 画面上有这些（面板 / 弹窗 / 对话框）就先不动：隔着弹窗点下面的页签、「可领取」、「<」都会点错（复查 NAV-S2）
MODAL_RE = re.compile(r'收集数|获得道具|点击空白处关闭|物品详情|^取消$|^确认$')
HUB_EDGE_X = (0.13, 0.985)                 # 往期活动卡片的完成度 / 名字碰到这两条线算被屏幕切掉了，读数不可信（复查 RR-1）

# ---- 主线：「剧情战役」章节列表（界面记录 20261001/112614_page、120724_page）----
STORY_TAB = '剧情战役'
STORY_TOP_BAND = (0.45, 0.0, 0.70, 0.10)   # 顶上一排页签里的「剧情战役」（实测中心 0.57W, 0.05H）
STORY_PROGRESS = '章节进度'                 # 右边，有时和百分比读成一块「章节进度/.100%」
# 丢一个字也认：10/12 09:48 当时读成「节进度」，12 秒里一帧都没认出章节列表，换章停下（I-50）
STORY_PROGRESS_RE = re.compile('章节进|节进度')
STORY_CONFIRM = '确认'
STORY_CONFIRM_BAND = (0.75, 0.85, 1.0, 1.0)  # 右下角「确认」（实测 (2823,1678)@3200x1800）
STORY_ROW_X = 0.25                          # 左边一列章节名，中心在这条线左边
STORY_ROW_Y = (0.09, 0.86)                  # 再往下是「猎人评定」「同调演练」两张卡（0.89H 以下）
STORY_CODE_X = 0.10                         # 章节名左边的章节号
STORY_ROW_DY = 0.035                        # 章节号和章节名差这么多以内（x 屏高）算同一行
STORY_CODE_RE = re.compile(r'(\d{1,2}\.\d)')
STORY_SEL_SLASH_RE = re.compile(r'/\s*\d{1,2}\.\d')   # 选中那行章节号前面的「/」（「0/ 5.0」）；「8.7/」后面那个是对勾，不算
STORY_LOCKED_RE = re.compile(r'[Cc]-\s*\d')  # 锁着的章节只有小字「C-6.5.」（解锁的是大字「6.0」，前面的「0」是红色「!」）；
                                             # 10/12 09:48 读成小写「c-11.0.n」（I-50 复查）
STORY_MAP_BAND = (0.30, 0.0, 0.70, 0.09)    # 主线章节地图顶上正中「/5.0异频共振」「/5.0 异频共振」
STORY_MAP_RE = re.compile(r'^\W*(\d{1,2}\.\d)\s*(\w.*)$')
# 图上量的（120724 / 112614 两张）：选中那行左边深色条的灰度中位数 52、别的行 235；
# 解锁的章节名黑字（字里最暗 5% 的灰度 39~42），锁着的灰字（145~171）
STORY_BAR_DARK = 110
STORY_BAR_LIGHT = 170
STORY_TITLE_DARK = 95
STORY_TITLE_GREY = 120
STORY_TEXT_CONTRAST = 40                    # 字和底至少差这么多才按颜色判（截图异常 / 纯色帧不判）
STORY_END_FRESH = 90.0                      # 上一关这么多秒之内刚打完（12:07:21 打完 5-10，12:07:22 就是列表了）
STORY_MAP_FRESH = 2700.0                    # 这么久之内在那一章的地图上待过（一关最长 40 分钟，多阶段的关十几分钟常见）
STORY_REENTER_TRIES = 3                     # 游戏退回列表后回这一章：「确认」最多点几次（列表刚出来那几秒不响应）
STORY_REENTER_WAIT = 8.0                    # 每点一次等这么久看有没有回到地图

def _center(b):
    return b.x + b.width / 2.0, b.y + b.height / 2.0


def _in_band(b, band, W, H):
    cx, cy = _center(b)
    return band[0] * W <= cx <= band[2] * W and band[1] * H <= cy <= band[3] * H


# 章节地图顶上正中、活动名下面那行大字：往期活动是「活动剧情」；当期活动「琥珀月光」（10/10 10:28 的 102818_page）是「剧情模式」，
# 右下角也没有「剧情」「物资」页签（页签在活动首页：挑战模式 / 物资模式 / 剧情模式）
MAP_HEADS = (EVENT_MAP_TITLE, '剧情模式', '物资模式', '挑战模式')


def map_head_title(boxes, W, H):
    """顶上正中大字（活动剧情 / 剧情模式…）正上方的活动名，不要求有页签 —— 只给打过的关卡记录（stage_memory）当名字用，
    换章（event_map_info）照旧要两个页签。活动名至少两个汉字（「琥珀月光」旁边的花体字「MSTYL」读成「M/TY」，不算）。没有返回 None。"""
    named = [b for b in boxes if getattr(b, 'name', None)]
    heads = [b for b in named if b.name.strip() in MAP_HEADS and _in_band(b, TITLE_BAND, W, H)]
    if not heads:
        return None
    hb = heads[0]
    hx, hy = _center(hb)
    above = [b for b in named if b is not hb and _in_band(b, TITLE_BAND, W, H) and _center(b)[1] < hy
             and abs(_center(b)[0] - hx) <= 0.1 * W and len(re.findall(r'[一-鿿]', b.name)) >= 2]
    return min(above, key=lambda b: hy - _center(b)[1]).name.strip() if above else None


def event_map_info(boxes, W, H):
    """活动章节地图（顶上「活动剧情」+ 右下角「剧情」「物资」两个页签）读出来的东西；不是返回 None。

    :return: dict(title 活动名 / None, title_box, tabs={'剧情': 框, '物资': 框}, counter=(已收集, 总数) / None,
                  claimable=「可领取」框 / None)
    """
    named = [b for b in boxes if getattr(b, 'name', None)]
    head = [b for b in named if b.name == EVENT_MAP_TITLE and _in_band(b, TITLE_BAND, W, H)]
    if not head:
        return None
    tabs = {}
    for b in named:
        if b.name in (TAB_STORY, TAB_SUPPLY) and _in_band(b, TAB_BAND, W, H):
            tabs.setdefault(b.name, b)
    if len(tabs) < 2:
        return None
    hb = head[0]
    hx, hy = _center(hb)
    above = [b for b in named if b is not hb and _in_band(b, TITLE_BAND, W, H) and _center(b)[1] < hy
             and abs(_center(b)[0] - hx) <= 0.1 * W and len(norm_title(b.name)) >= 2]
    title = min(above, key=lambda b: hy - _center(b)[1]).name if above else None
    counter, claimable = map_counter(named, W, H)
    return dict(title=title, title_box=hb, tabs=tabs, counter=counter, claimable=claimable)


def map_counter(boxes, W, H):
    """章节地图左下角的收集数：((已收集, 总数) / None, 「可领取」框 / None)。
    往期活动和主线的地图是同一个版式（10/11 主线 6.7「45/45 可领取」、7.0 的收集数面板截图上量过）。"""
    counter, claimable = None, None
    for b in boxes:
        name = getattr(b, 'name', None)
        if not name or not _in_band(b, COUNTER_BAND, W, H):
            continue
        c = parse_counter(name)
        if c:
            counter = c
        if name == REWARD_CLAIMABLE:
            claimable = b
    return counter, claimable


def parse_counter(text):
    """「15/15」-> (15, 15)；右上角的「!」角标粘在后面（「15/150」「15/15!」）也认（见 COUNTER_BADGE_RE）。读不出返回 None。"""
    t = re.sub(r'\s+', '', text or '')
    m = COUNTER_RE.match(t) or COUNTER_BADGE_RE.match(t)
    if not m:
        return None
    got, total = int(m.group(1)), int(m.group(2))
    if total > COUNTER_TOTAL_MAX and got <= total // 10:
        total //= 10
    if got > total:
        return None                     # 收集的比总数还多：读错了，别拿它判「收满了」
    return got, total


def tab_of_codes(codes):
    """关卡编号第二个字母是 F 的是「物资」页签（NF / AF / BF / MF…），S / A / Z 是「剧情」。拿不准返回 None。"""
    ks = [c for c in codes if c and re.match(r'^[A-Z]{2}-\d', c)]
    if not ks:
        return None
    f = sum(1 for c in ks if c[1] == 'F')
    if f == len(ks):
        return TAB_SUPPLY
    if f == 0:
        return TAB_STORY
    return None


def selected_tab(frame, tabs):
    """哪个页签亮着：选中的按钮底色是金色 / 红色（饱和度高），没选中的是灰的。分不清返回 None。"""
    if frame is None or len(tabs) < 2:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]
    sat = {}
    for name, b in tabs.items():
        cx, cy = _center(b)
        x0, x1 = int(max(0, cx - 0.04 * W)), int(min(W, cx + 0.04 * W))
        y0, y1 = int(max(0, cy - 0.016 * H)), int(min(H, cy + 0.016 * H))
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        sat[name] = float(np.median(roi[..., 1]))
    a, b = sat.get(TAB_STORY), sat.get(TAB_SUPPLY)
    if a is None or b is None or abs(a - b) < TAB_SAT_MARGIN:
        return None
    return TAB_STORY if a > b else TAB_SUPPLY


def tab_verdict(seen, edges, cleared, counter):
    """这个页签还有没有能打的：'todo' 有可打的 / 'done' 打完了 / 'blocked' 剩下的都锁着 / 'unsure' 说不准。

    :param seen: 这一趟拖着看过的 (编号, 状态)；同一个编号取最好的那次（看全了的那次）
    :param edges: 这一趟两头都拖到了吗（{'left', 'right'}）
    :param cleared: 这次运行里打完的编号（画面上没认出对勾也算打完）
    :param counter: 左下角收集数 (已收集, 总数)；收满了就是游戏自己说这页打完了
    """
    rank = {'unknown': 0, 'locked': 1, 'completed': 2, 'available': 3}
    best = {}
    for code, st in seen:
        if code in cleared:
            st = 'completed'
        if rank.get(st, 0) >= rank.get(best.get(code), -1):
            best[code] = st
    states = [s for s in best.values() if s != 'unknown']      # 看不全的卡在别的视野里看全过
    if 'available' in states:
        return 'todo'
    if counter and counter[1] > 0 and counter[0] >= counter[1]:
        return 'done'
    swept = {'left', 'right'} <= set(edges or ())
    if swept and states and all(s == 'completed' for s in states):
        return 'done'
    if swept and 'locked' in states:
        return 'blocked'
    return 'unsure'


def hub_info(boxes, W, H):
    """「战役推进 · 往期活动」页：读出这一屏的活动卡片（从左到右）；不是这一页返回 None。

    :return: dict(cards=[dict(title, box, x, pct)])，pct 读不出来是 None
    """
    named = [b for b in boxes if getattr(b, 'name', None)]
    if not any(b.name == HUB_MARK and _in_band(b, HUB_TOP_BAND, W, H) for b in named):
        return None
    if not any(b.name in HUB_TABS and _in_band(b, HUB_TOP_BAND, W, H) for b in named):
        return None
    pcts = []
    for b in named:
        m = HUB_PCT_RE.match(re.sub(r'\s+', '', b.name))
        if m and HUB_CARD_Y[0] * H <= _center(b)[1] <= HUB_CARD_Y[1] * H:
            if b.x <= HUB_EDGE_X[0] * W or b.x + b.width >= HUB_EDGE_X[1] * W:
                continue                # 被屏幕边切掉的卡：「完成度100%」会读成 10 / 1，名字也会少字
            pcts.append((b, int(m.group(1))))
    cards = []
    for pb, pct in pcts:
        px, py = _center(pb)
        above = [t for t in named if t is not pb and not HUB_PCT_RE.match(re.sub(r'\s+', '', t.name))
                 and 0 < py - _center(t)[1] <= HUB_TITLE_GAP * H and abs(_center(t)[0] - px) <= HUB_TITLE_DX * W
                 and len(norm_title(t.name)) >= 2]
        if not above:
            continue                    # 名字被挡住 / 卡片露出一半：这张先不算
        t = min(above, key=lambda t: py - _center(t)[1])
        if t.x <= HUB_EDGE_X[0] * W or t.x + t.width >= HUB_EDGE_X[1] * W:
            continue
        cards.append(dict(title=t.name, box=t, x=_center(t)[0], pct=pct if pct <= 100 else None))
    cards.sort(key=lambda c: c['x'])
    return dict(cards=cards)


def back_arrow_visible(frame):
    """左上角 (93,86)@3200 那里是不是一个细的、尖朝左的「<」（亮底暗字、暗底亮字都认）。

    实测：活动章节地图（深色）、开着收集数面板（变暗）、往期活动页（浅底深色箭头）都认；
    「获得道具」弹窗、战斗、剧情对话画面上不认（tests/chapter/nav_image_test.py）。"""
    if frame is None:
        return False
    H, W = frame.shape[:2]
    s = W / 3200.0
    crop = frame[int(35 * s):int(140 * s), int(55 * s):int(140 * s)]
    if crop.size == 0:
        return False
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = float(np.median(g))
    for pol in (1, -1):
        d = (g - med) * pol
        t = max(18.0, 0.45 * float(d.max()))
        m = (d >= t).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh, area = [int(v) for v in st[i]]
        if area < 40 * s * s or bw <= 0 or bh <= 0:
            continue
        if not (18 <= bw / s <= 50 and 40 <= bh / s <= 80 and 1.3 <= bh / float(bw) <= 3.0
                and 0.12 <= area / float(bw * bh) <= 0.5):
            continue
        comp = lab == i
        if float(d[comp].mean()) < 25:
            continue
        ys, xs = np.nonzero(comp)
        left = xs <= xs.min() + max(2, int(0.15 * bw))
        if abs((ys[left].mean() - (by + bh / 2.0)) / bh) > 0.15:
            continue                    # 尖不在左边正中
        right = xs >= xs.max() - max(2, int(0.3 * bw))
        ry = (ys[right] - by) / float(bh)
        if (ry < 0.3).mean() >= 0.25 and (ry > 0.7).mean() >= 0.25 and ((ry > 0.35) & (ry < 0.65)).mean() <= 0.15:
            return True                 # 右边是上下两个尾巴、中间空着
    return False


def _norm_code(code):
    return re.sub(r'^0+(?=\d)', '', code) if code else None


def _same_story(code_a, title_a, code_b, title_b):
    """主线是不是同一章：章节名对得上，或者两边的章节号都读得出、一样。章节名 OCR 会读成繁体、粘上别的字
    （10/11 18:07 列表上「疊影效应」、地图上「叠影效应」），章节号是数字，不受影响（I-39）。"""
    if same_title(title_a, title_b):
        return True
    a, b = _norm_code(code_a), _norm_code(code_b)
    return bool(a and b and a == b)


def story_map_title(boxes, W, H):
    """主线章节地图顶上的「/5.0异频共振」-> ('5.0', '异频共振')；不是返回 None。"""
    for b in boxes:
        name = getattr(b, 'name', None) or ''
        if not _in_band(b, STORY_MAP_BAND, W, H):
            continue
        m = STORY_MAP_RE.match(name.strip())
        if m and len(re.findall(r'[\u4e00-\u9fff]', m.group(2))) >= 2:
            return _norm_code(m.group(1)), m.group(2).strip()
    return None


def _gray_patch(frame, x0, y0, x1, y1, W, H):
    """frame 上 (x0,y0)-(x1,y1)（W x H 坐标）那一块的灰度；取不到返回 None。"""
    if frame is None:
        return None
    fh, fw = frame.shape[:2]
    sx, sy = fw / float(W), fh / float(H)
    a = frame[max(0, int(y0 * sy)):max(0, int(y1 * sy)), max(0, int(x0 * sx)):max(0, int(x1 * sx))]
    if a.size == 0 or a.shape[0] < 2 or a.shape[1] < 2:
        return None
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a


def story_list_info(boxes, W, H, frame=None):
    """「战役推进 · 剧情战役」的章节列表；不是这一页返回 None。

    :return: dict(rows=[dict(title, box, code, cy, unlocked True/False/None, selected)]（从上到下）,
                  selected=选中的那行 / None, next=它下面那行 / None, confirm=「确认」框)
    选中 / 解锁优先按图判（深色条、黑字灰字，见 STORY_BAR_DARK 上面），图上判不出再看 OCR（「/」、大字章节号）。
    """
    named = [b for b in boxes if getattr(b, 'name', None)]
    if not any(b.name == STORY_TAB and _in_band(b, STORY_TOP_BAND, W, H) for b in named):
        return None
    if not any(STORY_PROGRESS_RE.search(b.name) and _center(b)[0] > 0.6 * W for b in named):
        return None
    confirm = [b for b in named if b.name == STORY_CONFIRM and _in_band(b, STORY_CONFIRM_BAND, W, H)]
    if not confirm:
        return None
    codes = [b for b in named if _center(b)[0] < STORY_CODE_X * W
             and (STORY_CODE_RE.search(b.name) or STORY_LOCKED_RE.search(b.name))]
    rows = []
    for b in named:
        cx, cy = _center(b)
        if cx >= STORY_ROW_X * W or not (STORY_ROW_Y[0] * H <= cy <= STORY_ROW_Y[1] * H):
            continue
        # 「2.0第二信号」：章节号和名字读成了一块。左边的橙色「!」还会读成「0」「ρ」粘在最前面（10/11「010.0远日点·下篇」
        # 「ρ15.0 火花之罪·下篇」，以前整串都当成名字，点完对不上，换章停了 4 次，I-39）—— 章节号前面不是汉字的都不要
        # 打完的章节，章节号后面的橙色对勾会读成「/」粘在名字前面（10/12 01:10「16.0/歧路双轨·上篇」「12.5/交织突进」，I-50 复查）
        m = re.match(r'^[^\u4e00-\u9fff]*?(\d{1,2}\.\d)\.?[\s/\uff0f]*(?=[\u4e00-\u9fff])', b.name)
        title = (b.name[m.end():] if m else b.name).strip()
        if len(re.findall(r'[\u4e00-\u9fff]', title)) < 2:
            continue
        code = m.group(1) if m else None
        label = False
        slash = bool(m and '/' in b.name[:m.start(1)])        # 选中的标记是章节号**前面**的「/」，后面那个是对勾
        for c in codes:
            if c is b or abs(_center(c)[1] - cy) > STORY_ROW_DY * H:
                continue
            if STORY_LOCKED_RE.search(c.name):
                label = True
            else:
                code = code or STORY_CODE_RE.search(c.name).group(1)
                slash = slash or bool(STORY_SEL_SLASH_RE.search(c.name))
        rows.append(dict(title=title, box=b, code=_norm_code(code), cy=cy, label=label, slash=slash,
                         unlocked=None, selected=False))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r['cy'])
    bars = []
    for r in rows:
        b = r['box']
        bar = _gray_patch(frame, 0.002 * W, r['cy'] - 0.01 * H, 0.012 * W, r['cy'] + 0.01 * H, W, H)
        bars.append(None if bar is None else float(np.median(bar)))
        t = _gray_patch(frame, b.x, b.y, b.x + b.width, b.y + b.height, W, H)
        dark = None
        if t is not None:
            p5, med = float(np.percentile(t, 5)), float(np.median(t))
            if med - p5 >= STORY_TEXT_CONTRAST:
                dark = True if p5 <= STORY_TITLE_DARK else (False if p5 >= STORY_TITLE_GREY else None)
        if dark is None:
            dark = False if r['label'] else (True if r['code'] else None)
        r['unlocked'] = dark
    dark_rows = [i for i, v in enumerate(bars) if v is not None and v <= STORY_BAR_DARK]
    if len(dark_rows) == 1 and all(v is not None and v >= STORY_BAR_LIGHT for i, v in enumerate(bars)
                                   if i != dark_rows[0]):
        sel = dark_rows[0]
    else:
        slash = [i for i, r in enumerate(rows) if r['slash']]
        sel = slash[0] if len(slash) == 1 else None
    if sel is not None:
        rows[sel]['selected'] = True
    return dict(rows=rows, selected=rows[sel] if sel is not None else None,
                next=rows[sel + 1] if sel is not None and sel + 1 < len(rows) else None, confirm=confirm[0])


class ChapterNavMixin:
    """挂在 RealtimeAssistTask 上：章节地图「没有可推的」之后，切页签 / 领奖励 / 回往期活动选下一章；
    主线一章打完、游戏自己退回「剧情战役」章节列表之后，进下一章。

    宿主要有：ocr / click_box / click_relative / back / sleep / next_frame / frame / width / height / _now /
    _drag（support_pick）/ _save_stuck_image（stage_runner）/ log_info，以及 stage_select 留下的
    _scroll_seen / _sweep_edges / _no_todo_t / _no_todo_cards / _run_mem（这次运行的记忆，按地图标题分开）。
    开关（实例属性）：_auto_next_chapter / _switch_tabs_first / _claim_rewards_first。
    """

    # ---------------------------------------------------------------- 入口

    def _maybe_next_chapter(self, boxes):
        """章节地图上刚下了「没有可推的」：这一页打完了就切页签，两页都打完就领奖励、回往期活动选下一章。

        :return: True = 做了点什么（切了页签 / 换了章）
        """
        if not getattr(self, '_auto_next_chapter', True):
            return False
        t = getattr(self, '_no_todo_t', None)
        if t is None or time.time() - t > NAV_FRESH:
            return False                # 不是这一轮刚下的结论（缓存 / 你在操作 / 拖到一半你接手了）
        W, H = self.width, self.height
        info = event_map_info(boxes, W, H)
        if info is None:
            smap = story_map_title(boxes, W, H)
            if smap is not None:
                return self._maybe_story_leave(smap[1], smap[0])
            self._nav_note('*', 'not_event', '章节地图上没有可打的了，但这不是「往期活动」的活动章节'
                                             '（没有右下角的「剧情」「物资」页签），不自动换章')
            return False
        # 动手前重新看一眼：这一轮开头那一屏可能已经过时了（拖动途中弹了东西 / 你点了别处，复查 F1 / NAV-S1）；
        # 开着面板 / 弹窗时先不动（复查 NAV-S2）。后面点的位置都用这一屏的
        self.next_frame()
        fresh = self.ocr()
        info2 = event_map_info(fresh, W, H)
        if info2 is None or not same_title(info2['title'], info['title']):
            return False
        if any(MODAL_RE.search(b.name or '') for b in fresh):
            return False
        info = info2
        title = info['title']
        if not title:
            self._nav_note('*', 'no_title', '章节地图上没有可打的了，但读不出顶上的活动名，不自动换章')
            return False
        mem = self._nav_memory(title)
        if mem['disabled']:
            return False
        codes = [c for c, _ in (getattr(self, '_scroll_seen', None) or ())]
        codes += [c['code'] for c in (getattr(self, '_no_todo_cards', None) or ())]
        cur = self._nav_selected_tab(info)
        by_code = tab_of_codes(codes)
        if cur is None or (by_code is not None and by_code != cur):
            cur = by_code if cur is None else None
        if cur is None:
            self._nav_note(title, 'tab', '「%s」读不出现在是哪个页签（剧情 / 物资），不自动换章' % title)
            return False
        # 这一趟一下都没挪动过：两头「没动」分不清是一屏放得下还是拖法不灵了，不算看完了整张地图（复查 F2）
        edges = getattr(self, '_sweep_edges', None) if getattr(self, '_sweep_moved', False) else set()
        verdict = tab_verdict(
            set(getattr(self, '_scroll_seen', None) or ()) |
            {(c['code'], c['state']) for c in (getattr(self, '_no_todo_cards', None) or ())},
            edges, set(self._run_mem(title)['cleared']), info['counter'])
        if verdict in ('todo', 'unsure'):
            if verdict == 'unsure':
                self._nav_note(title, 'unsure_' + cur, '「%s」的「%s」页签上没找到可打的，但还没确认整张地图两头都看过、'
                                                         '收集数也没满，先不切页签 / 不换章' % (title, cur), notify=True)
            return False
        # 这一页里有这次运行打输过 / 点了两次都没打开的关：没打完，不能当成「打完了」换章（复查 NAV-S3 / F3：
        # 以前它们被算成「锁着」，判 blocked，接着就切页签、换章，说「打完了」）。切到另一页去打可以，离开不行
        run = self._run_mem(title)
        lost, dead = set(run['lost']), run['dead']
        stuck = sorted(set(c for c in codes if c in lost or dead.get(c, 0) >= 2))
        if stuck:
            verdict = 'lost'
        mem['tabs'][cur] = verdict
        other = TAB_SUPPLY if cur == TAB_STORY else TAB_STORY
        if (getattr(self, '_switch_tabs_first', True) and mem['tabs'].get(other) not in ('done', 'lost')
                and mem['switches'] < NAV_TAB_SWITCH_MAX):
            return self._nav_run(self._nav_switch_tab, info, cur, other, verdict)
        bad = [t for t, v in mem['tabs'].items() if v == 'lost']
        if bad:
            self._nav_note(title, 'lost', '「%s」的%s页签有打输过 / 点不开的关%s，这一章没打完，不自动换章 —— '
                                          '你处理完之后我接着打' % (title, '、'.join('「%s」' % t for t in bad),
                                                                   '（%s）' % '、'.join(stuck) if stuck else ''))
            return False
        return self._nav_run(self._nav_leave_chapter, info)

    # ---------------------------------------------------------------- 记忆 / 小工具

    def _nav_memory(self, title):
        allm = getattr(self, '_nav_chapters', None)
        if allm is None:
            allm = self._nav_chapters = {}
        key = norm_title(title)
        if key not in allm:
            allm[key] = dict(tabs={}, switches=0, fails=0, disabled=False, noted=set())
        return allm[key]

    def _nav_note(self, title, key, msg, notify=False):
        """同一章同一种情况只说一次。notify=True 的都是「不往下走了、交给你」：走「需要你处理」的醒目通知。"""
        mem = self._nav_memory(title or '*')
        if key not in mem['noted']:
            mem['noted'].add(key)
            if notify:
                self._nav_alert(msg)
            else:
                self.log_info(msg)

    def _nav_alert(self, msg):
        """要你处理：有卡住提醒（stuck_alert）就醒目地弹，没有（测试桩）就照旧 notify。"""
        alert = getattr(self, 'alert_user', None)
        if alert is not None:
            alert(msg)
        else:
            self.log_info(msg, notify=True)

    def _nav_selected_tab(self, info):
        return selected_tab(self.frame, info['tabs'])

    def _nav_back_arrow_ok(self):
        return back_arrow_visible(self.frame)

    def _nav_reset_map_memory(self):
        """换了页签 / 换了章：章节地图那边的「找过了、没东西」都作废。"""
        self._stage_cache = None
        self._stage_idle_sig = None
        self._scroll_seen = set()
        self._sweep_edges = set()
        self._sweep_moved = False
        self._no_todo_t = None
        self._no_todo_cards = None
        # 一屏放得下的短地图拖不动会算进「拖不动」（复查 F2 之后又算了）：换页签 / 换章就清零，免得连着几张短地图就不再拖
        self._chapter_drag_fail = 0

    def _nav_run(self, fn, *args):
        """一整段换章动作：期间你一动键鼠就停（和借助战同一个关口），停止信号照常往外抛。"""
        prev = (getattr(self, '_sr_guarding', False), getattr(self, '_sr_guard_what', None))
        self._sr_guarding, self._sr_guard_what = True, '换章'
        try:
            return bool(fn(*args))
        except UserTakeoverError as e:
            pend = getattr(self, '_nav_claiming', None)
            if pend:
                # 领奖励领到一半：面板 / 弹窗是它自己点开的，你停手 CLAIM_RESUME_IDLE 秒它接着领（_maybe_resume_claim）；
                # 你一直不碰，停在「获得道具」上（认成「关卡外」）也算它卡住（I-57 的卡住提醒）
                self._nav_stopped_t = self._now()
                if pend.get('seen'):
                    tail = '收集数面板是我点开的，你 %d 秒不操作游戏我就接着领完' % CLAIM_RESUME_IDLE
                else:
                    # 面板还没看到（多半刚点了「可领取」、正在弹出来）：别说得像已经开着一样（复查 R4）
                    tail = '我刚点了「可领取」，面板要是开了，你 %d 秒不操作游戏我就接着领完' % CLAIM_RESUME_IDLE
                self.log_info('%s，先停下交给你 —— %s' % (e, tail), notify=True)
            else:
                self.log_info('%s，先停下交给你（界面停在哪，下一轮从那接着认）' % e, notify=True)
            return False
        finally:
            self._sr_guarding, self._sr_guard_what = prev

    def _nav_wait(self, pred, timeout=NAV_WAIT):
        """每秒看一眼，直到 pred(这一屏的 OCR) 成立；超时返回 None。"""
        end = self._now() + timeout
        while True:
            self.sleep(NAV_POLL)
            self.next_frame()
            boxes = self.ocr()
            if pred(boxes):
                return boxes
            if self._now() >= end:
                return None

    def _nav_fail(self, title, msg):
        self._nav_claiming = None
        mem = self._nav_memory(title)
        mem['fails'] += 1
        # 停在它自己点出来的画面上（「获得道具」弹窗之类，认成「关卡外」）：卡住提醒也得管（I-57，见 RealtimeAssistTask._stuck_mine）
        self._nav_stopped_t = self._now()
        self._nav_alert('自动换章停下：%s' % msg)
        try:
            self._save_stuck_image('nav')
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            pass
        if mem['fails'] >= NAV_FAIL_MAX and not mem['disabled']:
            mem['disabled'] = True
            self.log_info('「%s」自动换章连着失败 %d 次，这次运行里不再试，交给你' % (title, mem['fails']),
                          notify=True)
        return False

    @staticmethod
    def _nav_find(boxes, text, band=None, W=None, H=None):
        return [b for b in boxes if getattr(b, 'name', None) == text
                and (band is None or _in_band(b, band, W, H))]

    def _nav_click(self, box, name=None):
        self.click_box(box, after_sleep=0.8 + min(getattr(self, 'ui_delay', 0.8), 0.8))

    # ---------------------------------------------------------------- 1. 切页签

    def _nav_switch_tab(self, info, cur, other, verdict):
        title = info['title']
        mem = self._nav_memory(title)
        self.log_info('「%s」的「%s」页签%s，切到「%s」页签接着打' % (
            title, cur, '打完了' if verdict == 'done' else '剩下的关卡都锁着', other), notify=True)
        if getattr(self, '_claim_rewards_first', True) and info['claimable'] is not None:
            # 这一页的收集奖励先领了再走（复查 F8：以前只领离开时那一页的）
            if not self._nav_claim(info):
                return False
            self.next_frame()
            info = event_map_info(self.ocr(), self.width, self.height) or info
        self._nav_click(info['tabs'][other])
        W, H = self.width, self.height

        def lit(boxes):
            i = event_map_info(boxes, W, H)
            return i is not None and self._nav_selected_tab(i) == other
        if self._nav_wait(lit) is None:
            return self._nav_fail(title, '点了「%s」页签，%d 秒内没看到它亮起来' % (other, NAV_WAIT))
        mem['switches'] += 1                    # 真切过去了才算一次（复查 NAV-S9）
        self._nav_reset_map_memory()
        return True

    # ---------------------------------------------------------------- 2. 领收集奖励

    def _nav_panel_buttons(self, boxes):
        """收集数面板开着：返回还能点的「领取」；没开着返回 None。"""
        W, H = self.width, self.height
        if not self._nav_find(boxes, REWARD_PANEL_MARK):
            return None
        return self._nav_find(boxes, REWARD_BTN, REWARD_BTN_BAND, W, H)

    def _nav_claim(self, info, on_map=None):
        """点「可领取」、把面板上的「领取」都点掉、关面板。on_map(boxes)：关完面板回到了哪种地图（默认往期活动的章节地图）。

        领的途中记着「正在领哪一章」（self._nav_claiming）：你接手了（UserTakeoverError）就留着，
        你停手之后 _maybe_resume_claim 接着领完；领完 / 失败就清掉。"""
        title = info['title']
        self.log_info('「%s」左下角收集数「可领取」，先把奖励领了' % title)
        self._nav_claiming = dict(title=title, t=self._now(), on_map=on_map, resumes=0)
        try:
            panel = self._nav_open_panel(info)
            if panel is None:
                return self._nav_fail(title, '点了「可领取」，没看到收集数面板')
            self._nav_claiming['seen'] = True
            ok = self._nav_claim_rows(title, panel, on_map)
        except UserTakeoverError:
            raise
        except BaseException:
            self._nav_claiming = None
            raise
        self._nav_claiming = None
        return ok

    def _nav_open_panel(self, info):
        """点「可领取」等收集数面板。没开、画面还是那张地图（没弹别的）就再点，最多 CLAIM_OPEN_TRIES 次（I-63）。
        :return: 面板开着的那一屏的 OCR / None"""
        W, H = self.width, self.height
        btn = info['claimable']
        for i in range(CLAIM_OPEN_TRIES):
            self._nav_click(btn)
            last = i + 1 >= CLAIM_OPEN_TRIES
            panel = self._nav_wait(lambda b: self._nav_panel_buttons(b) is not None,
                                   timeout=NAV_WAIT if last else CLAIM_OPEN_WAIT)
            if panel is not None or last:
                return panel
            self.next_frame()
            fresh = self.ocr()
            if self._nav_panel_buttons(fresh) is not None:
                return fresh
            _, again = map_counter(fresh, W, H)
            if again is None or any(MODAL_RE.search(b.name or '') for b in fresh):
                return None              # 画面变了 / 弹了别的：不再点
            btn = again
            self.log_info('点了「可领取」%.0f 秒没打开收集数面板（那一下游戏多半没收到），再点一次' % CLAIM_OPEN_WAIT)
        return None

    def _nav_claim_rows(self, title, panel, on_map=None):
        """面板开着：把「领取」挨个点掉（每个关掉「获得道具」），再关面板。"""
        for _ in range(REWARD_MAX):
            btns = self._nav_panel_buttons(panel)
            if not btns:
                break
            self._nav_click(btns[0])
            n = len(btns)

            def claimed(b):
                left = self._nav_panel_buttons(b)        # 面板上还剩几个「领取」；面板被挡住时是 None
                return bool(self._nav_find(b, POPUP_CLOSE)) or (left is not None and len(left) < n)
            got = self._nav_wait(claimed)
            if got is None:
                return self._nav_fail(title, '点了「领取」，没看到「获得道具」')
            close = self._nav_find(got, POPUP_CLOSE)
            if close:
                got = self._nav_close_popup(close[0])
                if got is None:
                    return self._nav_fail(title, '关了「获得道具」，没回到收集数面板')
            panel = got
        else:
            if self._nav_panel_buttons(panel):
                return self._nav_fail(title, '「领取」点了 %d 次还在' % REWARD_MAX)
        return self._nav_close_panel(title, on_map)

    def _maybe_resume_claim(self, boxes):
        """领收集奖励领到一半你接手了（I-63，10/14 11:42）：面板 / 「获得道具」是它自己点开的，以前下一轮被
        「有弹窗就别动」的关口拦住，什么都不做，只能等你来领。你 CLAIM_RESUME_IDLE 秒没操作游戏了就接着领完。

        :return: None = 不归这里管（主循环照常往下走）；False = 是它开的、还在等你停手（这一轮别的都不做，
                 免得在它自己开的面板上提醒你「关掉弹窗」）；True = 接着领了
        """
        pend = getattr(self, '_nav_claiming', None)
        if not pend:
            return None
        now = self._now()
        if now - pend['t'] > CLAIM_RESUME_MAX or pend.get('resumes', 0) >= CLAIM_RESUME_TIMES:
            self._nav_claiming = None
            return None
        W, H = self.width, self.height
        panel = self._nav_panel_buttons(boxes)
        popup = self._nav_find(boxes, POPUP_CLOSE) if panel is None else []
        if panel is None and not popup:
            if event_map_info(boxes, W, H) is not None or story_map_title(boxes, W, H) is not None:
                self._nav_claiming = None       # 回到地图了（你领完 / 关掉了）：之后再开的面板不是它的
            return None
        if panel is not None and not self._claim_panel_ours(pend, boxes):
            return None                         # 开着的面板不是这一章的：不碰
        touch = pend.get('touch', pend['t'])
        ops = self._user_ops_since(pend['t']) if hasattr(self, '_user_ops_since') else None
        if ops:
            touch = max([touch] + [t for t, _ in ops])
        elif ops is None:
            seen = [getattr(self, '_stuck_user_t', None)]
            recent = self._user_recent_seconds() if hasattr(self, '_user_recent_seconds') else None
            if recent is not None:
                seen.append(now - recent)
            touch = max([touch] + [t for t in seen if t is not None])
        pend['touch'] = touch
        if now - touch < CLAIM_RESUME_IDLE:
            return False
        pend['resumes'] = pend.get('resumes', 0) + 1
        self.log_info('收集数面板是我刚才点开的，你 %d 秒没%s了：接着把「%s」的奖励领完（要自己来就点一下游戏画面，我会再停）'
                      % (CLAIM_RESUME_IDLE, '操作游戏' if ops is not None else '碰游戏', pend['title']), notify=True)
        self._nav_stopped_t = None
        return self._nav_run(self._nav_resume_claim, pend, boxes)

    def _claim_forget_tick(self, boxes):
        """主循环每一轮都看（在「你在动」的关口之前，只看不点）：记着「正在领」的时候，连着两帧既不是收集数面板、
        也不是「获得道具」，就忘掉 —— 你已经领完 / 关掉 / 走开了，之后再看到的面板、弹窗不是它的（I-63 复查）。"""
        pend = getattr(self, '_nav_claiming', None)
        if not pend:
            return
        if self._nav_panel_buttons(boxes) is not None or self._nav_find(boxes, POPUP_CLOSE):
            pend['away'] = 0
            return
        pend['away'] = pend.get('away', 0) + 1
        if pend['away'] >= 2:
            self._nav_claiming = None

    def _claim_panel_ours(self, pend, boxes):
        """开着的收集数面板是不是它正在领的那一章的（面板盖在地图上，顶上的章节名还读得到）。"""
        on_map = pend.get('on_map')
        if on_map is not None:
            try:
                if on_map(boxes):
                    return True
            except STOP_EXCEPTIONS:
                raise
            except Exception:
                pass
        return any(same_title(b.name, pend['title']) for b in boxes if getattr(b, 'name', None))

    def _nav_resume_claim(self, pend, boxes):
        """接着领：开着「获得道具」就先关掉，回到面板上把剩下的「领取」点完、关面板。"""
        title, on_map = pend['title'], pend.get('on_map')
        self._nav_claiming = dict(pend, t=self._now(), touch=self._now(), seen=True, away=0)  # 再被你打断就从这一刻重新算
        try:
            if self._nav_panel_buttons(boxes) is None:
                close = self._nav_find(boxes, POPUP_CLOSE)
                boxes = self._nav_close_popup(close[0]) if close else None
                if boxes is None:
                    return self._nav_fail(title, '接着领：关了「获得道具」，没回到收集数面板')
                # 关完弹窗回到的面板也要是这一章的（复查 R2：你中途去别的章自己领、弹出来的「获得道具」它分不出来）。
                # 刚关完的那一帧顶上的章节名可能还没读出来，再看两帧
                for _ in range(2):
                    if self._claim_panel_ours(pend, boxes):
                        break
                    self.sleep(NAV_POLL)
                    self.next_frame()
                    boxes = self.ocr()
                else:
                    if not self._claim_panel_ours(pend, boxes):
                        self.log_info('关掉「获得道具」之后回到的不是「%s」的收集数面板，不是我开的，不接着领了' % title)
                        self._nav_claiming = None
                        return False
            ok = self._nav_claim_rows(title, boxes, on_map)
        except UserTakeoverError:
            raise
        except BaseException:
            self._nav_claiming = None
            raise
        self._nav_claiming = None
        return ok

    def _nav_close_popup(self, close):
        """关「获得道具」：等它弹稳再点「点击空白处关闭」；点完还开着就再点（最多 POPUP_CLOSE_TRIES 次，I-57）。
        :return: 回到收集数面板的那一屏的 OCR / None（弹窗关不掉、或者关了却没回到面板）"""
        def back(b):
            return self._nav_panel_buttons(b) is not None
        for i in range(POPUP_CLOSE_TRIES):
            self.sleep(POPUP_SETTLE)
            self._nav_click(close)
            end = self._now() + (NAV_WAIT if i == POPUP_CLOSE_TRIES - 1 else POPUP_CLOSE_WAIT)
            still = []
            while True:
                self.sleep(NAV_POLL)
                self.next_frame()
                boxes = self.ocr()
                if back(boxes):
                    return boxes
                still = self._nav_find(boxes, POPUP_CLOSE)
                if self._now() >= end:
                    break
            if not still:
                return self._nav_wait(back)          # 弹窗没了、面板还没回来：再等一会儿，不对着认不出的画面乱点
            close = still[0]
            if i + 1 < POPUP_CLOSE_TRIES:
                self.log_info('「获得道具」还开着（刚才那下游戏没收到），再点一次「点击空白处关闭」')
        return None

    def _nav_close_panel(self, title, on_map=None):
        """关收集数面板：先点右上角认出来的叉，认不出就按返回键。"""
        frame = self.frame
        hit = None
        if close_x is not None and frame is not None and frame.shape[1] >= 640:
            try:
                hits = close_x.detect_close_x(frame, band=REWARD_X_BAND)
            except STOP_EXCEPTIONS:
                raise
            except Exception:
                hits = []
            hit = hits[0] if hits else None
        if hit is not None:
            x, y = hit['x'] / float(frame.shape[1]), hit['y'] / float(frame.shape[0])
            self.click_relative(x, y, after_sleep=0.8 + min(getattr(self, 'ui_delay', 0.8), 0.8),
                                name='收集数面板的叉')
        else:
            self.back()
            self.sleep(0.8 + min(getattr(self, 'ui_delay', 0.8), 0.8))   # 分开睡：等的时候你动了手也看得到（复查 NAV-S10）
        W, H = self.width, self.height
        if on_map is None:
            def on_map(b):
                return event_map_info(b, W, H) is not None
        back = self._nav_wait(lambda b: not self._nav_find(b, REWARD_PANEL_MARK) and on_map(b))
        if back is None:
            return self._nav_fail(title, '收集数面板关不掉')
        return True

    # ---------------------------------------------------------------- 3. 回往期活动、选下一章

    def _nav_leave_chapter(self, info):
        title = info['title']
        W, H = self.width, self.height
        if getattr(self, '_claim_rewards_first', True) and info['claimable'] is not None:
            if not self._nav_claim(info):
                return False
            self.next_frame()
            again = event_map_info(self.ocr(), W, H)
            if again is None:
                return self._nav_fail(title, '领完奖励没回到章节地图')
        if not self._nav_back_arrow_ok():
            return self._nav_fail(title, '左上角没认出「<」，不点')
        self.log_info('「%s」打完了，点左上角「<」回「往期活动」选下一章' % title, notify=True)
        self.click_relative(*BACK_ARROW, after_sleep=0.8 + min(getattr(self, 'ui_delay', 0.8), 0.8),
                            name='章节地图左上角「<」')
        boxes = self._nav_wait(lambda b: hub_info(b, W, H) is not None)
        if boxes is None:
            return self._nav_fail(title, '点了「<」，%d 秒内没看到「往期活动」页面' % NAV_WAIT)
        return self._nav_pick_next(boxes, title)

    def _nav_hub_drag(self, boxes, direction):
        """往期活动的卡片横着拖一下。direction=+1 看右边（更新的活动）。-> (动了吗, 拖完这一屏的 OCR)"""
        W, H = self.width, self.height
        before = {norm_title(c['title']): c['x'] for c in (hub_info(boxes, W, H) or {}).get('cards', [])}
        x0, x1 = HUB_DRAG_X if direction > 0 else HUB_DRAG_X[::-1]
        self._drag(x0, HUB_DRAG_Y, x1, HUB_DRAG_Y, key='left')
        self.sleep(0.5 + min(getattr(self, 'ui_delay', 0.8), 0.8))
        self.next_frame()
        boxes2 = self.ocr()
        hub = hub_info(boxes2, W, H)
        if hub is None:
            return False, boxes2
        after = {norm_title(c['title']): c['x'] for c in hub['cards']}
        common = set(before) & set(after)
        if common:
            shift = float(np.median([after[k] - before[k] for k in common]))
            return abs(shift) >= HUB_MOVED * W, boxes2
        return bool(after) and set(after) != set(before), boxes2

    def _nav_pick_next(self, boxes, title):
        """在往期活动里找到刚打完的「title」那张卡，点它右边第一张完成度不到 100% 的。

        这一页有时停在刚打的那张附近，有时停在最左边：先往右拖着找，拖到头还没见到就往左找；
        见到之后，它右边这一屏没有就接着往右拖。"""
        W, H = self.width, self.height
        direction, turned, passed, skipped = +1, False, False, []
        for _ in range(2 * HUB_DRAG_MAX + 2):
            hub = hub_info(boxes, W, H)
            if hub is None:
                return self._nav_fail(title, '往期活动页面不见了')
            cards = hub['cards']
            idx = next((i for i, c in enumerate(cards) if same_title(c['title'], title)), None)
            if idx is not None:
                passed = True
                right = cards[idx + 1:]
            elif passed and direction > 0:
                right = cards               # 刚打完的那张已经拖到左边外面了
            else:
                right = None
            if right is not None:
                for c in right:
                    if c['pct'] is None or c['pct'] < 100:
                        return self._nav_enter(c, title, skipped)
                    if c['title'] not in skipped:
                        skipped.append(c['title'])
                if direction < 0:
                    direction = +1          # 往左找到了它，右边一屏没有：回头往右
            moved, boxes = self._nav_hub_drag(boxes, direction)
            if moved:
                continue
            if not passed and not turned:
                direction, turned = -1, True
                continue
            break
        if not passed:
            return self._nav_fail(title, '往期活动里左右都找过了，没见到刚打完的「%s」' % title)
        return self._nav_fail(title, '往期活动里「%s」右边没有没打完的了%s' % (
            title, '（%s 都是 100%%）' % '、'.join(skipped) if skipped else ''))

    def _nav_enter(self, card, prev, skipped):
        W, H = self.width, self.height
        self.log_info('往期活动里「%s」右边%s下一个没打完的是「%s」（完成度 %s），点开' % (
            prev, '（跳过已经 100%% 的 %s）' % '、'.join(skipped) if skipped else '',
            card['title'], '%d%%' % card['pct'] if card['pct'] is not None else '读不出'), notify=True)
        self._nav_click(card['box'])

        def arrived(boxes):
            i = event_map_info(boxes, W, H)
            return i is not None and i['title'] is not None and same_title(i['title'], card['title'])
        if self._nav_wait(arrived, NAV_ENTER_WAIT) is None:
            return self._nav_fail(prev, '点了「%s」，%d 秒内没进到它的章节地图' % (card['title'], NAV_ENTER_WAIT))
        self._nav_reset_map_memory()
        self.log_info('进了「%s」，接着自动选关' % card['title'], notify=True)
        return True

    # ---------------------------------------------------------------- 5. 主线：打完一章，从章节列表进下一章

    def _note_story_map(self, boxes):
        """在主线章节地图上：记下是哪一章（顶上「/5.0异频共振」）。游戏打完一章自己退回章节列表时，拿它对上刚打完的是哪一章。"""
        t = story_map_title(boxes, self.width, self.height)
        if t:
            self._story_map = (t[0], t[1], self._now())

    def _story_frame(self):
        return self.frame

    def _story_info(self, boxes):
        return story_list_info(boxes, self.width, self.height, self._story_frame())

    def _story_arrived(self, boxes, title, code=None):
        t = story_map_title(boxes, self.width, self.height)
        return t is not None and _same_story(t[0], t[1], code, title)

    def _maybe_story_next(self, boxes):
        """「剧情战役」章节列表：刚自动打完一章的最后一关、游戏自己退回到这里，先点「确认」回这一章的地图接着选关。

        游戏退回列表 ≠ 这一章打完了：主线最后一关打完它就退回来，支线（SL / BH …）可能还没打 —— 10/11 12:46 用「继续前进」
        从 9-8 连打到 9-10，退回列表就直接进了 10.0，9.0 地图上 4 秒前还是「可打」的 SL-9-4、BH-9-4 没打（I-37）。
        回到地图上：还有没打的照常选；真打完了，「没有可推的」那条（_maybe_story_leave）会拖一遍整张图确认、点「<」进下一章。

        你自己点过来的（打完之后动过键鼠）、刚打开任务、选中的不是刚才在打的那一章：都不动。
        :return: True = 回到了这一章的地图
        """
        if not getattr(self, '_auto_next_chapter', True):
            return False
        info = self._story_info(boxes)
        if info is None:
            return False
        now = self._now()
        end = getattr(self, '_stage_end', None)
        smap = getattr(self, '_story_map', None)
        if end is None or end[1] not in ('victory', 'left') or now - end[0] > STORY_END_FRESH:
            return False                # 不是刚自动打完一关就到了这里
        if smap is None or now - smap[2] > STORY_MAP_FRESH:
            return False
        code, title = smap[0], smap[1]
        if getattr(self, '_story_user_end', None) == end[0]:
            return False                # 这一关打完之后你操作过：刚才说过不自动进了，这次也不进
        ops_fn = getattr(self, '_user_ops_since', None)
        ops = ops_fn(end[0]) if ops_fn else None
        if ops is not None:
            touched = bool(ops)         # input_watch 记着的：游戏切到后台也看得见你刚才操作过（I-60）
        else:
            user_t = self._user_recent_seconds()
            touched = user_t is not None and user_t < now - end[0]
        if touched:
            self._story_user_end = end[0]
            self._nav_note(title, 'story_user', '回到了「剧情战役」章节列表，但上一关打完之后你动过键鼠，'
                                                '不自动进下一章（你在操作）')
            return False
        mem = self._nav_memory(title)
        if mem['disabled']:
            return False
        sel, nxt = info['selected'], info['next']
        if sel is None:
            self._nav_note(title, 'story_sel', '回到了「剧情战役」章节列表，但认不出选中的是哪一章，不自动进下一章',
                           notify=True)
            return False
        if not _same_story(sel['code'], sel['title'], code, title):
            self._nav_note(title, 'story_other', '回到了「剧情战役」章节列表，选中的是「%s」，不是刚才在打的「%s」，'
                                                 '不自动进下一章' % (sel['title'], title), notify=True)
            return False
        # 动手前重新看一眼：还是这一页、选中的还是它（这一轮开头那一屏可能已经过时了）
        self.next_frame()
        info2 = self._story_info(self.ocr())
        if (info2 is None or info2['selected'] is None
                or not _same_story(info2['selected']['code'], info2['selected']['title'], code, title)
                or info2.get('confirm') is None):
            return False
        return self._nav_run(self._story_reenter, info2, title, code)

    def _story_reenter(self, info, title, code=None):
        """游戏退回了章节列表、选中的还是刚打完的这一章：点「确认」回它的地图（见 _maybe_story_next）。"""
        self._stage_end = None          # 这次退回列表只处理一次
        self.log_info('主线「%s」的最后一关打完了，游戏退回了「剧情战役」章节列表：先回这一章的地图，'
                      '还有没打的关接着打，真打完了再进下一章' % title, notify=True)
        # 列表刚退回来那几秒点了不响应（10/12 00:24:52 点的那下列表没动，I-42）：没回到地图就再看一眼、还是这一章再点一次
        for attempt in range(STORY_REENTER_TRIES):
            self._nav_click(info['confirm'])
            if self._nav_wait(lambda b: self._story_arrived(b, title, code), STORY_REENTER_WAIT) is not None:
                break
            self.next_frame()
            info = self._story_info(self.ocr())
            if (info is None or info['selected'] is None or info.get('confirm') is None
                    or not _same_story(info['selected']['code'], info['selected']['title'], code, title)):
                return self._nav_fail(title, '选中「%s」点了「确认」，%d 秒内没回到它的章节地图，列表也变了'
                                      % (title, STORY_REENTER_WAIT))
            if attempt + 1 < STORY_REENTER_TRIES:
                self.log_info('点了「确认」没进去，再点一次')
        else:
            return self._nav_fail(title, '选中「%s」点了 %d 次「确认」都没回到它的章节地图' % (title, STORY_REENTER_TRIES))
        self._nav_reset_map_memory()
        self.log_info('回到了「%s」的地图，接着自动选关' % title)
        return True

    def _story_enter_next(self, info, title):
        nxt = info['next']
        self._stage_end = None          # 这次退回列表只处理一次
        self.log_info('主线「%s」打完了，回到了「剧情战役」章节列表：选下一章「%s%s」，点「确认」进去' % (
            title, nxt['code'] + ' ' if nxt['code'] else '', nxt['title']), notify=True)
        self._nav_click(nxt['box'])

        def picked(boxes):
            if self._story_arrived(boxes, nxt['title'], nxt['code']):
                return True             # 点一下就直接进去了
            i = self._story_info(boxes)
            return (i is not None and i['selected'] is not None
                    and _same_story(i['selected']['code'], i['selected']['title'], nxt['code'], nxt['title']))
        got = self._nav_wait(picked)
        if got is None:
            return self._nav_fail(title, '点了「%s」，%d 秒内没看到它被选中' % (nxt['title'], NAV_WAIT))
        if not self._story_arrived(got, nxt['title'], nxt['code']):
            self._nav_click(self._story_info(got)['confirm'])
            if self._nav_wait(lambda b: self._story_arrived(b, nxt['title'], nxt['code']), NAV_ENTER_WAIT) is None:
                return self._nav_fail(title, '选中「%s」点了「确认」，%d 秒内没进到它的章节地图' % (
                    nxt['title'], NAV_ENTER_WAIT))
        self._nav_reset_map_memory()
        self._story_map = None
        self.log_info('进了「%s」，接着自动选关' % nxt['title'], notify=True)
        return True

    # ---------------------------------------------------------------- 6. 主线：停在打完的章节地图上，回列表进下一章

    def _maybe_story_leave(self, title, code=None):
        """主线章节地图上刚下了「没有可推的」：整张地图都打完了就点「<」回章节列表进下一章；剩下的锁着就说清楚、交给你。"""
        W, H = self.width, self.height
        self.next_frame()
        fresh = self.ocr()
        t2 = story_map_title(fresh, W, H)
        if t2 is None or not _same_story(t2[0], t2[1], code, title) or any(MODAL_RE.search(b.name or '') for b in fresh):
            return False                # 画面已经不是这一章的地图了，或者开着弹窗
        mem = self._nav_memory(title)
        if mem['disabled']:
            return False
        # 左下角收集数「可领取」：先领了（10/11 你说「把主线领奖励加上」；面板和往期活动的一样）。打没打完都领 ——
        # 剩下的锁着、拿不准打完没有的时候也领。领完要回到这一章的地图才接着往下判
        counter0, claimable = map_counter(fresh, W, H)      # 领之前的收集数（领完那一屏读不出时用它）
        if getattr(self, '_claim_rewards_first', True):
            if claimable is not None:
                def on_map(b):
                    t = story_map_title(b, W, H)
                    return t is not None and _same_story(t[0], t[1], code, title)
                if not self._nav_run(self._nav_claim, dict(title=title, claimable=claimable), on_map):
                    return False
                self.next_frame()
                fresh = self.ocr()
                if not on_map(fresh) or any(MODAL_RE.search(b.name or '') for b in fresh):
                    return False
        cards = getattr(self, '_no_todo_cards', None) or ()
        seen = set(getattr(self, '_scroll_seen', None) or ()) | {(c['code'], c['state']) for c in cards}
        run = self._run_mem(title)          # 只算这一章的记忆：别的章节也有同样的编号（I-36）
        cleared = set(run['cleared'])
        edges = getattr(self, '_sweep_edges', None) if getattr(self, '_sweep_moved', False) else set()
        # 主线左下角的收集数（5.0 打完是 42/48）不随通关收满：没满说明不了什么。收满了、看到的又都打完了，就是打完了
        # （10/12 01:30「逆序矩阵」困难 15/15，往左那下没拖动、两头没都看到，以前判「说不准」停在那里，I-45）
        verdict = tab_verdict(seen, edges, cleared, None)
        counter = map_counter(fresh, W, H)[0] or counter0
        if (verdict == 'unsure' and counter and counter[1] > 0 and counter[0] >= counter[1]
                and tab_verdict(seen, {'left', 'right'}, cleared, None) == 'done'):
            self.log_info('主线「%s」左下角收集数 %d/%d 收满了，看到的关也都打完了，算整张地图打完了' % (title, counter[0], counter[1]))
            verdict = 'done'
        lost, dead = set(run['lost']), run['dead']
        stuck = sorted(set(c for c, _ in seen if c in lost or dead.get(c, 0) >= 2))
        locked = sorted(set(c for c, st in seen if st == 'locked' and c not in cleared and c not in stuck))
        if stuck:
            self._nav_note(title, 'story_lost', '主线「%s」有打输过 / 点不开的关（%s），这一章没打完，不自动换章 —— '
                                                '你处理完之后我接着打' % (title, '、'.join(stuck)), notify=True)
            return False
        if verdict == 'done':
            return self._nav_run(self._story_leave_chapter, title, code)
        if locked and verdict in ('blocked', 'unsure'):
            self._nav_note(title, 'story_blocked', '主线「%s」这一章还没打完：剩下的关卡都锁着（%s），多半要先提升指挥官等级'
                                                   '或满足别的解锁条件 —— 不自动换章，交给你' % (title, '、'.join(locked[:6])),
                           notify=True)
        elif verdict == 'unsure':
            # 以前只记一行日志、不提醒，停在那里像是卡住了（10/12 01:30，I-45）
            self._nav_note(title, 'story_unsure', '主线「%s」上没找到可打的，但地图两头没都拖到、判不出整张图打完没有，先不换章 —— '
                                                  '你看一眼：打完了就点左上角「<」选下一章，还有没打的就点开它' % title,
                           notify=True)
        return False

    def _story_leave_chapter(self, title, code=None):
        if not self._nav_back_arrow_ok():
            return self._nav_fail(title, '左上角没认出「<」，不点')
        self.log_info('主线「%s」整张地图都打完了，点左上角「<」回「剧情战役」章节列表选下一章' % title, notify=True)
        self.click_relative(*BACK_ARROW, after_sleep=0.8 + min(getattr(self, 'ui_delay', 0.8), 0.8),
                            name='章节地图左上角「<」')
        boxes = self._nav_wait(lambda b: self._story_info(b) is not None)
        if boxes is None:
            return self._nav_fail(title, '点了「<」，%d 秒内没看到「剧情战役」章节列表' % NAV_WAIT)
        info = self._story_info(boxes)
        sel, nxt = info['selected'], info['next']
        if sel is None or not _same_story(sel['code'], sel['title'], code, title):
            return self._nav_fail(title, '回到章节列表，选中的不是「%s」（读到的是「%s」）' % (
                title, sel['title'] if sel else '没认出选中的'))
        if nxt is None or nxt['unlocked'] is not True:
            self._nav_note(title, 'story_locked', '「%s」打完了，但下一章%s，不自动进 —— 交给你' % (
                title, '「%s」%s' % (nxt['title'], '还锁着' if nxt['unlocked'] is False else '认不出解锁了没有')
                if nxt else '在这一屏上没有'), notify=True)
            return False
        return self._story_enter_next(info, title)
