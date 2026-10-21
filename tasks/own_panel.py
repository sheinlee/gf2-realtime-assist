# -*- coding: utf-8 -*-
"""「我的人形」页签的面板读数 —— 纯函数，只用 numpy / cv2 / re，不 import ok。

输入：一帧 BGR 画面 + 整屏 OCR 框（对象带 name/x/y/width/height，或元组 (name, x, y, w, h[, conf])，
单位都是这一帧的像素）。所有位置都按屏宽 W / 屏高 H 的占比写，2560x1440、3072x1728、3200x1800
三种窗口以及缩到 0.6 倍都验过（tests/sim/own_panel_test.py）。

**判不准一律往安全的方向报**：
- 页签：下划线判不清就返回 None，调用方当「不是我的人形页」处理。
- 勾选框：判不清 / 勾选框那块被列表上沿裁掉 / 附近有说不清的橙色，一律报 checked=True、sure=False
  —— 点一张其实已出击的卡会把它撤下来（取消出击），所以「没出击」只在很确定时才报。
- 元素筛选：下划线和图标亮度两路都对得上才给序号，否则 None。

量出来的数（10/02、10/10 的 debug_tiles 现场图：49 张我的人形页 + 7 张好友助战页，另有 88 张别的界面，
1.0 倍和 0.6 倍各一遍）：
- 页签下划线：橙色（HSV 5~25 / S>=150 / V>=180），y=0.061~0.065H。当前页签那一行橙色占比 1.00（116 次），
  另一个页签 0.00；所有图里不该有下划线的位置最大 0.097（一张教程图）。阈值 >=0.50 算有、<=0.15 算没有。
- 元素筛选条（只在我的人形页，y=0.145~0.177H 是图标，0.183~0.186H 是下划线）：
  选中的图标是白的（V≈240），没选中的是灰的（V 最大 117）；下划线也是白的（V 240~249），条子底色中位数 34~53。
  每格下划线列占比：选中 1.00、其余 0.00；图标白像素占比：选中 0.156~0.218、其余 0.000。
  好友助战页这一带是卡片立绘，底色中位数 140~199，直接判 None。
  图标中心 x=0.7906W + k*0.0388W（k=0 是「全部」，k=5 只露出半个，条子右边缘 0.9845W）。
- 勾选框：橙色方块，中心 = 作战效能数字框中心 + (0.0226~0.0233W, -0.1075~-0.1103H)，
  0.0130W x 0.0231H。0.016W 见方窗口里橙色占比：勾上 0.511~0.544（0.6 倍 0.486~0.547），没勾 0.000
  （244 / 819 次）；窗口周围一圈的橙色面积比：没勾 <=0.014。右边那张卡的火属性角标在 +0.0475W（宽 0.005W），
  离窗口右边 0.016W、离周围一圈的右边 0.008W，不会混进来。

对抗验证后补的四处：
0. 「没勾」的门槛收紧：窗口橙色 <= 0.01（原 0.05）、附近 <= 0.04（原 0.10）。真图没勾的卡在 1.0/0.6/0.5/0.4 倍
   和 4K 下 frac 全是 0.000、near 最大 0.014；旧门槛下勾选框从上往下被挡住 85% 就会读成「确定没勾」。
1. 列表淡入 / 只压暗列表（页签下划线还亮着）时，橙色勾的 V 掉到 180 以下就读成「确定没勾」。
   现在报「没勾」还要求这张卡的作战效能数字够白：数字框内 V 的 95 分位 >= 215
   （实测 1.0 / 0.6 / 0.5 / 0.4 倍和 4K 下最低 240 / 238 / 239 / 233.8 / 242）。列表往面板底色（V≈35）淡时，
   数字在 alpha <= 0.86~0.87 就不够白，勾在 alpha <= 0.70~0.71 才读丢；纯压暗时是 0.88~0.89 对 0.74~0.75。
2. 元素筛选条被横向拖过 0.7~1.25 格时，「物理」的亮图标 + 下划线落进第 0 格，旧版报 0（全部）。
   现在第 0 格必须真是「全部」那个图标：四个菱形（上/下/左/右各一块，4 连通恰好 4 块、面积比 <= 2），
   整体框中心离 0.7906W 不超过 0.003W。实测 53 张图 x 0.33~1.5 倍全部通过，别的图标都只有 1~3 块。
   不是的话 index=None。
3. 传进来的 W/H 和 frame 尺寸对不上（>1%）时，旧版按传进来的 W/H 算位置，勾选框窗口会落到别处。
   现在页签 / 筛选判 None，卡片一律 checked=True、sure=False，read_own_panel 报 size_mismatch。
"""
import re

import cv2
import numpy as np

# ---------------------------------------------------------------- 文字
POWER_RE = re.compile(r'^\d{3,7}$')             # 卡片下方的「作战效能」
MY_TAB_RE = re.compile('我的人形')
SUPPORT_TAB_RE = re.compile('好友助战')
# 页签下面的下拉框：正常是「使用整备室配置」；选了编队预设会变成「编队1」（OCR 常读成「1编队1」）。
# **这个下拉框绝对不能点**：10/02 08:17 点到它把阵容整个换掉了（3/4 -> 0/4）。
DROPDOWN_PRESET_RE = re.compile('整备室|使用整备|备室配置')
DROPDOWN_FORMATION_RE = re.compile(r'编队\s*\d')
EMPTY_RE = re.compile('暂无可选|可选择人形')

# ---------------------------------------------------------------- 位置（屏宽 / 屏高占比）
PANEL_X0 = 0.76                                 # 右侧人形面板左边缘（往左是地图）
TAB_Y1 = 0.085                                  # 页签文字框中心在这之上
MY_TAB_XY = (0.813, 0.041)                      # 页签文字中心（实测 0.813~0.814 / 0.040~0.041）
SUPPORT_TAB_XY = (0.902, 0.042)
TAB_XY_TOL = 0.03                               # OCR 框中心离上面的位置超过它就不认
# 页签下划线：只在这一条里找。x 取页签底色条的中间一段，两头留余量。
TAB_LINE_Y = (0.052, 0.075)
MY_TAB_LINE_X = (0.785, 0.845)
SUPPORT_TAB_LINE_X = (0.875, 0.930)
TAB_LINE_HSV = ((5, 150, 180), (25, 255, 255))  # 与 support_pick.TAB_UNDERLINE_HSV_* 同一口径
TAB_LINE_ON = 0.50
TAB_LINE_OFF = 0.15

DROPDOWN_BAND = (0.072, 0.135)                  # 下拉框所在的高度（文字中心实测 0.101~0.102H）
DANGER_ZONE = (PANEL_X0, 0.070, 1.0, 0.137)     # 这个矩形里一下都不要点

GRID_Y = (0.25, 0.86)                           # 作战效能锚点的中心只认这一段
COL_X = (0.806, 0.879, 0.951)                   # 三列锚点中心（实测 0.804~0.808 / 0.878~0.879 / 0.950~0.952）
COL_TOL = 0.02
ROW_GAP = 0.05                                  # 锚点 y 相差超过它就是下一行（行距 0.156H）
ANCHOR_H = (0.008, 0.045)                       # 数字框高度（实测 1.0 倍 0.020~0.033H，0.6 倍低到 0.013H）
# 卡片列表的可见范围：第一行卡片上沿 ≈0.214H，最下面一行数字 ≈0.806H、卡片下沿 ≈0.82H，再往下是排序条。
# 列表往上滚过以后，顶上那行的勾选框会被裁掉 —— 勾选框没完整露出来就不下结论。
VIEW_Y = (0.200, 0.835)

PORTRAIT_DY = 0.065                             # 数字中心往上这么多是立绘（与 support_pick.CARD_PORTRAIT_DY 同）
CLICK_MIN_Y = 0.20                              # 点卡片的 y 不能比这更靠上（上面是筛选条 / 下拉框）

CHECK_DX, CHECK_DY = 0.023, -0.110              # 勾选框中心相对数字框中心
CHECK_HALF = 0.008                              # 取 0.016W 见方的窗口量橙色占比
CHECK_BOX = (0.0130, 0.0231)                    # 勾选框本身 W/H 占比
CHECK_HALF_H = 0.0125                           # 勾选框半高（H 占比，留了一点余量）
CHECK_SEARCH = (0.014, 0.025)                   # 窗口附近再看一圈有没有说不清的橙色
CHECK_ON = 0.30                                 # 窗口橙色占比 >= 它：勾上
CHECK_OFF = 0.01                                # <= 它且附近也干净：没勾（fixed：0.05->0.01，真图没勾的全是 0.000）
CHECK_NEAR_OFF = 0.04                           # 附近橙色面积 / 勾选框面积 超过它就不敢说没勾
                                                # （fixed：0.10->0.04；真图没勾最大 0.014。旧值下勾选框从上面被挡 85% 就读成没勾）

FILTER_X0 = 0.7906                              # 「全部」图标中心
FILTER_PITCH = 0.0388
FILTER_SLOTS = 6                                # 能看到的格子（第 6 个只露出一半）
FILTER_X1 = 0.983                               # 筛选条右边缘（实测 0.9845W），再往右不算
FILTER_HALF = 0.012                             # 每格量下划线 / 图标亮度的半宽
FILTER_ICON_Y = (0.140, 0.178)
FILTER_LINE_Y = (0.179, 0.192)
FILTER_STRIP = (0.770, 0.980)                   # 量条子底色用的 x 范围
FILTER_CLICK_Y = 0.161
FILTER_WHITE_V = 170                            # 白：V>=170 且 S<=60（选中 V≈240，没选中最大 117）
FILTER_WHITE_S = 60
FILTER_LINE_ON = 0.60                           # 这一格里有下划线的列占比
FILTER_LINE_OFF = 0.15
FILTER_ICON_ON = 0.04                           # 这一格图标里白像素占比（选中实测 ~0.17）
FILTER_ICON_OFF = 0.02
FILTER_DARK_V = 100                             # 条子底色（中位数）要比它暗，否则不是这块条子
# 第 0 格「全部」图标核对（fixed 新增）
ALL_GLYPH_HALF = 0.0125                         # 只在 0.7906W ± 这么宽里找（图标本身宽 0.0137~0.0153W）
ALL_GLYPH_Y = (0.146, 0.181)                    # 图标 y 0.150~0.177H（中心 0.1630~0.1641H）
ALL_GLYPH_GREY_V = 95                           # 没选中时图标是灰的（V 100~117），底色 <= 90
ALL_GLYPH_CX_TOL = 0.003                        # 实测偏差 -0.0001~+0.0005W；拖动 0.1 格 = 0.0039W
ALL_GLYPH_CY = (0.158, 0.169)
ALL_GLYPH_W = (0.011, 0.020)                    # 实测 0.0137~0.0153W
ALL_GLYPH_H = (0.020, 0.034)                    # 实测 0.0243~0.0274H
ALL_GLYPH_AREA_RATIO = 2.0                      # 四块菱形面积最大 / 最小（实测 <= 1.28）
# 卡片数字亮度（fixed 新增）：报「没勾」时这张卡的作战效能数字必须够白，否则当作被压暗 / 淡入中
DIGIT_V_MIN = 215                               # 数字框 V 的 95 分位（实测 >= 238）
SIZE_TOL = 0.01                                 # 传进来的 W/H 与 frame 尺寸相差超过 1% 就不认


def _bx(b):
    """OCR 框 -> (name, x, y, w, h)。对象或元组都行。"""
    if isinstance(b, (tuple, list)):
        return str(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])
    return str(b.name), float(b.x), float(b.y), float(b.width), float(b.height)


def _wh(frame, W, H):
    if frame is not None:
        h, w = frame.shape[:2]
        return (W or w), (H or h)
    return W, H


def _size_mismatch(frame, W, H):
    """传进来的 W/H 和 frame 的像素尺寸对不上 -> True（位置会全算错，一律不下结论）。"""
    if frame is None:
        return False
    h, w = frame.shape[:2]
    return bool((W and abs(W - w) > SIZE_TOL * w) or (H and abs(H - h) > SIZE_TOL * h))


def _find_tab(boxes, regex, xy, W, H):
    best = None
    for b in boxes or ():
        name, x, y, w, h = _bx(b)
        if not regex.search(name):
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        if cy > TAB_Y1 or cx < PANEL_X0:
            continue
        # OCR 把页签和旁边的字并成一个框时，按字数把框裁到这 4 个字上
        m = regex.search(name)
        if len(name) > 4 and m:
            per = w / len(name)
            x, w = x + per * m.start(), per * (m.end() - m.start())
            cx = (x + w / 2) / W
        if abs(cx - xy[0]) > TAB_XY_TOL or abs(cy - xy[1]) > TAB_XY_TOL:
            continue
        d = abs(cx - xy[0]) + abs(cy - xy[1])
        if best is None or d < best[0]:
            best = (d, (name, int(x), int(y), int(w), int(h)))
    return best[1] if best else None


def own_panel_info(boxes, W, H):
    """只看 OCR：页签框、下拉框是哪种、列表是不是空的。

    返回 dict(dropdown='preset'|'formation'|None, empty=bool, my_tab_box, support_tab_box,
    dropdown_box)。框是 (name, x, y, w, h)，找不到是 None。
    dropdown：'preset' = 「使用整备室配置」，'formation' = 「编队N」（两个都读到时报 'formation'）。
    **dropdown 是 None 本身就说明这多半不是我的人形页**（好友助战页没有这个下拉框）。
    """
    info = dict(dropdown=None, dropdown_box=None, empty=False,
                my_tab_box=_find_tab(boxes, MY_TAB_RE, MY_TAB_XY, W, H),
                support_tab_box=_find_tab(boxes, SUPPORT_TAB_RE, SUPPORT_TAB_XY, W, H))
    for b in boxes or ():
        name, x, y, w, h = _bx(b)
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        if cx < PANEL_X0:
            continue
        if DROPDOWN_BAND[0] <= cy <= DROPDOWN_BAND[1]:
            kind = ('formation' if DROPDOWN_FORMATION_RE.search(name)
                    else 'preset' if DROPDOWN_PRESET_RE.search(name) else None)
            if kind and info['dropdown'] != 'formation':
                info['dropdown'], info['dropdown_box'] = kind, (name, int(x), int(y), int(w), int(h))
        if GRID_Y[0] - 0.05 <= cy <= GRID_Y[1] and EMPTY_RE.search(name):
            info['empty'] = True
    return info


def _orange(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(TAB_LINE_HSV[0], np.uint8), np.array(TAB_LINE_HSV[1], np.uint8)) > 0


def tab_line_fracs(frame, W=None, H=None):
    """两个页签下划线带里「最橙的那一行」的橙色占比 -> (我的人形, 好友助战)。"""
    W, H = _wh(frame, W, H)
    y0, y1 = int(TAB_LINE_Y[0] * H), int(TAB_LINE_Y[1] * H)
    out = []
    for x0, x1 in (MY_TAB_LINE_X, SUPPORT_TAB_LINE_X):
        sub = frame[y0:y1, int(x0 * W):int(x1 * W)]
        out.append(float(_orange(sub).mean(axis=1).max()) if sub.size else 0.0)
    return tuple(out)


def own_tab_active(frame, W=None, H=None):
    """True = 我的人形是当前页签；False = 好友助战是当前页签；None = 看不出来（当作不是）。"""
    if _size_mismatch(frame, W, H):
        return None
    my, sup = tab_line_fracs(frame, W, H)
    if my >= TAB_LINE_ON and sup <= TAB_LINE_OFF:
        return True
    if sup >= TAB_LINE_ON and my <= TAB_LINE_OFF:
        return False
    return None


def _digit_v(vch, anchor):
    """作战效能数字框里 V 的 95 分位（白字 ≈240；列表被压暗 / 淡入时会掉下去）。"""
    x, y, w, h = anchor
    sub = vch[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
    return float(np.percentile(sub, 95)) if sub.size else 0.0


def _check_state(org, acx, acy, W, H, digit_v=255.0):
    """一张卡的勾选框 -> (checked, sure, frac, near)。org 是整帧的橙色掩码。

    frac = 0.016W 见方窗口里的橙色占比；near = 窗口周围一圈（±0.014W, ±0.025H）橙色面积 / 勾选框面积。
    digit_v = 这张卡数字框的亮度；不够白（< DIGIT_V_MIN）就不报「没勾」。
    """
    ex, ey = acx + CHECK_DX * W, acy + CHECK_DY * H
    r = CHECK_HALF * W
    x0, x1, y0, y1 = int(round(ex - r)), int(round(ex + r)), int(round(ey - r)), int(round(ey + r))
    fh, fw = org.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > fw or y1 > fh:
        return True, False, 0.0, 0.0
    frac = float(org[y0:y1, x0:x1].mean())
    # 勾选框必须整块落在列表可见范围里，被裁掉一截就不下「没勾」的结论
    visible = (ey - CHECK_HALF_H * H >= VIEW_Y[0] * H) and (ey + CHECK_HALF_H * H <= VIEW_Y[1] * H)
    sx, sy = CHECK_SEARCH[0] * W, CHECK_SEARCH[1] * H
    near = org[max(0, int(ey - sy)):int(ey + sy), max(0, int(ex - sx)):int(ex + sx)]
    near_ratio = float(near.sum()) / max(1.0, CHECK_BOX[0] * W * CHECK_BOX[1] * H)
    if frac >= CHECK_ON:
        return True, bool(visible), frac, near_ratio
    if frac <= CHECK_OFF and near_ratio <= CHECK_NEAR_OFF and visible and digit_v >= DIGIT_V_MIN:
        return False, True, frac, near_ratio
    return True, False, frac, near_ratio


def own_cards(frame, boxes, W=None, H=None):
    """我的人形列表里看得见的卡片，按 (行, 列) 排好。

    每张：dict(power, anchor=(x,y,w,h), click=(x,y), checked, sure, frac, near, digit_v, row, col)
    - anchor 是「作战效能」数字的 OCR 框；click 是立绘上的点（数字中心往上 0.065H，但不高过 0.20H）。
    - checked=True 表示已出击（橙色勾），**判不准也报 True**，此时 sure=False。
      只有 checked=False 的卡才可以点（点已出击的卡会把它撤下来）。
    - row 是可见范围里的第几行（列表滚过以后不是绝对行号），col 是 0/1/2。
    **只读格子，不判断当前是不是我的人形页** —— 好友助战页的卡片也长这样，先用 own_tab_active 把关。
    """
    mismatch = _size_mismatch(frame, W, H)
    W, H = _wh(frame, W, H)
    org = _orange(frame)
    vch = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
    raw = []
    for b in boxes or ():
        name, x, y, w, h = _bx(b)
        name = name.strip()
        if not POWER_RE.match(name):
            continue
        cx, cy = x + w / 2, y + h / 2
        if cx < PANEL_X0 * W or not (GRID_Y[0] * H <= cy <= GRID_Y[1] * H):
            continue
        if not (ANCHOR_H[0] * H <= h <= ANCHOR_H[1] * H):
            continue
        dx = [abs(cx / W - c) for c in COL_X]
        col = int(np.argmin(dx))
        if dx[col] > COL_TOL:
            continue
        raw.append((cy, col, int(name), (int(x), int(y), int(w), int(h)), cx))
    raw.sort()
    cards, row, last_y = [], -1, None
    for cy, col, power, anchor, cx in raw:
        if last_y is None or cy - last_y > ROW_GAP * H:
            row += 1
            last_y = cy
        dv = _digit_v(vch, anchor)
        checked, sure, frac, near = _check_state(org, cx, cy, W, H, dv)
        if mismatch:
            checked, sure = True, False
        click = (int(round(cx)), int(round(max(cy - PORTRAIT_DY * H, CLICK_MIN_Y * H))))
        cards.append(dict(power=power, anchor=anchor, click=click, checked=checked, sure=sure,
                          frac=round(frac, 3), near=round(near, 3), digit_v=round(dv, 1), row=row, col=col))
    cards.sort(key=lambda c: (c['row'], c['col']))
    # 同一格读出两个数字：说明锚点认错了，这两张都不许点
    seen = {}
    for c in cards:
        seen.setdefault((c['row'], c['col']), []).append(c)
    for same in seen.values():
        if len(same) > 1:
            for c in same:
                c['checked'], c['sure'] = True, False
    return cards


def filter_scores(frame, W=None, H=None):
    """元素筛选条每一格的 (下划线列占比, 图标白像素占比)，以及条子底色亮度（中位数）。"""
    W, H = _wh(frame, W, H)
    hsv = cv2.cvtColor(frame[int(0.12 * H):int(0.20 * H), int(PANEL_X0 * W):W], cv2.COLOR_BGR2HSV)
    oy, ox = int(0.12 * H), int(PANEL_X0 * W)
    white = (hsv[..., 2] >= FILTER_WHITE_V) & (hsv[..., 1] <= FILTER_WHITE_S)
    ly0, ly1 = int(FILTER_LINE_Y[0] * H) - oy, int(FILTER_LINE_Y[1] * H) - oy
    iy0, iy1 = int(FILTER_ICON_Y[0] * H) - oy, int(FILTER_ICON_Y[1] * H) - oy
    need = max(1, int(round(0.0015 * H)))           # 下划线 ~0.0035H 厚，一列里至少这么多白像素才算
    line, icon = [], []
    for k in range(FILTER_SLOTS):
        c = FILTER_X0 + k * FILTER_PITCH
        a = int((c - FILTER_HALF) * W) - ox
        b = int(min(c + FILTER_HALF, FILTER_X1) * W) - ox
        if b - a < 3:
            line.append(0.0)
            icon.append(0.0)
            continue
        line.append(float((white[ly0:ly1, a:b].sum(axis=0) >= need).mean()))
        icon.append(float(white[iy0:iy1, a:b].mean()))
    sx0, sx1 = int(FILTER_STRIP[0] * W) - ox, int(FILTER_STRIP[1] * W) - ox
    v = hsv[..., 2]
    dark_icon = float(np.median(v[iy0:iy1, sx0:sx1]))
    dark_line = float(np.median(v[ly0:ly1, sx0:sx1]))
    return dict(line=line, icon=icon, strip_v=max(dark_icon, dark_line))


def all_glyph_check(frame, lit, W=None, H=None):
    """第 0 格是不是「全部」图标、而且没被横向拖开 -> (ok, info)。

    lit=True 用白色阈值（选中时 V≈240），False 用灰色阈值（没选中时 V 100~117）。
    「全部」= 四个菱形，上 / 下 / 左 / 右各一块；别的图标（物理六边形、火、电、冰、半个圆点）4 连通只有 1~3 块。
    """
    W, H = _wh(frame, W, H)
    a = int(round((FILTER_X0 - ALL_GLYPH_HALF) * W))
    b = int(round((FILTER_X0 + ALL_GLYPH_HALF) * W))
    y0, y1 = int(round(ALL_GLYPH_Y[0] * H)), int(round(ALL_GLYPH_Y[1] * H))
    hsv = cv2.cvtColor(frame[y0:y1, a:b], cv2.COLOR_BGR2HSV)
    thr = FILTER_WHITE_V if lit else ALL_GLYPH_GREY_V
    m = ((hsv[..., 2] >= thr) & (hsv[..., 1] <= FILTER_WHITE_S)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return False, dict(why='empty')
    tot = float(st[1:, 4].sum())
    mh, mw = m.shape
    big = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 0.08 * tot:
            continue
        if x == 0 or y == 0 or x + w >= mw or y + h >= mh:
            return False, dict(why='touches_border')       # 图标被拖到窗口边上，或者有别的亮东西
        big.append((float(cen[i][0]), float(cen[i][1]), int(area), (x, y, x + w, y + h)))
    if len(big) != 4:
        return False, dict(why='blobs=%d' % len(big))
    gx = sum(p[0] for p in big) / 4.0
    gy = sum(p[1] for p in big) / 4.0
    d = [(p[0] - gx, p[1] - gy) for p in big]
    dirs = (sum(1 for dx, dy in d if dy < -abs(dx)), sum(1 for dx, dy in d if dy > abs(dx)),
            sum(1 for dx, dy in d if dx < -abs(dy)), sum(1 for dx, dy in d if dx > abs(dy)))
    areas = [p[2] for p in big]
    bx0 = min(p[3][0] for p in big); by0 = min(p[3][1] for p in big)
    bx1 = max(p[3][2] for p in big); by1 = max(p[3][3] for p in big)
    cx, cy = (a + (bx0 + bx1) / 2.0) / W, (y0 + (by0 + by1) / 2.0) / H
    gw, gh = (bx1 - bx0) / float(W), (by1 - by0) / float(H)
    info = dict(dirs=dirs, area_ratio=round(max(areas) / float(min(areas)), 2), dcx=round(cx - FILTER_X0, 4),
                cy=round(cy, 4), w=round(gw, 4), h=round(gh, 4))
    ok = (dirs == (1, 1, 1, 1) and max(areas) <= ALL_GLYPH_AREA_RATIO * min(areas)
          and abs(cx - FILTER_X0) <= ALL_GLYPH_CX_TOL and ALL_GLYPH_CY[0] <= cy <= ALL_GLYPH_CY[1]
          and ALL_GLYPH_W[0] <= gw <= ALL_GLYPH_W[1] and ALL_GLYPH_H[0] <= gh <= ALL_GLYPH_H[1])
    if not ok:
        info['why'] = 'shape'
    return ok, info


def own_filter_state(frame, W=None, H=None, detail=False):
    """我的人形页元素筛选条上选中的是第几个 -> (index, all_click) [, scores]。

    index：0 = 「全部」，1 = 第二个图标……；看不准（条子不在、没有或不止一条下划线、
    下划线和亮图标对不上、第 0 格不是没拖开的「全部」图标）就是 None。all_click 是「全部」图标中心，
    **点之前先确认是我的人形页**，点完重读一次确认 index == 0。
    好友助战页选的筛选会带到我的人形页（10/10 三张 deploy_failed 都是第二个图标亮、只剩 3 张卡）。
    """
    mismatch = _size_mismatch(frame, W, H)
    if mismatch:
        W = H = None                                # 尺寸对不上：分数按 frame 自己的尺寸算，序号一律 None
    W, H = _wh(frame, W, H)
    sc = filter_scores(frame, W, H)
    all_click = (int(round(FILTER_X0 * W)), int(round(FILTER_CLICK_Y * H)))
    idx = None
    if sc['strip_v'] < FILTER_DARK_V and not mismatch:
        on = [k for k, f in enumerate(sc['line']) if f >= FILTER_LINE_ON]
        mid = [k for k, f in enumerate(sc['line']) if FILTER_LINE_OFF < f < FILTER_LINE_ON]
        if len(on) == 1 and not mid:
            k = on[0]
            lit = [j for j, f in enumerate(sc['icon']) if f >= FILTER_ICON_ON]
            dim_ok = all(f <= FILTER_ICON_OFF for j, f in enumerate(sc['icon']) if j != k)
            # 第 6 格只露出半个图标，亮像素少，放宽到「别的格子都不亮」即可
            if dim_ok and (lit == [k] or (k == FILTER_SLOTS - 1 and not lit)):
                idx = k
    if idx is not None:
        # 第 0 格必须是没被拖开的「全部」图标，否则格子编号不可信，all_click 也不一定点得到「全部」
        g_ok, g_info = all_glyph_check(frame, idx == 0, W, H)
        sc['all_glyph'] = g_info
        if not g_ok:
            idx = None
    if detail:
        return idx, all_click, sc
    return idx, all_click


def in_danger_zone(x, y, W, H):
    """(x, y) 落在下拉框那一块就返回 True —— 调用方点任何东西之前过一遍。"""
    x0, y0, x1, y1 = DANGER_ZONE
    return x0 * W <= x <= x1 * W and y0 * H <= y <= y1 * H


def read_own_panel(frame, boxes, W=None, H=None):
    """一次读全：dict(ok, reason, own_tab, filter, all_click, cards, **own_panel_info)。

    ok=True 才说明「这是我的人形页（下划线 + 页签文字 + 下拉框都对得上）、元素筛选是「全部」」；
    列表空不空另看 empty / cards，下拉框是不是编队预设另看 dropdown。ok=False 时 reason 说为什么：
    size_mismatch / not_own_tab / tab_unknown / no_my_tab_text / no_dropdown / filter_unknown / filter_not_all /
    empty_and_cards。
    """
    mismatch = _size_mismatch(frame, W, H)
    W0, H0 = W, H
    W, H = _wh(frame, W, H)
    info = own_panel_info(boxes, W, H)
    tab = own_tab_active(frame, W0, H0)
    fidx, all_click = own_filter_state(frame, W0, H0)
    cards = own_cards(frame, boxes, W0, H0)
    out = dict(info, own_tab=tab, filter=fidx, all_click=all_click, cards=cards, ok=False, reason='')
    if mismatch:
        out['reason'] = 'size_mismatch'
    elif tab is not True:
        out['reason'] = 'not_own_tab' if tab is False else 'tab_unknown'
    elif info['my_tab_box'] is None:
        out['reason'] = 'no_my_tab_text'
    elif info['dropdown'] is None:
        out['reason'] = 'no_dropdown'
    elif fidx is None:
        out['reason'] = 'filter_unknown'
    elif fidx != 0:
        out['reason'] = 'filter_not_all'
    elif info['empty'] and cards:
        out['reason'] = 'empty_and_cards'
    else:
        out['ok'] = True
    return out
