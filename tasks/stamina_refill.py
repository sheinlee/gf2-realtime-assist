# -*- coding: utf-8 -*-
"""「情报拼图补充」框：认出三格、读中间那格的数量、看橙色选中框在哪一格。只认不点（点在 stage_runner._sr_refill）。

你 10/11 说「情报拼图不够时中间有能补充的，点击确定补充」。框里横着三格（10/11 103107 / 104525 / 111517 和
10/1 的 5 张图上量的，2560 / 3200 两种宽度位置都一样）：

  左：调度室储存的情报拼图。默认选着它，数量经常是 0，这时点「确认」什么都不补（I-18 那次连点了 6 下）
  中：补充用的道具（这几张图上是 3 / 5 / 0 个）
  右：晶体货币（数字和右上角那个一样）—— **一下都不点**

三格的数量字中心在标题中心左右各 0.098W、标题顶往下 0.218H；选中的那格外面一圈橙色（外圈 25% 是橙色，别的格 0%）。
有没有货看数量字的颜色：0 个是红字，有货是白字（8 张：白字 2.8%～3.7%、红字 0；0 个的红字 4.6%～5%、白字 0）。
不靠 OCR 读数：格子上的小数字整屏 OCR 8 张漏了 4 张，单裁一块读也只读出 5 张（读出来的倒都对，只拿来写日志）。
"""
import re

import cv2
import numpy as np

REFILL_TITLE = '情报拼图补充'
SLOT_DX = 0.098                  # 相邻两格数量字中心的间距（×W）
COUNT_DY = 0.218                 # 数量字中心在标题顶往下多少（×H）
COUNT_WIN = (0.03, 0.025)        # 认数量字的范围：中心左右 0.03W、上下 0.025H（离隔壁格 0.098W，碰不到）
DIGIT_WIN = (0.025, 0.012)       # 看数量字颜色的那块：中心左右 0.025W、上下 0.012H
WHITE_MIN, RED_MIN = 0.012, 0.02  # 白字 / 红字像素至少占这么多（实测 ≥0.028 / ≥0.046）
OTHER_MAX = 0.005                # 另一种颜色不能超过这么多
DIGIT_MAX = 0.2                  # 字的笔画很细：白 / 红占得比这还多就不是数量字（那块整片发亮 = 认错了地方）
DARK_MIN = 0.5                   # 数量字压在一条深色底上（实测中间那格 0.78～0.81 是深色）
FRAME_BOX = (0.039, 0.108, 0.031)  # 一格的外框：数量字中心左右 0.039W、往上 0.108H、往下 0.031H
ICON_DY = 0.045                  # 点一格点在图标上：数量字中心往上 0.045H
ORANGE_LO, ORANGE_HI = (5, 150, 170), (22, 255, 255)   # HSV；选中框的橙色
SELECTED_MIN = 0.12              # 外圈里橙色占这么多才算选中（实测选中 0.25、没选中 0.00）
OTHERS_MAX = 0.05                # 别的格外圈的橙色不能超过这么多（两格都像选中 = 认不准）
CONFIRM_DY = 0.38                # 「确认」按钮在标题顶往下 0.44H；只认 0.38H 往下的
STAMINA_BOX = (0.75, 0.08)       # 右上角「2/240」：x 中心 > 0.75W、y < 0.08H
QTY_RE = re.compile(r'数量\s*(\d{1,4})')
COUNT_RE = re.compile(r'^\s*[xX×]?\s*(\d{1,6})\s*$')
STAMINA_RE = re.compile(r'^\s*(\d{1,3})\s*/\s*(\d{2,3})\s*$')


def refill_title(boxes):
    """框的标题「情报拼图补充」；不是这个框返回 None。"""
    for b in boxes or []:
        if REFILL_TITLE in (getattr(b, 'name', None) or ''):
            return b
    return None


def slot_center(title, k, W, H):
    """第 k 格（0 左 / 1 中 / 2 右）数量字的中心（整屏像素）。"""
    cx = title.x + title.width / 2.0
    return cx + (k - 1) * SLOT_DX * W, title.y + COUNT_DY * H


def icon_point(title, k, W, H):
    """点第 k 格时点的位置（图标中间，不点在数量字上）。"""
    x, y = slot_center(title, k, W, H)
    return x, y - ICON_DY * H


def slot_stock(frame, title, k, W, H):
    """第 k 格有没有货：数量字是白的 True、红的（0 个）False、看不清 None。

    :return: (True / False / None, 白字占比, 红字占比)
    """
    try:
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return None, 0.0, 0.0
        fh, fw = frame.shape[:2]
        sx, sy = fw / float(W), fh / float(H)
        x, y = slot_center(title, k, W, H)
        dx, dy = DIGIT_WIN[0] * W, DIGIT_WIN[1] * H
        x0, x1 = max(0, int((x - dx) * sx)), min(fw, int((x + dx) * sx))
        y0, y1 = max(0, int((y - dy) * sy)), min(fh, int((y + dy) * sy))
        if x1 <= x0 or y1 <= y0:
            return None, 0.0, 0.0
        hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1, :3]), cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        white = float(((s < 60) & (v > 200)).mean())
        red = float((((h < 10) | (h > 170)) & (s > 120) & (v > 150)).mean())
        dark = float((v < 110).mean())
    except (cv2.error, ValueError, IndexError):
        return None, 0.0, 0.0
    if dark < DARK_MIN or white > DIGIT_MAX or red > DIGIT_MAX:
        return None, round(white, 3), round(red, 3)
    if white >= WHITE_MIN and red <= OTHER_MAX:
        return True, round(white, 3), round(red, 3)
    if red >= RED_MIN and white <= OTHER_MAX:
        return False, round(white, 3), round(red, 3)
    return None, round(white, 3), round(red, 3)


def parse_count(boxes, title, k, W, H):
    """OCR 结果里第 k 格的数量（整数）；没读到返回 None。只信落在那格数量字附近、离中心最近的一个纯数字。只拿来写日志。"""
    x, y = slot_center(title, k, W, H)
    dx, dy = COUNT_WIN[0] * W, COUNT_WIN[1] * H
    best = None
    for b in boxes or []:
        m = COUNT_RE.match(getattr(b, 'name', None) or '')
        if not m:
            continue
        bx, by = b.x + b.width / 2.0, b.y + b.height / 2.0
        if abs(bx - x) > dx or abs(by - y) > dy:
            continue
        d = abs(bx - x) + abs(by - y)
        if best is None or d < best[0]:
            best = (d, int(m.group(1)))
    return None if best is None else best[1]


def _orange_ring(mask, rect, t):
    """rect 外框内外各 t 像素那一圈里，橙色像素的占比。"""
    x0, y0, x1, y1 = rect
    h, w = mask.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - t), max(0, y0 - t), min(w, x1 + t), min(h, y1 + t)
    if X1 <= X0 or Y1 <= Y0:
        return 0.0
    ring = np.ones((Y1 - Y0, X1 - X0), bool)
    ix0, iy0 = max(0, x0 + t - X0), max(0, y0 + t - Y0)
    ring[iy0:max(iy0, y1 - t - Y0), ix0:max(ix0, x1 - t - X0)] = False
    return float(mask[Y0:Y1, X0:X1][ring].mean()) if ring.any() else 0.0


def slot_orange(frame, title, W, H):
    """三格外圈里橙色的占比 [左, 中, 右]；画面读不了返回 None。按画面实际大小换算（画面可能和 W×H 不一样大）。"""
    try:
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return None
        fh, fw = frame.shape[:2]
        sx, sy = fw / float(W), fh / float(H)
        hsv = cv2.cvtColor(np.ascontiguousarray(frame[..., :3]), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI) > 0
        t = max(2, int(round(0.004 * fw)))
        out = []
        for k in range(3):
            x, y = slot_center(title, k, W, H)
            hw, up, down = FRAME_BOX
            rect = (int(round((x - hw * W) * sx)), int(round((y - up * H) * sy)),
                    int(round((x + hw * W) * sx)), int(round((y + down * H) * sy)))
            out.append(round(_orange_ring(mask, rect, t), 3))
        return out
    except (cv2.error, ValueError, IndexError):
        return None


def selected_slot(frame, title, W, H):
    """选中的是第几格（0 / 1 / 2）；认不准（没有一格像选中、或者不止一格像）返回 None。

    :return: (index 或 None, 三格的橙色占比)
    """
    ratios = slot_orange(frame, title, W, H)
    if not ratios:
        return None, ratios
    on = [k for k, r in enumerate(ratios) if r >= SELECTED_MIN]
    if len(on) != 1 or any(r > OTHERS_MAX for k, r in enumerate(ratios) if k != on[0]):
        return None, ratios
    return on[0], ratios


def quantity(boxes):
    """框里「取出数量 0」那行的数（选了别的格可能换字，只认「数量 N」）；没有返回 None。"""
    for b in boxes or []:
        m = QTY_RE.search(getattr(b, 'name', None) or '')
        if m:
            return int(m.group(1))
    return None


def stamina(boxes, W, H):
    """右上角的情报拼图「2/240」-> (2, 240)；读不到返回 None。"""
    for b in boxes or []:
        m = STAMINA_RE.match(getattr(b, 'name', None) or '')
        if m and b.x + b.width / 2.0 > STAMINA_BOX[0] * W and b.y < STAMINA_BOX[1] * H:
            return int(m.group(1)), int(m.group(2))
    return None


def confirm_button(boxes, title, H):
    """框里的「确认」（在标题下面 0.38H 往下）；没有返回 None。"""
    for b in boxes or []:
        if (getattr(b, 'name', None) or '').strip() == '确认' and b.y > title.y + CONFIRM_DY * H:
            return b
    return None
