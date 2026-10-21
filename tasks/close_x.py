# -*- coding: utf-8 -*-
"""右上角的关闭「×」检测（教学 / 引导，ISSUES.md I-13）。纯 cv2 / numpy，不靠颜色、不靠 ok 的模板库（独立版没有）。

做法（形状 / 连通块）：
  1. 只看搜索带（默认 x >= 0.80W、y <= 0.20H），转灰度
  2. 不分深浅：白顶帽（亮笔画）和黑顶帽（暗笔画），核约 27px@3200 —— 比它宽的面板、按钮、天空都去掉
  3. 几档对比度阈值切连通块，留下大致方形、边长 >= 36px@3200 的块
  4. 在块和几个居中的子窗口里量 X 的几何：四条半对角线都有笔画、中心交叉、四个三角楔区是空的
     —— 「+」、六 / 八瓣星、箭头、对勾、三角、文字都过不了
  5. 楔区和外圈的灰度要一致（ws）：干掉「负空间的 X」，比如部署面板上全元素图标四颗菱形之间的缝
  6. 同一行旁边有差不多高的字的（「3X」倍速按钮的 3）不算
所有尺寸都是 3200 宽下的像素，按画面宽度缩放。

9/30 调查（scratchpad/tutorial/cv-x-detector）：120 张真实画面（部署、章节地图、战斗、弹窗）全图扫描 0 次误报；
章节地图上「收集数」奖励弹窗那个真实的关闭叉 3/3 找到（中心误差 < 1px）；合成的叉 1114/1224 找到；
画上去的「+」「>」「>|」星形「3X」对勾等 612 个干扰 0 次误报。教学画面那个叉本身还没有真图。
"""
import cv2
import numpy as np

REF_W = 3200.0
REF_H = 1800.0


def _odd(n):
    n = int(round(n))
    return n if n % 2 else n + 1


def contrast_maps(gray, s):
    k = _odd(max(9, 27 * s))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker)
    blk = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)
    return top, blk


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n]
    u = (xx + 0.5) / n
    v = (yy + 0.5) / n
    d1 = np.abs(u - v) / np.sqrt(2)
    d2 = np.abs(u + v - 1) / np.sqrt(2)
    return u, v, np.minimum(d1, d2)


def _wedge_mask(n, t_rel):
    u, v, dmin = _grid(n)
    excl = min(0.25, max(0.12, t_rel / 2 + 0.08))
    return (dmin > excl) & (u > 0.06) & (u < 0.94) & (v > 0.06) & (v < 0.94)


def x_features(M, wx, wy, n):
    """X-geometry features of binary mask M (uint8 0/1) inside square window (wx, wy, n)."""
    Hh, Ww = M.shape
    if wx < 0 or wy < 0 or wx + n > Ww or wy + n > Hh or n < 8:
        return None
    win = M[wy:wy + n, wx:wx + n]
    r = max(1, int(round(0.045 * n)))
    dil = cv2.dilate(win, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    N = max(16, n // 2)

    def cov(t0, t1, diag):
        ts = np.linspace(t0, t1, N)
        xs = np.clip(np.round(ts * (n - 1)).astype(int), 0, n - 1)
        ys = xs if diag == 1 else (n - 1 - xs)
        return float(dil[ys, xs].mean())
    arms = [cov(.10, .40, 1), cov(.60, .90, 1), cov(.10, .40, 2), cov(.60, .90, 2)]
    c0 = int(round(0.5 * (n - 1)))
    rc = max(1, int(round(0.08 * n)))
    centre = float(dil[max(0, c0 - rc):c0 + rc + 1, max(0, c0 - rc):c0 + rc + 1].max())
    u, v, dmin = _grid(n)
    t_rel = float(win[dmin < 0.2].sum()) / (2 * np.sqrt(2) * n) / n
    wedge = _wedge_mask(n, t_rel)
    if wedge.sum() < 8:
        return None
    wf = float(win[wedge].mean())
    quads = (wedge & (v < u) & (v < 1 - u), wedge & (v > u) & (v > 1 - u),
             wedge & (u < v) & (u < 1 - v), wedge & (u > v) & (u > 1 - v))
    wmax = max(float(win[m].mean()) if m.sum() else 0.0 for m in quads)
    return dict(arms=arms, arm_min=min(arms), centre=centre, wedge=wf, wedge_max=wmax, t_rel=t_rel, fill=float(win.mean()))


def wedge_vs_surround(gray, wx, wy, n, t_rel, polarity):
    """(median wedge grey - median surround grey) / |stroke grey - wedge grey|; ~0 for a real X on a plain bg."""
    H, W = gray.shape
    win = gray[wy:wy + n, wx:wx + n].astype(np.float32)
    wm = _wedge_mask(n, t_rel)
    wedge_lum = float(np.median(win[wm]))
    u, v, dmin = _grid(n)
    on = (dmin < max(0.5 / n, t_rel / 4)) & (np.hypot(u - .5, v - .5) < 0.45)
    stroke = float(np.percentile(win[on], 90 if polarity == 'light' else 10))
    m = max(2, int(0.25 * n))
    X0, Y0, X1, Y1 = max(0, wx - m), max(0, wy - m), min(W, wx + n + m), min(H, wy + n + m)
    ring = gray[Y0:Y1, X0:X1].astype(np.float32)
    mask = np.ones(ring.shape, bool)
    mask[wy - Y0:wy - Y0 + n, wx - X0:wx - X0 + n] = False
    sur = float(np.median(ring[mask])) if mask.any() else wedge_lum
    return (wedge_lum - sur) / (abs(stroke - wedge_lum) + 1e-3)


DEFAULTS = dict(
    band=(0.80, 0.0, 1.0, 0.20),   # x0, y0, x1, y1 as fractions of W, H
    smin=36, smax=130,             # X side px @3200. real GF2 popup close-X = 60 px; unit-bar crossed swords <= 31 (size_ref)
    thresholds=(35, 60, 100),      # top-hat / black-hat contrast levels (grey levels)
    max_aspect=1.3,
    arm_min=0.80, wedge=0.06, wedge_max=0.12, fill_max=0.55,
    subwin=(1.0, 0.9, 0.8, 0.7, 0.62, 0.55, 0.5, 0.45),  # sweep: +1.4pt recall (ring_touch), FP unchanged
    ws_max=0.35,                   # None disables the wedge-vs-surround gate
    win_min_frac=0.75,             # sub-windows may be down to win_min_frac*smin as long as the component itself is >= smin
    text_ctx=True,
)


def detect_close_x(img, **kw):
    """Return list of dicts {x, y (frame px centre), size (px), size_ref (@3200), score, polarity, ws, ...}, best first."""
    P = dict(DEFAULTS); P.update(kw)
    H, W = img.shape[:2]
    # 16:9 下两个一样；窄屏（4:3 等）时 UI 多半跟着高度缩放，按大的那个算，最小尺寸不会被压小（复查 BS-3）
    s = max(W / REF_W, H / REF_H)
    bx0, by0, bx1, by1 = int(P['band'][0] * W), int(P['band'][1] * H), int(P['band'][2] * W), int(P['band'][3] * H)
    roi = img[by0:by1, bx0:bx1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    top, blk = contrast_maps(gray, s)
    smin, smax = P['smin'] * s, P['smax'] * s
    out = []
    for pol, cmap in (('light', top), ('dark', blk)):
        for thr in P['thresholds']:
            M = (cmap >= thr).astype(np.uint8)
            n, lab, st, cen = cv2.connectedComponentsWithStats(M, connectivity=8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                side = max(w, h)
                if side < smin or side > smax / 0.5:
                    continue
                if max(w, h) > P['max_aspect'] * min(w, h):
                    continue
                cx, cy = x + w / 2.0, y + h / 2.0
                best = None
                for f in P['subwin']:
                    nwin = int(round(side * f))
                    if nwin < P['win_min_frac'] * smin or nwin > smax:
                        continue
                    wx, wy = int(round(cx - nwin / 2.0)), int(round(cy - nwin / 2.0))
                    F = x_features(M, wx, wy, nwin)
                    if F is None:
                        continue
                    ok = (F['arm_min'] >= P['arm_min'] and F['centre'] > 0 and F['wedge'] <= P['wedge']
                          and F['wedge_max'] <= P['wedge_max'] and F['fill'] <= P['fill_max'])
                    if not ok:
                        continue
                    ws = wedge_vs_surround(gray, wx, wy, nwin, F['t_rel'], pol)
                    if P['ws_max'] is not None and abs(ws) > P['ws_max']:
                        continue
                    score = F['arm_min'] - 2.0 * F['wedge'] - F['wedge_max'] - 0.5 * abs(ws)
                    if best is None or score > best['score']:
                        best = dict(x=bx0 + cx, y=by0 + cy, size=nwin, size_ref=nwin / s, score=score, polarity=pol,
                                    thr=int(thr), f=f, ws=ws, feats=F, bbox=(bx0 + wx, by0 + wy, nwin, nwin))
                if best is None:
                    continue
                if P['text_ctx']:
                    hh = h
                    for j in range(1, n):
                        if j == i:
                            continue
                        xj, yj, wj, hj, aj = st[j]
                        if hj < 0.6 * hh or hj > 1.5 * hh or aj < 0.05 * a:
                            continue
                        if abs((yj + hj / 2.0) - cy) > 0.3 * hh:
                            continue
                        gap = max(xj - (x + w), x - (xj + wj))
                        if -0.1 * hh < gap < 0.6 * hh and not (xj <= x and yj <= y and xj + wj >= x + w and yj + hj >= y + h):
                            best['text_ctx'] = True
                            break
                    if best.get('text_ctx'):
                        continue
                out.append(best)
    out.sort(key=lambda d: -d['score'])
    merged = []
    for d in out:
        if all(np.hypot(d['x'] - m['x'], d['y'] - m['y']) > 0.4 * max(d['size'], m['size']) for m in merged):
            merged.append(d)
    return merged
