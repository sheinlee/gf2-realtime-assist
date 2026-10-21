# -*- coding: utf-8 -*-
"""章节地图上关卡卡片的状态：已完成 / 未解锁 / 可打（ok-gf2 扩展）。不定义 BaseTask 子类。

以前的判据只在一种卡片样式上标定过（编号下方的青色底边 = 已解锁、暖色像素 = 已完成），
「拂晓的回响」（白卡 + 金色「嶄」字）和「密米尔的返演」（深色水墨卡）一张卡都没有青条，
于是什么都不算解锁；金色「嶄」字又把没打过的 ZS-1-2 抬成了「已完成」。

跨所有样式都不变的只有两个图标：
  - **挂锁**（编号左边的白色挂锁）       -> 未解锁
  - **橙色圆形对勾**（卡片右上角）        -> 已完成
两个都没有 -> 可打。图标可能在屏幕外（卡片被屏幕边缘截断）时判「不确定」，**不点**。

两套独立的检测器找这两个图标，任意一个认出来就算有（模板匹配抗色调/亮度/噪声，形状规则抗框偏差、
从不误报，两者失效的场景不重叠）：
  15 张手工标注的卡 × 39 种画面条件（缩放 0.25~1.25、明暗、色调、噪声、模糊、JPEG），
  含「留一种样式不参与标定」的交叉验证：585/585，0 次把已完成 / 未解锁判成可打。
一切尺寸都以编号文字的高度为单位，换分辨率不用改。
下面是离线设计时的英文原注释。
"""

# Stage-card state on the GF2 chapter map -- FINAL detector (fusion of two independent engines).
# 
# Self-contained: numpy + cv2 only, no ok imports, pure functions of (BGR frame, code box).
# 
#     classify_card(frame, code_box, ref_h=None) -> dict(state, lock_score, check_score, certain, ...)
#         state: 'locked' | 'completed' | 'available' | 'unknown'
#     frame_ref_h(boxes)            -> robust per-frame code-text height (pass it as ref_h)
#     normalize_code_boxes(boxes)   -> repair merged raw OCR boxes (only needed without extract_stage_codes)
# 
# Two engines look for the SAME two icons (common to every observed card style):
#   * padlock  -- white padlock LEFT of the code (locked cards)
#   * badge    -- orange disc with a white tick at the card's top-right (completed cards)
# 
#   TM (template engine, from tm1/stage_icons.py, verbatim): masked normalised cross-correlation of
#      embedded templates (3 padlocks, 2 badges). Invariant to gain/offset/hue/saturation/noise, but
#      its windows and scale range are tight around the code box -> sensitive to wrong box sizes.
#   SH (shape engine, from designer2/stage_state.py, verbatim): binarisation ladder + morphological
#      shape tests. Very specific (no false fire anywhere in the background scans) and tolerant to box
#      errors, but uses absolute colour/brightness gates -> loses the badge under a hue shift of
#      >= 9 OpenCV units, global darkening to 0.6, desaturation to 0.3, and the padlock under noise sigma >= 20.
# 
# Fusion (judge's decision after re-running both designs' evals plus new stress tests, see eval_final.py):
#   an icon is PRESENT if EITHER engine fires at its own validated threshold (TM >= 0.70, SH >= 0.5).
#   state = locked if padlock present, else completed if badge present, else available -- except that
#   'available' needs positive evidence that nothing is hidden: if an icon area of the card may be
#   outside the frame, the card is 'unknown' (never clicked).
#   Rationale: the expensive mistakes for pick_next_stage are calling a completed / locked card
#   'available' (replay loop / clicking a locked stage); calling an available card completed only makes
#   the bot stop and log. Each engine covers the other's blind spots (disjoint failure modes in the
#   judge's stress tests); both engines had 0 false fires on every chapter-map background.
# 
# Scores (for logging): lock_score / check_score = max over engines of a normalised margin, where
# 0 = that engine's threshold, +1 = a clean icon (TM 0.85 / SH 1.0), negative = absent. So
# 'present' <=> score >= 0. certain = both engines agree on every icon that was evaluated and the
# state is not 'unknown' -- log disagreements, they are the cases to look at first.
# 
# Scale: every size is in units of the code-text height h. OCR box heights are noisy (+-30 % between
# cards of one frame), so pass ref_h = frame_ref_h(all code boxes of the frame): the box keeps its left
# edge, right edge and vertical centre, and its height becomes ref_h.

import base64
import math

import cv2
import numpy as np

# =====================================================================================================
# Fusion parameters
# =====================================================================================================
TM_LOCK_T = 0.70          # TM thresholds (tm1: validated incl. leave-one-style-out; interval 0.64-0.73)
TM_BADGE_T = 0.70
TM_Z_SPAN = 0.15          # TM 0.85 -> +1
SH_LOCK_T = 0.50          # SH pass marks (designer2: a-priori 0.5 = every shape term at its pass mark)
SH_BADGE_T = 0.50
SH_Z_SPAN = 0.50          # SH 1.0 -> +1
# icon reach, in h, used only to decide whether an icon could be hidden by the frame edge
# (measured: padlock centre 0.8h / 2.4h / 4.2-4.6h left of the code in styles A / B / C, half-width
#  <= 0.35h; badge centre <= 0.38h right of and <= 0.56h above the code, radius <= 0.4h)
LOCK_REACH = 5.0          # padlock may be up to 5.0h left of the code's left edge
# FIX-A：这两个数最初是在原始 OCR 框（h≈54）上量的；实际传进来的是裁剪重读后的紧框（h≈45），
# 换成紧框的单位后对勾最远到编号上方 1.27h、右方 1.05h。原值会让「屏幕上边缘刚好切掉对勾」的
# 已完成卡被判成可打（离线复现过）。
BADGE_REACH_X = 1.15      # badge may reach 1.15h right of the code's right edge
BADGE_REACH_UP = 1.35     # ... and 1.35h above the code's top edge
# 绿色勾（9/30「密米尔的返演·下篇」活动章节）：在编号**右下方**，圆心约右 0.9h、下 1.0h，半径约 0.55h，
# 白勾的长笔画再往右上伸一点。离屏幕右边 / 下边比这近的卡，勾可能在屏幕外：判「看不全」，别当成可打
GREEN_REACH_X = 1.7
GREEN_REACH_DOWN = 1.6


def _z_tm(v, t):
    return (float(v) - t) / TM_Z_SPAN


def _z_sh(v, t):
    return (float(v) - t) / SH_Z_SPAN


def frame_ref_h(boxes):
    """Robust code-text height of one frame. boxes: iterable of (x, y, w, h) or objects with .height.

    FIX-E：取「中位数 0.75~1.3 倍以内」那些框的中位数。以前是「不超过最小值 1.3 倍」的中位数，
    一个读短了的框就能把整屏所有卡的搜索窗口都缩小（缩到 0.7 倍时已完成卡会漏判）。
    """
    hs = [float(b.height if hasattr(b, 'height') else b[3]) for b in boxes]
    hs = [v for v in hs if v > 0]
    if not hs:
        return None
    m = float(np.median(hs))
    return float(np.median([v for v in hs if 0.75 * m <= v <= 1.3 * m]))


def classify_card(frame, code_box, ref_h=None, lazy=True, tm_banks=None, sh_params=None, code=None):
    """State of the stage card whose code text is at code_box = (x, y, w, h) in frame px.

    :param frame: BGR uint8 frame (the whole captured client area)
    :param code_box: (x, y, w, h) of the stage-code text (tight box from extract_stage_codes, or a raw OCR box)
    :param ref_h: per-frame code height (frame_ref_h over all code boxes); None = use the box's own h
    :param lazy: stop as soon as the state is decided (order SH lock, TM lock, SH badge, TM badge); the state
                 is identical to lazy=False, only the scores of skipped engines are None and a disagreement
                 with a skipped engine is not reported. Locked card ~20 ms, available card ~150 ms.
    :param tm_banks: optional (lock_bank, badge_bank) for the TM engine (tests only)
    :param sh_params: optional parameter overrides for the SH engine (tests only)
    :return: dict(state, lock_score, check_score, certain, disagree, tm_lock, sh_lock, tm_badge, sh_badge,
                  lock_clipped, badge_clipped, box, lock_at, badge_at)
             check_score / tm_lock / tm_badge / sh_badge are None when lazy skipped them.
    """
    if code and ref_h:
        # FIX-B：框明显比这个编号应有的宽度宽 —— 左边并进了图形（挂锁 / 金色字 / 六边形）。
        # 编号是右对齐的，保留右边缘收窄；否则挂锁落在框里面、跑出搜索窗口，未解锁会被判成可打。
        from stage_codes import anchor_right
        code_box = anchor_right(tuple(float(v) for v in code_box), ref_h, code)
    x, y, w, h = [float(v) for v in code_box]
    if ref_h:
        cy = y + h / 2.0
        h = float(ref_h)
        y = cy - h / 2.0
    box = (x, y, w, h)
    H, W = frame.shape[:2]
    lb, bb = tm_banks if tm_banks else (None, None)

    # Under the OR rule the first engine that fires decides, so lazy mode runs the cheap SH engine first
    # and skips what cannot change the state (SH lock ~20 ms, TM lock ~75 ms, SH badge ~2 ms, TM badge ~50 ms).
    sl, sli = sh_lock_score(frame, box, sh_params)
    s_lock = sl >= SH_LOCK_T
    tl, tli, t_lock = None, {}, False
    if not (lazy and s_lock):
        tl, tli = tm_lock_score(frame, box, lb)
        t_lock = tl >= TM_LOCK_T
    lock_score = max([_z_sh(sl, SH_LOCK_T)] + ([_z_tm(tl, TM_LOCK_T)] if tl is not None else []))
    is_lock = t_lock or s_lock
    disagree = [] if (tl is None or t_lock == s_lock) else ['lock(tm %.2f / sh %.2f)' % (tl, sl)]

    tb = sb = gb = check_score = None
    tbi, sbi, gbi = {}, {}, {}
    t_badge = s_badge = g_badge = False
    if not (lazy and is_lock):
        sb, sbi = sh_badge_score(frame, box, sh_params)
        s_badge = sb >= SH_BADGE_T
        if not (lazy and s_badge):
            # 绿色勾（另一种卡片样式的「已完成」）：橙色勾那两套检测都认不出它（见 sh_green_badge_score）
            gb, gbi = sh_green_badge_score(frame, box)
            g_badge = gb >= SH_GREEN_T
        if not (lazy and (s_badge or g_badge)):
            tb, tbi = tm_badge_score(frame, box, bb)
            t_badge = tb >= TM_BADGE_T
        check_score = max([_z_sh(sb, SH_BADGE_T)] + ([_z_tm(tb, TM_BADGE_T)] if tb is not None else [])
                          + ([(gb - SH_GREEN_T) / SH_Z_SPAN] if gb is not None else []))
        if tb is not None and t_badge != s_badge and not g_badge:
            disagree.append('badge(tm %.2f / sh %.2f)' % (tb, sb))
    is_badge = t_badge or s_badge or g_badge

    lock_clipped = x - LOCK_REACH * h < 0
    badge_clipped = ((x + w + BADGE_REACH_X * h > W) or (y - BADGE_REACH_UP * h < 0)
                     or (x + w + GREEN_REACH_X * h > W) or (y + h / 2.0 + GREEN_REACH_DOWN * h > H))
    if is_lock:
        state = 'locked'
    elif is_badge:
        state = 'completed'
    elif lock_clipped or badge_clipped:
        state = 'unknown'          # an icon could be off-screen: never treat as available
    else:
        state = 'available'
    lock_at = (tli.get('cx'), tli.get('cy')) if t_lock else (
        (sli['body'][0] + sli['body'][2] / 2.0, sli['body'][1] + sli['body'][3] / 2.0) if s_lock else None)
    badge_at = None
    if is_badge:
        badge_at = (tbi.get('cx'), tbi.get('cy')) if t_badge else (
            sbi.get('centre') if s_badge else gbi.get('centre'))
    return dict(state=state, lock_score=round(lock_score, 3),
                check_score=None if check_score is None else round(check_score, 3),
                certain=(state != 'unknown' and not disagree), disagree=disagree,
                tm_lock=None if tl is None else round(float(tl), 3), sh_lock=round(float(sl), 3),
                tm_badge=None if tb is None else round(float(tb), 3),
                sh_badge=None if sb is None else round(float(sb), 3),
                green_badge=None if gb is None else round(float(gb), 3),
                lock_clipped=lock_clipped, badge_clipped=badge_clipped,
                box=tuple(round(v, 1) for v in box), lock_at=lock_at, badge_at=badge_at)


def normalize_code_boxes(boxes, ratio=None):
    """(from tm1) Repair raw OCR code boxes merged with the graphics on their left (e.g. ' ZA-1-2'
    427x112 while clean reads are ~165x54). Not needed for boxes from extract_stage_codes.
    boxes: list of (x, y, w, h) -> new list."""
    if not boxes:
        return []
    hmin = min(b[3] for b in boxes)
    clean = [b for b in boxes if b[3] <= 1.3 * hmin]
    ref_h = float(np.median([b[3] for b in clean]))
    if ratio is None:
        ratio = float(np.median([b[2] / float(b[3]) for b in clean]))
    out = []
    for x, y, w, h in boxes:
        if h > 1.3 * ref_h:
            pad = (h - ref_h) / 2.0
            right = x + w - pad
            cy = y + h / 2.0
            nw = ratio * ref_h
            out.append((right - nw, cy - ref_h / 2.0, nw, ref_h))
        else:
            out.append((x, y, w, h))
    return out


# =====================================================================================================
# TM engine -- template matching (tm1/stage_icons.py, unchanged logic; names prefixed tm_)
# =====================================================================================================
CANON_H = 40.0                                   # code-text height after normalisation (px)
TM_SCALES = {'lock': tuple(1.08 ** k for k in range(-6, 7)),
             'badge': tuple(1.08 ** k for k in range(-8, 8))}
TM_MIN_STD = 6.0
#   lock : x in [code_left + x0*h, code_left + x1*h],   y in [code_cy + y0*h, code_cy + y1*h]
#   badge: x in [code_right + x0*h, code_right + x1*h], y in [code_top + y0*h, code_cy + y1*h]
TM_LOCK_WIN = (-6.2, 0.3, -1.4, 1.5)
TM_BADGE_WIN = (-1.2, 2.0, -1.9, 0.3)


def _rep_lock(bgr):
    g = cv2.GaussianBlur(bgr.min(axis=2).astype(np.float32), (0, 0), 1.0)
    return np.dstack([cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)])


def _rep_badge(bgr):
    f = bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    return np.dstack([np.clip(r - g, 0, 255), np.clip(r - b, 0, 255), f.min(axis=2)])


_REPS = {'lock': _rep_lock, 'badge': _rep_badge}


def _dec(s):
    return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_UNCHANGED)


def tm_load_bank(kind, exclude_styles=()):
    """Templates of kind ('lock' / 'badge'); exclude_styles for leave-one-style-out tests."""
    out = []
    for item in TM_EMBEDDED.get(kind, []):
        name, style, png = item[:3]
        if style in exclude_styles:
            continue
        bgra = _dec(png)
        out.append({'name': name, 'style': style, 'bgr': bgra[:, :, :3].copy(),
                    'mask': (bgra[:, :, 3] > 127).astype(np.uint8),
                    'parts': item[3] if len(item) > 3 else None})
    return out


_TM_BANK_CACHE = {}


def _tm_default_bank(kind):
    if kind not in _TM_BANK_CACHE:
        _TM_BANK_CACHE[kind] = tm_load_bank(kind)
    return _TM_BANK_CACHE[kind]


def _tm_prep_bank(bank, kind):
    rep = _REPS[kind]
    for t in bank:
        if '_scaled' in t:
            continue
        th0, tw0 = t['bgr'].shape[:2]
        lst = []
        for s in TM_SCALES[kind]:
            w, h = max(3, int(round(tw0 * s))), max(3, int(round(th0 * s)))
            rs = rep(cv2.resize(t['bgr'], (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR))
            if rs.ndim == 2:
                rs = rs[:, :, None]
            m = cv2.resize(t['mask'], (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            rect = bool(m.all())
            if t.get('parts') and rect:
                spans = [(int(round(a * h)), int(round(b * h))) for a, b in t['parts']]
            else:
                spans = [(0, h)]
            bands = []
            for r0, r1 in spans:
                sub, mb = rs[r0:r1], m[r0:r1]
                n = float(mb.sum())
                if n < 9:
                    break
                mean = (sub * mb[:, :, None]).reshape(-1, sub.shape[2]).sum(0) / n
                tz = (sub - mean) * mb[:, :, None]
                tnorm = float(np.sqrt((tz * tz).sum()))
                if tnorm <= 1e-6:
                    break
                bands.append((r0, r1, [np.ascontiguousarray(tz[:, :, i], dtype=np.float32) for i in range(tz.shape[2])],
                              None if rect else np.ascontiguousarray(mb), n, tnorm))
            if len(bands) == len(spans):
                lst.append((s, h, w, bands))
        t['_scaled'] = lst
    return bank


def tm_window_rect(code_box, kind, frame_shape):
    x, y, w, h = code_box
    if kind == 'lock':
        x0, x1, y0, y1 = TM_LOCK_WIN
        rx0, rx1 = x + x0 * h, x + x1 * h
        ry0, ry1 = y + h / 2 + y0 * h, y + h / 2 + y1 * h
    else:
        x0, x1, y0, y1 = TM_BADGE_WIN
        rx0, rx1 = x + w + x0 * h, x + w + x1 * h
        ry0, ry1 = y + y0 * h, y + h / 2 + y1 * h
    fh, fw = frame_shape[:2]
    rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
    rx1, ry1 = min(fw, int(np.ceil(rx1))), min(fh, int(np.ceil(ry1)))
    return rx0, ry0, rx1, ry1


def _tm_canon_roi(frame, rect, code_h, kind):
    x0, y0, x1, y1 = rect
    f = CANON_H / float(code_h)
    roi = frame[y0:y1, x0:x1]
    roi = cv2.resize(roi, (max(1, int(round(roi.shape[1] * f))), max(1, int(round(roi.shape[0] * f)))),
                     interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    return _REPS[kind](roi), f


def _tm_response_maps(R, kind, bank):
    """Masked TM_CCOEFF_NORMED summed over channels, min over bands (see tm1/stage_icons.py)."""
    _tm_prep_bank(bank, kind)
    if R.ndim == 2:
        R = R[:, :, None]
    R = np.ascontiguousarray(R, dtype=np.float32)
    H, W, C = R.shape
    chans = [np.ascontiguousarray(R[:, :, i]) for i in range(C)]
    R2 = (R * R).sum(axis=2)
    integ = None
    for t in bank:
        for s, th, tw, bands in t['_scaled']:
            if th > H or tw > W:
                continue
            oh, ow = H - th + 1, W - tw + 1
            res = None
            for r0, r1, tz, mb, n, tnorm in bands:
                bh = r1 - r0
                num = cv2.matchTemplate(chans[0][r0:H - th + r1], tz[0], cv2.TM_CCORR)
                for i in range(1, C):
                    num += cv2.matchTemplate(chans[i][r0:H - th + r1], tz[i], cv2.TM_CCORR)
                if mb is None:
                    if integ is None:
                        integ = [cv2.integral(c, sdepth=cv2.CV_64F) for c in [R2] + chans]

                    def box(ii):
                        return (ii[r0 + bh:r0 + bh + oh, tw:tw + ow] - ii[r0:r0 + oh, tw:tw + ow]
                                - ii[r0 + bh:r0 + bh + oh, :ow] + ii[r0:r0 + oh, :ow]).astype(np.float32)
                    var = box(integ[0])
                    for ii in integ[1:]:
                        s1 = box(ii)
                        var -= s1 * s1 / n
                else:
                    var = cv2.matchTemplate(R2[r0:H - th + r1], mb, cv2.TM_CCORR)
                    for c in chans:
                        s1 = cv2.matchTemplate(c[r0:H - th + r1], mb, cv2.TM_CCORR)
                        var -= s1 * s1 / n
                var = np.maximum(var, 0)
                r = num / (np.sqrt(var) * tnorm + 1e-6)
                r[var < (TM_MIN_STD * TM_MIN_STD) * n * C] = -1
                res = r if res is None else np.minimum(res, r)
            yield t, s, res, tw, th


def _tm_match_region(frame, rect, code_h, kind, bank=None):
    if bank is None:
        bank = _tm_default_bank(kind)
    x0, y0, x1, y1 = rect
    if x1 - x0 < 4 or y1 - y0 < 4 or code_h <= 0:
        return -1.0, {}
    R, f = _tm_canon_roi(frame, rect, code_h, kind)
    best, info = -1.0, {}
    for t, s, res, tw, th in _tm_response_maps(R, kind, bank):
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best:
            best = float(mx)
            info = {'cx': x0 + (loc[0] + tw / 2.0) / f, 'cy': y0 + (loc[1] + th / 2.0) / f,
                    'scale': round(s, 3), 'tpl': t['name']}
    return best, info


def tm_lock_score(frame, code_box, bank=None):
    return _tm_match_region(frame, tm_window_rect(code_box, 'lock', frame.shape), code_box[3], 'lock', bank)


def tm_badge_score(frame, code_box, bank=None):
    return _tm_match_region(frame, tm_window_rect(code_box, 'badge', frame.shape), code_box[3], 'badge', bank)


# =====================================================================================================
# SH engine -- colour/shape rules (designer2/stage_state.py, unchanged logic; names prefixed sh_)
# =====================================================================================================
SH_NO_CANDIDATE = -1.0
SH_LOCK_ROI_LEFT = 7.0          # body centre x in [code_left - 7h, code_left + 0.2h]
SH_LOCK_ROI_RIGHT = 0.2
SH_LOCK_ROI_HALF_H = 0.9        # body centre y within cy +- 0.9h
SH_LOCK_PAD = 1.0
SH_LOCK_PAD_UP = 2.0
SH_LOCK_PAD_DOWN = 1.6
SH_LOCK_THRESHOLDS = tuple(range(250, 69, -10))   # grey-level ladder
SH_LOCK_CHROMA_BASE = 30        # neutral: max-min channel <= 30 + 0.15*max
SH_LOCK_CHROMA_REL = 0.15
SH_LOCK_OPEN_K = 0.18           # opening kernel, x h
SH_LOCK_BRIGHT_TERMS_LO = 0.40
SH_LOCK_DARK_TERMS_LO = 0.50
SH_BADGE_ROI = (-0.8, 1.8, -2.0, 0.4)   # x: code_right + [-0.8h, +1.8h], y: cy + [-2.0h, +0.4h]
SH_BADGE_CHROMA_MIN = 90
SH_BADGE_V_MIN = 120
SH_BADGE_ASPECT = (0.70, 1.43)
SH_BADGE_WHITE_V = 170
SH_BADGE_WHITE_CHROMA = 60
SH_BADGE_TERMS = dict(circ=(0.60, 0.25), white=(0.02, 0.08), orange=(0.30, 0.30))   # (lo, span)
SH_DEFAULTS = dict(LOCK_BODY_W=(0.29, 1.39), LOCK_BODY_ASPECT=(1.14, 2.43), BADGE_HUE=(5, 19),
                   BADGE_DIAM=(0.24, 1.58))
_OFF_FRAME, _OFF_ROI = 'off_frame', 'off_roi'


def _sh_p(p):
    d = dict(SH_DEFAULTS)
    if p:
        d.update(p)
    return d


def _clip_rect(x0, y0, x1, y1, W, H):
    return max(0, int(round(x0))), max(0, int(round(y0))), min(W, int(round(x1))), min(H, int(round(y1)))


def _sh_band(mask, by, bx, bh, bw, fy0, fy1, fx0, fx1, frame_edge=(False, False, False, False)):
    Hm, Wm = mask.shape
    ya, yb = by + int(math.floor(fy0 * bh)), by + int(math.ceil(fy1 * bh))
    xa, xb = bx + int(math.floor(fx0 * bw)), bx + int(math.ceil(fx1 * bw))
    crossed = (xa < 0, ya < 0, xb > Wm, yb > Hm)
    if any(crossed):
        return _OFF_FRAME if all(e for c, e in zip(crossed, frame_edge) if c) else _OFF_ROI
    if yb <= ya or xb <= xa:
        return _OFF_ROI
    return mask[ya:yb, xa:xb]


def _sh_longest_runs(mask):
    best = np.zeros(mask.shape[1], np.int32)
    cur = np.zeros(mask.shape[1], np.int32)
    for row in mask:
        cur = np.where(row, cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _sh_shackle_terms(bright, bx, by, bw, bh, fe):
    legs = _sh_band(bright, by, bx, bh, bw, -0.55, -0.15, 0.0, 1.0, fe)
    colband = _sh_band(bright, by, bx, bh, bw, -1.00, 0.0, 0.0, 1.0, fe)
    arch = _sh_band(bright, by, bx, bh, bw, -1.00, -0.40, 0.36, 0.64, fe)
    above = _sh_band(bright, by, bx, bh, bw, -1.45, -1.05, 0.10, 0.90, fe)
    left = _sh_band(bright, by, bx, bh, bw, 0.15, 0.85, -0.35, -0.08, fe)
    right = _sh_band(bright, by, bx, bh, bw, 0.15, 0.85, 1.08, 1.35, fe)
    below = _sh_band(bright, by, bx, bh, bw, 1.12, 1.45, 0.10, 0.90, fe)
    if any(isinstance(b, str) for b in (legs, arch, above, colband)):
        return None
    prof = legs.mean(axis=0)
    n = prof.shape[0]

    def cols(a, b):
        return slice(int(a * n), max(int(a * n) + 1, int(math.ceil(b * n))))

    runs = _sh_longest_runs(colband)
    unit = max(1.0, 0.45 * bh)
    t = {
        'legL': min(1.0, float(runs[cols(0.06, 0.46)].max()) / unit),
        'legR': min(1.0, float(runs[cols(0.54, 0.94)].max()) / unit),
        'hole': 1.0 - float(prof[cols(0.42, 0.58)].mean()),
        'corner': 1.0 - max(float(prof[cols(0.0, 0.07)].mean()), float(prof[cols(0.93, 1.0)].mean())),
        'arch': float(arch.mean(axis=1).max()),
        'above': 1.0 - float(above.mean()),
    }
    for name, b in (('left', left), ('right', right), ('below', below)):
        if b is _OFF_ROI:
            t[name] = 0.0
        elif b is not _OFF_FRAME:
            t[name] = 1.0 - float(b.mean())
    return t


def _sh_norm_term(name, v):
    lo = SH_LOCK_BRIGHT_TERMS_LO if name in ('legL', 'legR', 'arch', 'fill') else SH_LOCK_DARK_TERMS_LO
    return (v - lo) / 0.4


def sh_lock_score(frame, box, p=None):
    p = _sh_p(p)
    x, y, w, h = [float(v) for v in box]
    H, W = frame.shape[:2]
    cy = y + h / 2.0
    x0, y0, x1, y1 = _clip_rect(x - (SH_LOCK_ROI_LEFT + SH_LOCK_PAD) * h, cy - SH_LOCK_PAD_UP * h,
                                x + (SH_LOCK_ROI_RIGHT + SH_LOCK_PAD) * h, cy + SH_LOCK_PAD_DOWN * h, W, H)
    info = {'score': SH_NO_CANDIDATE}
    if x1 - x0 < 4 or y1 - y0 < 4 or h < 4:
        return SH_NO_CANDIDATE, info
    patch = frame[y0:y1, x0:x1]
    roi = patch.astype(np.int16)
    mx, mn = roi.max(axis=2), roi.min(axis=2)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    neutral = (mx - mn) <= (SH_LOCK_CHROMA_BASE + SH_LOCK_CHROMA_REL * mx)
    k = max(3, int(round(SH_LOCK_OPEN_K * h)))
    kern = np.ones((k, k), np.uint8)
    cx_min, cx_max = (x - SH_LOCK_ROI_LEFT * h) - x0, (x + SH_LOCK_ROI_RIGHT * h) - x0
    cy_min, cy_max = (cy - SH_LOCK_ROI_HALF_H * h) - y0, (cy + SH_LOCK_ROI_HALF_H * h) - y0
    fe = (x0 == 0, y0 == 0, x1 == W, y1 == H)
    mh, mw = gray.shape
    best = SH_NO_CANDIDATE
    for thr in SH_LOCK_THRESHOLDS:
        bright = (gray >= thr) & neutral
        if bright.sum() < k * k:
            continue
        opened = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, kern)
        n, _, stats, cents = cv2.connectedComponentsWithStats(opened, connectivity=8)
        for i in range(1, n):
            bx, by, bw, bh, _area = [int(v) for v in stats[i]]
            if not (p['LOCK_BODY_W'][0] * h <= bw <= p['LOCK_BODY_W'][1] * h):
                continue
            asp = bw / float(max(bh, 1))
            if not (p['LOCK_BODY_ASPECT'][0] <= asp <= p['LOCK_BODY_ASPECT'][1]):
                continue
            ccx, ccy = cents[i]
            if not (cx_min <= ccx <= cx_max and cy_min <= ccy <= cy_max):
                continue
            if (bx == 0 and not fe[0]) or (by == 0 and not fe[1]) or \
                    (bx + bw == mw and not fe[2]) or (by + bh == mh and not fe[3]):
                continue
            terms = _sh_shackle_terms(bright, bx, by, bw, bh, fe)
            if terms is None:
                continue
            terms['fill'] = float(bright[by:by + bh, bx:bx + bw].mean())
            s = min(_sh_norm_term(k2, v) for k2, v in terms.items())
            if s > best:
                best = s
                weakest = min(terms, key=lambda k2: _sh_norm_term(k2, terms[k2]))
                info = dict(score=s, thr=thr, body=(bx + x0, by + y0, bw, bh), weakest=weakest,
                            body_w_h=round(bw / h, 3), aspect=round(asp, 3),
                            terms={k2: round(v, 2) for k2, v in terms.items()})
    info['score'] = best
    return best, info


def sh_badge_score(frame, box, p=None):
    p = _sh_p(p)
    x, y, w, h = [float(v) for v in box]
    H, W = frame.shape[:2]
    cy, right = y + h / 2.0, x + w
    x0, y0, x1, y1 = _clip_rect(right + SH_BADGE_ROI[0] * h, cy + SH_BADGE_ROI[2] * h,
                                right + SH_BADGE_ROI[1] * h, cy + SH_BADGE_ROI[3] * h, W, H)
    info = {'score': SH_NO_CANDIDATE}
    if x1 - x0 < 4 or y1 - y0 < 4 or h < 4:
        return SH_NO_CANDIDATE, info
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, val = hsv[:, :, 0], hsv[:, :, 2]
    r16 = roi.astype(np.int16)
    chroma = r16.max(axis=2) - r16.min(axis=2)
    orange = (hue >= p['BADGE_HUE'][0]) & (hue <= p['BADGE_HUE'][1]) & \
             (chroma >= SH_BADGE_CHROMA_MIN) & (val >= SH_BADGE_V_MIN)
    white = (val >= SH_BADGE_WHITE_V) & (chroma <= SH_BADGE_WHITE_CHROMA)
    kc = max(3, int(round(0.12 * h)))
    joined = cv2.morphologyEx(orange.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    best = SH_NO_CANDIDATE
    for i in range(1, n):
        bx, by, bw, bh, _area = [int(v) for v in stats[i]]
        d = 0.5 * (bw + bh)
        if not (p['BADGE_DIAM'][0] * h <= d <= p['BADGE_DIAM'][1] * h):
            continue
        if not (SH_BADGE_ASPECT[0] <= bw / float(max(bh, 1)) <= SH_BADGE_ASPECT[1]):
            continue
        pts = np.column_stack(np.nonzero((lab == i) & orange))[:, ::-1].astype(np.int32)
        if len(pts) < 8:
            continue
        hull = cv2.convexHull(pts)
        (ex, ey), er = cv2.minEnclosingCircle(pts)
        circ = float(cv2.contourArea(hull)) / (math.pi * er * er + 1e-6)
        hm = np.zeros(orange.shape, np.uint8)
        cv2.fillConvexPoly(hm, hull, 1)
        inner = hm.astype(bool)
        n_in = float(inner.sum())
        if n_in < 8:
            continue
        raw = dict(circ=circ, white=float((white & inner).sum()) / n_in,
                   orange=float((orange & inner).sum()) / n_in)
        terms = {k2: (v - SH_BADGE_TERMS[k2][0]) / SH_BADGE_TERMS[k2][1] for k2, v in raw.items()}
        s = min(terms.values())
        if s > best:
            best = s
            info = dict(score=s, centre=(round(ex + x0), round(ey + y0)), diam_h=round(2 * er / h, 3),
                        hue_med=float(np.median(hue[orange & inner])), weakest=min(terms, key=terms.get),
                        raw={k2: round(v, 3) for k2, v in raw.items()})
    info['score'] = best
    return best, info


# 绿色勾：青绿色实心圆（半径约 0.55h，饱和度 ≥200、亮度 ≥150）+ 中间白勾，外面一圈同色但更暗、更淡的光晕。
# 在 9/30 三张现场图的 9 张已完成卡上量的（原图和 0.6 倍各一遍，得分 0.23~0.45）。光晕不算进圆里
# （饱和度 90~160、亮度 100~147），否则「圆」会量成 2h 宽、圆度只有 0.5。
SH_GREEN_ROI = (-0.3, 2.3, -0.4, 2.1)   # x: code_right + [..]h, y: cy + [..]h（编号右下方）
SH_GREEN_HUE = (72, 92)
SH_GREEN_S_MIN = 185
SH_GREEN_V_MIN = 145
SH_GREEN_WHITE_V = 190
SH_GREEN_WHITE_S = 110
SH_GREEN_DIAM = (0.55, 1.7)             # x h（含伸出圆外的白勾笔画）
SH_GREEN_ASPECT = (0.6, 1.65)
# 圆度及格线 0.52：毛笔刷样的勾（PA-1-2）在 1920 下、地图停在某些亚像素位置时圆度只有 0.596（复查 CH-3），
# 0.60 就漏了；别的样式在这一步连候选都没有（得分 -1），放低不会多认
SH_GREEN_TERMS = dict(circ=(0.52, 0.25), green=(0.40, 0.30), white=(0.03, 0.08))   # (pass mark, span)
SH_GREEN_T = 0.0


def sh_green_badge_score(frame, box):
    """编号右下方有没有「青绿色实心圆 + 白勾」。得分 = 圆度 / 绿色占比 / 白色占比三项里最差的那项
    （0 = 刚好及格）。条状（旧样式解锁卡底部的青色条）、片状的绿色过不了圆度和尺寸。"""
    x, y, w, h = [float(v) for v in box]
    H, W = frame.shape[:2]
    cy, right = y + h / 2.0, x + w
    x0, y0, x1, y1 = _clip_rect(right + SH_GREEN_ROI[0] * h, cy + SH_GREEN_ROI[2] * h,
                                right + SH_GREEN_ROI[1] * h, cy + SH_GREEN_ROI[3] * h, W, H)
    info = {'score': SH_NO_CANDIDATE}
    if x1 - x0 < 6 or y1 - y0 < 6 or h < 4:
        return SH_NO_CANDIDATE, info
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    core = (hue >= SH_GREEN_HUE[0]) & (hue <= SH_GREEN_HUE[1]) & (sat >= SH_GREEN_S_MIN) & (val >= SH_GREEN_V_MIN)
    white = (val >= SH_GREEN_WHITE_V) & (sat <= SH_GREEN_WHITE_S)
    k = max(3, int(round(0.15 * h)))
    joined = cv2.morphologyEx((core | white).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    best = SH_NO_CANDIDATE
    for i in range(1, n):
        bx, by, bw, bh, _area = [int(v) for v in stats[i]]
        d = 0.5 * (bw + bh)
        if not (SH_GREEN_DIAM[0] * h <= d <= SH_GREEN_DIAM[1] * h):
            continue
        if not (SH_GREEN_ASPECT[0] <= bw / float(max(bh, 1)) <= SH_GREEN_ASPECT[1]):
            continue
        comp = lab == i
        pts = np.column_stack(np.nonzero(comp))[:, ::-1].astype(np.int32)
        if len(pts) < 8:
            continue
        hull = cv2.convexHull(pts)
        (ex, ey), er = cv2.minEnclosingCircle(pts)
        circ = float(cv2.contourArea(hull)) / (math.pi * er * er + 1e-6)
        hm = np.zeros(comp.shape, np.uint8)
        cv2.fillConvexPoly(hm, hull, 1)
        inner = hm.astype(bool)
        n_in = float(inner.sum())
        if n_in < 8:
            continue
        raw = dict(circ=circ, green=float((core & inner).sum()) / n_in, white=float((white & inner).sum()) / n_in)
        terms = {k2: (v - SH_GREEN_TERMS[k2][0]) / SH_GREEN_TERMS[k2][1] for k2, v in raw.items()}
        sc = min(terms.values())
        if sc > best:
            best = sc
            info = dict(score=sc, centre=(round(ex + x0), round(ey + y0)), diam_h=round(2 * er / h, 3),
                        weakest=min(terms, key=terms.get), raw={k2: round(v, 3) for k2, v in raw.items()})
    info['score'] = best
    return best, info


# =====================================================================================================
# TM embedded templates (generated by tm1/build_templates.py; alpha = match mask; optional row bands)
# =====================================================================================================
TM_EMBEDDED = {
    'lock': [
        ('lock_A', 'A',
         'iVBORw0KGgoAAAANSUhEUgAAACcAAAApCAYAAAC2qTBFAAAEw0lEQVRYCc3B/4/TdwHH8eer7d3nKripyEj2xZiob2650jfiO0rml6ibDFw0Bvlhq6uxTvxp/jX+ZPyhxlB6aswQ/DKiYohhWtolRn8wZqfG4Y7pDm5O1MCVe9kiculHsN2gyT0eCjGZTUohJrNJKcRkNimFmMwmpRCT2aQUYjK3xWBGievE7VCIybwBtpmUJN4IhZjMxIzNNXPZDPfu2MHWrVsoFAoI8V/2Ov3+VVZffZW/rFykf/UqYkDi9VCIyUzKpjRT4sH3vIsD+z7BjnvuYWZmBokBscGsr5vLly+z9Ps/cOwHP+KVlQuY10chJjMJm1KpRNpd5bEDj/L2bduY1AtLSxw78UP++OI5jAExCYWYzARs884H7uNLX6iz7W3bkMSQ19cxJk8SUoGh9fV1zna7fOd73+cf//wXkpiEQkxmLGPDp/c/woF9+1ChQL/f54WlJc72nue11/6O2LBuc/999/LhDz7E9u3bGbp06RJf/drX+dO5l5DEJBRiMuPYqCAOf/5zvHf3boZWVy/S/vZ3+c1vf4dtkLjBZmamxP6HP87+fY9QLBbB5ptHF3mu+zySmIRCTGYcm2KxwNOHv8j8/DxD58+fp3mkzYt/fgkkxIAAgxmw2fv+9/HEoc+SZRlDzxw/zsmfnkYFAWIchZjMODbFYoGnDz/F/PxOhpaXl2keOcq55ZcRAxI32BjYm/bw+KGDzM3NMfTM8ROcPHUaiQExjkJMZhybYrHAV778FDt37mRoeXmZ5pGjnFt+GTEgcYONgb1pD48fOsjc3BxDx46f4NlTp5EYEOMoxGT+HxsDxUKBz3xyPw+8434ErFy4wLM/PsUrF1cZEhvMfzwY3s3HPvIhstlZDJx57hd0f/VrQEhiHIWYzK3YmCGDwZgNQlwn8T9szJDZIMSAxJAYkLgVhZjMzdhseVOZt959F4VCAUlIYoOxuc6MEgIsEEPiGhvb2OZKv8/F1b9xZW0NJG5GISaTZzMzU+Lgpx7job0foFAocKdduXKZkz85xclTp5EYEHkKMZkc25SzWRpP1qjuqjAtnW6PZmsRMSCRpxCTybFNOZulUa9RrVSYlk63R7O1yJAk8hRiMiOMDeVslka9RrVSYVo63R7NVhsQkshTiMmMMDaUs4xGvUa1ssC0dLo9mq02ICSRpxCTGWFsKGcZjXqNamWBael0ezRbbUBIIk8hJjPC2FDOMhr1GtXKAtPS6fZottqAkESeQkxmhLGhnGU06jWqlQWmpdPt0Wy1ASGJPIWYzAhjQznLaNRrVCsLTEun26PZagNCEnkKMZkRxoZyltGo16hWFpiWTrdHs9UGhCTyFGIyI4wN5WyWRr1GtVJhWjrdHs1WGxCSyFOIyeTYppzN0qjXqFYqTEun26PZWmRIEnkKMZkc25SzWRr1GtVKhWnpdHs0W4sMSSJPISaTY5u5bJbGk08Qd+1iWn55tss3jn4LMSCRpxCTybMplUo8+vBH2ROrFItFrhG3x1xjoL+2xs9+foYznS4SAyJPISZzMzZv3rqFt9x9F8VCEcSdYTDQ7/f568oKa2t9kLgZhZjMLdhm2iRxKwoxmbEM5s4RA2IchZjMJqUQk9mkFGIym5RCTGaT+jfrDtABX1x5bwAAAABJRU5ErkJggg==',
         [(0.0, 0.5), (0.411, 1.0)]),
        ('lock_B', 'B',
         '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',
         [(0.0, 0.491), (0.404, 1.0)]),
        ('lock_C', 'C',
         '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',
         [(0.0, 0.491), (0.4, 1.0)]),
    ],
    'badge': [
        ('badge_A', 'A',
         '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'),
        ('badge_B', 'B',
         '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'),
        # badge_L：10/10「镜茧与囚蝶」活动的卡片（I-28）。橙色圆外面多一圈白环、下半个圆压在卡片的深色上沿，
        # badge_A / B 只配到 0.667（门槛 0.70），形状那套圆度也差一点 —— 打完的 LA-1-3～1-5 被当成可打又打了一遍。
        # 从你 20:36 的截图里 LA-1-4 那个勾裁的（只要橙色圆，圆形掩码，按编号高 44px 缩到 CANON_H）
        ('badge_L', 'L',
         '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'),
    ],
}
