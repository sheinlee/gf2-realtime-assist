# -*- coding: utf-8 -*-
"""章节地图上的关卡编号提取（ok-gf2 扩展）。和 support_pick.py 一样不定义 BaseTask 子类。

为什么不能直接拿整屏 OCR 的结果：onnxocr 先把图缩到长边 960 再检测文字，3072 宽的画面等于
缩到 0.31 倍，50px 高的编号只剩 16px —— 于是编号和左边的金色「嶄」字 / 六边形 / 声波图并成一个框
（「学ZS-1-2」、427x112 的「 ZA-1-2」），甚至读错（「草25-12」），暗色的未解锁编号干脆读不出来
（密米尔那张图三个 SA-2-x 只读出一个「SA-2」）。所以对每个「像编号」的片段裁一小块**原分辨率**
重新 OCR，拿到干净的编号和紧贴文字的框。被屏幕边缘截断的编号用「墨迹是否贴边」剔除。

四张实拍章节图 × 4 种缩放：旧规则认出 39/60 个编号，这里 60/60，0 个误报（部署 / 战斗 / 菜单等
22 张非章节截图、「V/D178-828-345」这类序列号、FPS 悬浮条都不会被当成编号）。
下面是离线设计时的英文原注释。
"""

# Robust stage-code extraction for the GF2 chapter map (self-contained, portable into stage_select.py).
# 
# Why the single full-frame OCR is not enough
# -------------------------------------------
# onnxocr's detector resizes the image so the LONG side is 960 px (det_limit_side_len=960,
# limit_type='max').  A 3072x1728 frame is detected at ~0.31x, i.e. a 50 px code becomes ~16 px.
# At that scale the detector
#   * merges the code with whatever graphic sits left of it (gold 「嶄」 glyph, hexagons, waveform)
#     -> '学ZS-1-2', ' ZA-1-2' with boxes 2-3x too big, or a misread '草25-12';
#   * truncates / misses low-contrast (locked) codes -> 'SA-2' for SA-2-1, SA-2-2 / SA-2-3 not seen.
# Because the long side is always squeezed to 960, a SMALLER window does not help the full pass
# (a 1536 px frame is detected at the same effective resolution), but a CROP of <= ~960 px is
# detected at native resolution.  So the fix is: take the full-frame tokens as *seeds* and re-OCR
# small crops around every code-like seed with the same engine.
# 
# Public API
# ----------
#     extract_stage_codes(frame, tokens, ocr_fn, deep='auto', verify='all', details=False)
#         -> [(code, (x, y, w, h))]          sorted left->right, top->bottom
#     frame  : BGR ndarray (the same frame the tokens came from)
#     tokens : full-frame OCR result; items may be ok.Box-like (name/confidence/x/y/width/height),
#              tuples (text, conf, x, y, w, h) or dicts {text, conf, x, y, w, h}
#     ocr_fn : ocr_fn(bgr_crop) -> tokens in CROP coordinates (same accepted formats).
#              In ok:  lambda img: self.ocr(frame=img)       (returns Box list, crop coords)
# 
# Everything size-dependent is expressed in units of h_ref (code text height measured on this
# frame), so it works for 3072x1728 as well as 1600x900 windows.

import re
import time
import unicodedata

import cv2
import numpy as np

# ---------------------------------------------------------------- patterns
# A code anywhere inside a token.  (?!\d) stops 'ZS-1-222' from yielding 'ZS-1-22'.
# 两种编号：活动 / 支线的「两个字母-章-节」（ZA-1-2、BH-2-5、SL-3-1），和主线的**纯数字**「章-节」
# （2-10、3-1 —— 主线的普通战斗 / 头目战就只显示这个，2026-09-29 因为不认它在章节地图上停了 7 次）。
# 纯数字的前面不能紧挨字母 / 数字 / 横杠：截断的「A-1-1」（其实是 ZA-1-1）不能当成「1-1」，
# 序列号「D868-558-882」也不能拆出编号来。
CODE_RE = re.compile(r'(?:[A-Z]{2}-|(?<![A-Z0-9-]))\d{1,2}-\d{1,2}(?!\d)')
LETTER_CODE_RE = re.compile(r'^[A-Z]{2}-')
# Something that looks like a damaged / partial code: 'SA-2', 'ZA-1-', 'A-1-1', '草25-12', '1-1', 'ZA-',
# and the numeric main-story codes split by the detector: '1.' + '-5', '2-' at a cut card
FRAG_RE = re.compile(r'[A-Z0-9]{1,2}-\d|\d-\d|[A-Z]{2}-$|^\d{1,2}[.-]$|^-\d{1,2}$')
# 主线卡片上方的类型标签。纯数字编号只有在画面上有这些标签（或者有字母编号）时才采信 ——
# 否则别的界面上偶然出现的「1-5」字样会被当成章节地图、甚至被点下去。
CARD_LABEL_RE = re.compile('战斗|剧情点')
# ...but not decoration / overlay text: '//D178-828-345', '12:55:13', 'CPU 36%'
JUNK_RE = re.compile(r'\d{3}|[/:%]')
_DASH_MAP = {ord(c): '-' for c in '‐‑‒–—―−一－_~'}

# ---------------------------------------------------------------- tunables (all in h_ref units)
PAD_CODE = (0.4, 0.35)       # crop padding (x, y) around a token that already contains a code
PAD_FRAG = (2.0, 0.5)        # around a fragment: room for the 1-2 missing characters on either side
PAD_FRAG_RETRY = (3.5, 1.0)  # second, larger try for a fragment that did not resolve
EXP_CODE_W = 3.24            # expected width of a 6-char code ('ZA-1-2') in h_ref units
MERGED_W = 1.6               # a token wider than 1.6 expected code widths is merged with graphics
PART_W = 1.5                 # ...then only 1.5 code widths at the code's end are re-OCR'd
PAD_REFINE = (0.5, 0.4)      # around the estimated sub-box of a code that is still merged in a crop
MAX_MERGED_WIN = (14.0, 5.0)  # overlapping crop windows are merged only up to this size
TOUCH_RATIO = 0.12           # a hit whose box is this close to an inner crop border may be truncated
CAP_RATIO = 0.66             # glyph cap height ~= 0.66 * det box height (measured 31-35 px vs 45-59 px boxes)
EDGE_GAP_CAP = 0.4           # truncated if the code's last glyph ink ends < 0.4 cap from the frame edge
                             # (inter-character gaps measured 1-9 px = <=0.28 cap; complete SA-2-x at the
                             #  right edge end 18 px = 0.55 cap before it; a cut 'ZA-1-4' ends at the edge)
DEEP_TILE_H = 37.0           # deep scan: tile long side = 37*h_ref  -> text ~26 px after det resize
DEEP_OVERLAP_H = 4.5         # tile overlap (a code is <= ~3.6 h_ref wide)
CARD_LEFT_W = 2.2            # card extends ~2.2 code widths left of the code (hexagons / lock / glyph)
CARD_RIGHT_W = 0.35          # ...and ~0.35 code widths right of it (completed check badge)
MIN_TEXT_PX = 40             # upscale crops (cubic) when h_ref is below this: at 0.5x-0.625x windows the
                             # 26-35 px codes otherwise split in the crop det ('SS-1' + '-1')
DET_LIMIT = 960              # onnxocr det_limit_side_len


def is_numeric_code(code):
    """'2-10' 这种主线纯数字编号（相对于 'BH-2-5' 这种带字母的）。"""
    return not LETTER_CODE_RE.match(code or '')


def numeric_codes_allowed(tokens):
    """这一帧能不能采信纯数字编号：画面上有卡片类型标签，或者已经有带字母的编号（那一定是章节地图）。"""
    for t in tokens or []:
        n = _tok(t)['norm']
        if CARD_LABEL_RE.search(n):
            return True
        if any(not is_numeric_code(m.group(0)) for m in CODE_RE.finditer(n)):
            return True
    return False


def norm_text(s):
    s = unicodedata.normalize('NFKC', str(s or '')).translate(_DASH_MAP).upper()
    return re.sub(r'\s+', '', s)


def _tok(t):
    if isinstance(t, dict):
        text, conf, x, y, w, h = t['text'], t.get('conf', 1.0), t['x'], t['y'], t['w'], t['h']
    elif hasattr(t, 'width') and hasattr(t, 'name'):
        text, conf, x, y, w, h = t.name, t.confidence, t.x, t.y, t.width, t.height
    else:
        text, conf, x, y, w, h = t[:6]
    return {'text': str(text), 'norm': norm_text(text), 'conf': float(conf),
            'x': int(round(x)), 'y': int(round(y)), 'w': max(1, int(round(w))), 'h': max(1, int(round(h)))}


def _classify(tok):
    """-> ('exact'|'embedded'|'fragment'|None, [(code, start, end)])"""
    n = tok['norm']
    # 纯数字编号只认「整个框就是它」：混在别的字里的（'草25-12' 其实是误读的 ZS-1-2）当碎片去重读
    hits = [(m.group(0), m.start(), m.end()) for m in CODE_RE.finditer(n)
            if not is_numeric_code(m.group(0)) or m.group(0) == n]
    if hits:
        if len(hits) == 1 and hits[0][0] == n:
            return 'exact', hits
        return 'embedded', hits
    if FRAG_RE.search(n) and not JUNK_RE.search(n):
        return 'fragment', []
    return None, []


def _char_w(c):
    if ord(c) > 0x2E80:
        return 1.0
    if c == '-':
        return 0.45
    if c.isdigit():
        return 0.55
    if c.isalpha():
        return 0.62
    return 0.4


def _code_w(code):
    """expected width of a code in h_ref units (~3.2 for 'ZA-1-2')"""
    return sum(_char_w(c) for c in code)


def _sub_box(tok, code, start, end, h_ref):
    """Geometric estimate of the code's own box inside a merged token.  The graphics that get
    merged (glyph, hexagons, waveform) sit LEFT of the code and are not part of the text, so a code
    at the end of the string is right-aligned in the box, one at the start is left-aligned."""
    n = tok['norm']
    x, y, w, h = tok['x'], tok['y'], tok['w'], tok['h']
    hh = min(h, 1.15 * h_ref)
    cw = min(w, _code_w(code) * h_ref)
    if end == len(n):
        x0 = x + w - cw
    elif start == 0:
        x0 = x
    else:
        ws = [_char_w(c) for c in n]
        x0 = x + w * sum(ws[:start]) / sum(ws)
    cy = y + h / 2.0
    return int(round(x0)), int(round(cy - hh / 2)), int(round(cw)), int(round(hh))


def _code_side(tok):
    """which end of a token holds the code(-like) text: 'right' / 'left' / None"""
    n = tok['norm']
    if tok.get('hits'):
        _, a, b = tok['hits'][-1]
    else:
        ms = list(FRAG_RE.finditer(n))
        if not ms:
            return None
        a, b = ms[0].start(), ms[-1].end()
        # extend over the rest of a code-like run ('草25-12' -> '25-12')
        while a > 0 and (n[a - 1].isalnum() and ord(n[a - 1]) < 128 or n[a - 1] == '-'):
            a -= 1
        while b < len(n) and (n[b].isalnum() and ord(n[b]) < 128 or n[b] == '-'):
            b += 1
    if b >= len(n) and a > 0:
        return 'right'
    if a == 0 and b < len(n):
        return 'left'
    return 'right' if b >= len(n) else None


def _estimate_h_ref(seeds, frame_h):
    hs = []
    for s in seeds:
        if s['kind'] == 'exact':
            r = s['w'] / float(s['h'])
            e = _code_w(s['norm'])
            if 0.7 * e <= r <= 1.15 * e:
                hs.append(s['h'])
    if hs:
        return float(np.median(hs))
    # no clean code: merged boxes are too tall, truncated reads too short -> smallest, but not below a
    # floor from the frame height (codes are ~0.029 H on a 16:9 client area, 49-53 px at 1728)
    hs = [s['h'] for s in seeds if s['kind']]
    if hs:
        return max(float(min(hs)), 0.02 * frame_h)
    return 0.029 * frame_h


def _pad_box(box, px, py, W, H):
    x, y, w, h = box
    return (max(0, int(x - px)), max(0, int(y - py)), min(W, int(x + w + px)), min(H, int(y + h + py)))


def _grow(a, b, target, L):
    """grow the span [a, b) symmetrically to exactly `target` inside [0, L) (shorter if L < target)"""
    target = int(round(target))
    if b - a >= target:
        return int(a), int(b)
    a2 = int(round((a + b - target) / 2.0))
    a2 = max(0, min(a2, L - target))
    return a2, min(L, a2 + target)


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _merge_windows(wins, h_ref):
    """union overlapping windows (fewer OCR calls), but never beyond MAX_MERGED_WIN"""
    wins = sorted(wins)
    out = []
    for w in wins:
        for i, o in enumerate(out):
            if _intersects(w, o):
                u = (min(w[0], o[0]), min(w[1], o[1]), max(w[2], o[2]), max(w[3], o[3]))
                if u[2] - u[0] <= MAX_MERGED_WIN[0] * h_ref and u[3] - u[1] <= MAX_MERGED_WIN[1] * h_ref:
                    out[i] = u
                    break
        else:
            out.append(w)
    return out


def _ink_chain(gray, box, h_ref):
    """x-extent (start, end) of the code's own glyph chain on the text-core rows.

    Ink = pixels on the other side of an Otsu threshold (computed inside the det box) than the det box's
    padding border (which is background).  Starting from the ink runs inside the box, the chain is extended
    to neighbouring runs separated by less than EDGE_GAP_CAP cap heights (= inter-character gaps), so a
    glyph cut by the frame edge makes the chain reach the edge."""
    H, W = gray.shape[:2]
    x, y, w, h = [int(v) for v in box]
    x, y = max(0, x), max(0, y)
    w, h = max(2, min(w, W - x)), max(2, min(h, H - y))
    cap = CAP_RATIO * min(h, 1.3 * h_ref)
    y0, y1 = y + int(0.2 * h), y + max(int(0.8 * h), int(0.2 * h) + 1)
    core = gray[y0:y1, x:x + w]
    t, _ = cv2.threshold(np.ascontiguousarray(core), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    per = np.concatenate([gray[y, x:x + w], gray[y + h - 1, x:x + w], gray[y:y + h, x], gray[y:y + h, x + w - 1]])
    bg_bright = float(np.median(per)) > t
    ext = int(1.5 * h_ref)
    xa, xb = max(0, x - ext), min(W, x + w + ext)
    reg = gray[y0:y1, xa:xb]
    ink = (reg <= t) if bg_bright else (reg > t)
    on = ink.mean(0) > 0.08
    idx = np.flatnonzero(on)
    if not len(idx):
        return None
    # runs of ink columns
    br = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[br + 1]] + xa
    ends = np.r_[idx[br], idx[-1]] + xa
    inside = [i for i in range(len(starts)) if ends[i] >= x + 0.1 * w and starts[i] <= x + 0.9 * w]
    if not inside:
        return None
    lo, hi = inside[0], inside[-1]
    gap = EDGE_GAP_CAP * cap
    while hi + 1 < len(starts) and starts[hi + 1] - ends[hi] - 1 < gap:
        hi += 1
    while lo > 0 and starts[lo] - ends[lo - 1] - 1 < gap:
        lo -= 1
    return int(starts[lo]), int(ends[hi]), cap, [(int(a), int(b)) for a, b in zip(starts[lo:hi + 1], ends[lo:hi + 1])]


def _truncated(gray, box, h_ref):
    """'left'/'right'/'top'/'bottom' if the code may be cut by the frame edge, else None"""
    H, W = gray.shape[:2]
    x, y, w, h = box
    if y <= 1:
        return 'top'
    if y + h >= H - 1:
        return 'bottom'
    ch = _ink_chain(gray, box, h_ref)
    if ch is None:                                   # no ink found: fall back to the det box itself
        m = 2 + 0.1 * h
        return 'left' if x < m else ('right' if x + w > W - m else None)
    a, b, cap, _ = ch
    if b >= W - 1 - EDGE_GAP_CAP * cap:
        return 'right'
    if a <= EDGE_GAP_CAP * cap:
        return 'left'
    return None


def _overlap_frac(a, b):
    """intersection / smaller area, boxes as (x, y, w, h)"""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / float(max(1, min(a[2] * a[3], b[2] * b[3])))


class _Runner:
    def __init__(self, frame, ocr_fn, h_ref, max_calls):
        self.frame, self.ocr_fn, self.h_ref, self.max_calls = frame, ocr_fn, h_ref, max_calls
        self.H, self.W = frame.shape[:2]
        self.calls = []

    def ocr(self, win, tag):
        """OCR a window (x0, y0, x1, y1) -> (tokens in FRAME coords, window actually used), or (None, win)
        when the call budget is spent.

        onnxocr's DetResizeForTest keeps a small image at ratio 1 but rounds each side to a multiple of 32,
        so a 109x42 crop is detected as 96x32 (squashed 0.76x vertically) and small codes fall apart
        ('AA-2-' + '-4').  So windows are grown (into real frame context) to multiples of 32, >= 64 px."""
        if len(self.calls) >= self.max_calls:
            return None, win
        x0, y0, x1, y1 = win
        f = 1.0
        if MIN_TEXT_PX and self.h_ref < MIN_TEXT_PX:
            # upscale small text, but only as far as the detector keeps it (it shrinks anything > 960 px)
            f = max(1.0, min(3.0, MIN_TEXT_PX / float(self.h_ref), DET_LIMIT / float(max(x1 - x0, y1 - y0))))
        tw, th = (x1 - x0) * f, (y1 - y0) * f
        if max(tw, th) <= DET_LIMIT:
            tw, th = 32 * np.ceil(max(tw, 64) / 32.0), 32 * np.ceil(max(th, 64) / 32.0)
            x0, x1 = _grow(x0, x1, tw / f, self.W)
            y0, y1 = _grow(y0, y1, th / f, self.H)
        win = (x0, y0, x1, y1)
        crop = self.frame[y0:y1, x0:x1]
        if f != 1.0:
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        if max(tw, th) <= DET_LIMIT:
            # pad the rounding remainder / a frame smaller than the window (replicated border, no fake edges)
            pw, ph = int(tw) - crop.shape[1], int(th) - crop.shape[0]
            if pw > 0 or ph > 0:
                crop = cv2.copyMakeBorder(crop, 0, max(0, ph), 0, max(0, pw), cv2.BORDER_REPLICATE)
        t = time.perf_counter()
        res = self.ocr_fn(np.ascontiguousarray(crop)) or []
        dt = time.perf_counter() - t
        self.calls.append({'tag': tag, 'win': win, 'size': (crop.shape[1], crop.shape[0]), 'time': dt})
        out = []
        for r in res:
            tk = _tok(r)
            tk['x'] = int(round(x0 + tk['x'] / f)); tk['y'] = int(round(y0 + tk['y'] / f))
            tk['w'] = max(1, int(round(tk['w'] / f))); tk['h'] = max(1, int(round(tk['h'] / f)))
            # clip to the real window (the replicated padding is not frame content)
            tk['w'] = max(1, min(tk['w'], x1 - tk['x'])); tk['h'] = max(1, min(tk['h'], y1 - tk['y']))
            out.append(tk)
        return out, win

    def touches_inner_border(self, box, win):
        """True if box is within TOUCH_RATIO*h of a window side that is NOT a frame side
        (the crop itself may have cut the text there)."""
        x, y, w, h = box
        m = TOUCH_RATIO * max(h, self.h_ref)
        x0, y0, x1, y1 = win
        return ((x0 > 0 and x - x0 < m) or (x1 < self.W and x1 - (x + w) < m) or
                (y0 > 0 and y - y0 < m) or (y1 < self.H and y1 - (y + h) < m))


def _hits_from(tokens, source, h_ref):
    """code candidates from a token list"""
    cands = []
    for tk in tokens:
        kind, hits = _classify(tk)
        # an exact text whose box is far too wide is merged with the graphics beside it (' ZA-1-2' 427x112)
        clean = kind == 'exact' and tk['w'] <= MERGED_W * _code_w(tk['norm']) * h_ref
        for code, s, e in hits:
            box = (tk['x'], tk['y'], tk['w'], tk['h']) if clean else _sub_box(tk, code, s, e, h_ref)
            cands.append({'code': code, 'box': box, 'conf': tk['conf'], 'source': source,
                          'exact': clean, 'raw': tk['text'], 'raw_box': (tk['x'], tk['y'], tk['w'], tk['h'])})
    return cands


_RANK = {('crop', True): 0, ('deep', True): 1, ('crop', False): 2, ('full', True): 3,
         ('deep', False): 4, ('full', False): 5}


def _crop_verify(run, seeds, h_ref, cands, log):
    """re-OCR windows around seeds; append crop candidates; returns seeds that stayed unresolved"""
    W, H = run.W, run.H
    wins = []
    for s in seeds:
        box = (s['x'], s['y'], s['w'], s['h'])
        pad = PAD_FRAG if s['kind'] == 'fragment' else PAD_CODE
        side = _code_side(s)
        if s['w'] > MERGED_W * EXP_CODE_W * h_ref and side:
            # merged with the graphics beside the code (gold glyph / hexagons / waveform): the whole token
            # re-merges even at native resolution ('十草2s-1-2'), so crop only the side holding the code text
            cw = PART_W * EXP_CODE_W * h_ref
            box = (s['x'] + s['w'] - cw, s['y'], cw, s['h']) if side == 'right' else (s['x'], s['y'], cw, s['h'])
            pad = PAD_CODE
        s['win'] = _pad_box(box, pad[0] * h_ref, pad[1] * h_ref, W, H)
        wins.append(s['win'])
    new = []
    for win in _merge_windows(wins, h_ref):
        toks, win = run.ocr(win, 'verify')
        if toks is None:
            break
        for c in _hits_from(toks, 'crop', h_ref):
            c['win'] = win
            new.append(c)
    # hits that touch an inner crop border might be truncated by the crop -> widen once and redo
    fixed = []
    for c in new:
        if run.touches_inner_border(c['box'], c['win']):
            win2 = _pad_box(c['box'], 2.0 * h_ref, 0.6 * h_ref, W, H)
            toks, win2 = run.ocr(win2, 'widen')
            log.append('crop hit %s touched crop border, widened' % c['code'])
            if toks:
                for c2 in _hits_from(toks, 'crop', h_ref):
                    c2['win'] = win2
                    if not run.touches_inner_border(c2['box'], win2):
                        fixed.append(c2)
            continue
        fixed.append(c)
    # a code still merged inside the crop (text longer than the code): one tight re-OCR
    out = []
    for c in fixed:
        if not c['exact']:
            win2 = _pad_box(c['box'], PAD_REFINE[0] * h_ref, PAD_REFINE[1] * h_ref, W, H)
            toks, win2 = run.ocr(win2, 'refine')
            got = [c2 for c2 in _hits_from(toks or [], 'crop', h_ref)
                   if c2['exact'] and c2['code'] == c['code'] and not run.touches_inner_border(c2['box'], win2)]
            if got:
                got[0]['win'] = win2
                out.append(got[0])
                continue
        out.append(c)
    cands.extend(out)
    # fragments that produced no code at their location: one bigger retry
    unresolved = []
    for s in seeds:
        sb = (s['x'], s['y'], s['w'], s['h'])
        if any(_overlap_frac(sb, c['box']) > 0.3 for c in out):
            continue
        if s['kind'] == 'fragment' and s['x'] > 0.5 * h_ref and s['x'] + s['w'] < W - 0.5 * h_ref:
            win2 = _pad_box(sb, PAD_FRAG_RETRY[0] * h_ref, PAD_FRAG_RETRY[1] * h_ref, W, H)
            toks, win2 = run.ocr(win2, 'retry')
            got = [c for c in _hits_from(toks or [], 'crop', h_ref) if not run.touches_inner_border(c['box'], win2)]
            got = [c for c in got if _overlap_frac(sb, c['box']) > 0.3]
            for c in got:
                c['win'] = win2
            cands.extend(got)
            if got:
                continue
        unresolved.append(s)
    return unresolved


def _deep_scan(run, h_ref):
    """tile the frame so the detector sees text at ~26 px; returns deep candidates"""
    W, H = run.W, run.H
    T = max(DET_LIMIT, DEEP_TILE_H * h_ref)
    ov = DEEP_OVERLAP_H * h_ref

    def spans(L):
        if L <= T:
            return [(0, L)]
        n = int(np.ceil((L - ov) / (T - ov)))
        step = (L - ov) / n
        return [(int(round(i * step)), int(round(min(L, i * step + step + ov)))) for i in range(n)]

    cands = []
    for (y0, y1) in spans(H):
        for (x0, x1) in spans(W):
            win = (x0, y0, x1, y1)
            toks, win = run.ocr(win, 'deep')
            if toks is None:
                return cands
            for c in _hits_from(toks, 'deep', h_ref):
                if not run.touches_inner_border(c['box'], win):
                    c['win'] = win
                    cands.append(c)
    return cands


def _select(cands):
    """cluster candidates by location, best source wins; then one location per code"""
    cands = sorted(cands, key=lambda c: (_RANK[(c['source'], c['exact'])], -c['conf']))
    clusters = []
    for c in cands:
        for cl in clusters:
            if _overlap_frac(cl[0]['box'], c['box']) > 0.3:
                cl.append(c)
                break
        else:
            clusters.append([c])
    best, dropped = {}, []
    for cl in clusters:
        w = cl[0]
        w['alts'] = sorted({c['code'] for c in cl if c['code'] != w['code']})
        w['n_votes'] = sum(1 for c in cl if c['code'] == w['code'])
        o = best.get(w['code'])
        if o is None:
            best[w['code']] = w
        else:
            a, b = sorted([o, w], key=lambda c: (_RANK[(c['source'], c['exact'])], -c['n_votes'], -c['conf']))
            best[w['code']] = a
            dropped.append(dict(b, reason='duplicate code, weaker read'))
    return list(best.values()), dropped


def extract_stage_codes(frame, tokens, ocr_fn, deep='auto', verify='all', details=False, max_calls=24,
                        allow_numeric=None):
    """Robust stage-code extraction.

    :param frame:  BGR frame the tokens were read from
    :param tokens: full-frame OCR tokens (ok Box list / tuples / dicts)
    :param ocr_fn: ocr_fn(bgr_crop) -> tokens in crop coordinates
    :param deep:   False = never tile-scan; True = always; 'auto' = only when pass 1 misread or truncated a
                   code that turned out to be fully visible (evidence that the full-frame detector misses
                   low-contrast codes on this map)
    :param verify: 'all' = re-OCR a crop around every code-like token (tight, consistent boxes);
                   'suspicious' = keep clean exact tokens with a plausible box as they are
    :param details: also return (accepted_dicts, rejected_dicts, info)
    :param max_calls: budget of extra OCR calls (crops + deep tiles)
    :param allow_numeric: 采信主线纯数字编号（'2-10'）吗；None = 看这一帧有没有卡片类型标签 / 字母编号
    :return: [(code, (x, y, w, h))] left->right (a vertical stack top->bottom); codes that may be cut by the
             frame edge are rejected.  details=True -> (result, accepted, rejected, info); each accepted dict
             has code/box/conf/source('crop'|'deep'|'full')/exact/alts(other codes read at the same place)/
             left_cut/right_cut (the CARD, not the code, extends past the frame: its lock/hexagons resp. its
             check badge may be off-screen, so the state classifier should treat it as uncertain)
    """
    H, W = frame.shape[:2]
    if allow_numeric is None:
        allow_numeric = numeric_codes_allowed(tokens)
    seeds = []
    for t in tokens or []:
        tk = _tok(t)
        kind, hits = _classify(tk)
        if not kind:
            continue
        tk['kind'], tk['hits'] = kind, hits
        seeds.append(tk)
    h_ref = _estimate_h_ref(seeds, H)
    # fragments much smaller than a code are decoration (style A's '//D178-828-345' is already JUNK). Pass-1
    # boxes are rough (det runs at ~0.31x): the truncated 'SA-2' read of SA-2-1 is only 0.37 h_ref tall.
    seeds = [s for s in seeds if not (s['kind'] == 'fragment' and (s['h'] < 0.3 * h_ref or s['w'] > 20 * h_ref))]
    run = _Runner(frame, ocr_fn, h_ref, max_calls)
    log = []

    cands = _hits_from(seeds, 'full', h_ref)
    for c in cands:
        c['win'] = None

    def plausible(s):
        e = _code_w(s['norm'])
        return (s['kind'] == 'exact' and s['h'] <= 1.35 * h_ref and 0.7 * e <= s['w'] / float(s['h']) <= 1.15 * e)

    to_verify = [s for s in seeds if verify == 'all' or not plausible(s)]
    unresolved = _crop_verify(run, to_verify, h_ref, cands, log) if to_verify else []

    # refine h_ref from native-resolution crop reads (more accurate, used for edge margin / deep tiles)
    hs = [c['box'][3] for c in cands if c['source'] == 'crop' and c['exact']]
    if hs:
        h_ref = float(np.median(hs))
        run.h_ref = h_ref

    selected, dropped = _select(cands)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def at_edge(box):
        return _truncated(gray, box, h_ref)

    # deep-scan trigger: a NORMAL-SIZE fragment (a truncated read, not a merge with a glyph) resolved into a
    # complete code that is not at the frame edge -> the full-frame detector struggles with this map's text
    repaired = []
    for s in seeds:
        if s['kind'] != 'fragment' or s['h'] > 1.35 * h_ref:
            continue
        sb = (s['x'], s['y'], s['w'], s['h'])
        for c in selected:
            if c['source'] == 'crop' and _overlap_frac(sb, c['box']) > 0.3 and not at_edge(c['box']):
                repaired.append('%s->%s' % (s['text'], c['code']))
    # ...or a fragment that stayed unreadable although it is not cut by the left/right frame edge
    stray = [s['text'] for s in unresolved if s['kind'] == 'fragment'
             and s['x'] >= h_ref and s['x'] + s['w'] <= W - h_ref]
    run_deep = deep is True or (deep == 'auto' and bool(repaired or stray))
    if run_deep:
        deep_c = _deep_scan(run, h_ref)
        # codes only the deep scan saw: verify with a tight crop so their boxes are consistent
        new = [c for c in deep_c if not any(_overlap_frac(c['box'], s['box']) > 0.3 for s in selected)]
        cands.extend(deep_c)
        if new:
            fake = [dict(text=c['raw'], norm=norm_text(c['raw']), conf=c['conf'], x=c['box'][0], y=c['box'][1],
                         w=c['box'][2], h=c['box'][3], kind='exact' if c['exact'] else 'embedded', hits=[])
                    for c in new]
            _crop_verify(run, fake, h_ref, cands, log)
        selected, dropped = _select(cands)

    accepted, rejected = [], list(dropped)
    for c in selected:
        if not allow_numeric and is_numeric_code(c['code']):
            rejected.append(dict(c, reason='numeric code but no chapter-map context on screen'))
            continue
        x, y, w, h = c['box']
        c['edge'] = at_edge(c['box'])
        c['left_cut'] = x - CARD_LEFT_W * w < 0
        c['right_cut'] = x + w + CARD_RIGHT_W * w > W
        if c['edge']:
            rejected.append(dict(c, reason='cut by frame edge (%s)' % at_edge(c['box'])))
        else:
            accepted.append(c)
    # left -> right; codes stacked in one column (branch nodes like SA-2-1/2/3) top -> bottom
    accepted.sort(key=lambda c: c['box'][0])
    cols, ordered = [], []
    for c in accepted:
        if cols and c['box'][0] - cols[-1][0]['box'][0] < h_ref:
            cols[-1].append(c)
        else:
            cols.append([c])
    for col in cols:
        ordered.extend(sorted(col, key=lambda c: c['box'][1]))
    accepted = ordered
    result = [(c['code'], tuple(int(v) for v in c['box'])) for c in accepted]
    if not details:
        return result
    seed_map = []
    for s in seeds:
        sb = (s['x'], s['y'], s['w'], s['h'])
        hit = [c for c in accepted + rejected if 'box' in c and _overlap_frac(sb, c['box']) > 0.3]
        seed_map.append((s['text'], sb, s['kind'], [(c['code'], 'ok' if c in accepted else c['reason']) for c in hit]))
    info = {'h_ref': h_ref, 'calls': run.calls, 'seeds': seed_map, 'extra_time': sum(c['time'] for c in run.calls),
            'deep': run_deep, 'deep_trigger': repaired + stray, 'log': log,
            'unresolved': [s['text'] for s in unresolved]}
    return result, accepted, rejected, info


# ---- 只留白字再读（I-56）
# 10/13 14:20「逃离赛博格」：卡片上的白色粗体编号 VA-1-1 上面压着一行金色花体水印「victor」，正好穿过编号下半截，
# 整屏 OCR 读成「τ1=1」「-1=1t0」「VA-1=2-0」，一串编号都读不出来，停了 40 分钟。白字是低饱和、高亮度，水印是金色（饱和）：
# 只留白字（变成白底黑字）再读，四张现场图上 VA-1-1～1-4、VF-1-1～1-3 全读对了，普通样式（密米尔 SA / SS）也读得对
WHITE_TEXT_S_MAX = 60
WHITE_TEXT_V_MIN = 170
# 编号里不会有、水印划过横杠时常读出来的字：卡片带里有带它的字块，就当这一帧的编号被盖住了
GARBLED_RE = re.compile(r'\d.*[=τ]|[=τ].*\d')


def white_text(frame, s_max=WHITE_TEXT_S_MAX, v_min=WHITE_TEXT_V_MIN):
    """只留白字：低饱和、高亮度的像素变黑，别的都变白（BGR，尺寸和坐标不变）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = np.full(frame.shape[:2], 255, np.uint8)
    out[(hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min)] = 0
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def garbled_codes(tokens, H, band=(0.18, 0.80)):
    """卡片带里像是被盖住的编号（「τ1=1」「VA-1=2-0」）。"""
    out = []
    for t in tokens or ():
        tk = _tok(t)
        cy = tk['y'] + tk['h'] / 2.0
        if band[0] * H <= cy <= band[1] * H and GARBLED_RE.search(tk['norm']) and len(tk['norm']) <= 12:
            out.append(tk['text'])
    return out


def has_stage_code(tokens):
    """cheap test for 'this is a chapter map' that also accepts merged tokens like '学ZS-1-2'
    and codes split in two tokens ('LA' + '-1-8', see split_code_joins).

    带字母的编号直接算；纯数字的（'2-10'）要画面上同时有卡片类型标签才算。
    """
    numeric = False
    for t in tokens or []:
        for m in CODE_RE.finditer(_tok(t)['norm']):
            if not is_numeric_code(m.group(0)):
                return True
            numeric = True
    if numeric and numeric_codes_allowed(tokens):
        return True
    return bool(split_code_joins(tokens))


# 整屏 OCR 把一个编号断成两截（10/13「瑞骋迎春至」：橙黑卡片上的白字，背后是笔刷纹理）：
# 'LA' + '-1-8'、'LA' + 'A-1-1'（两个框重叠，A 读了两遍）。同一行、左右挨着才拼
SPLIT_LEFT_RE = re.compile(r'^[A-Z]{2}-?$')
SPLIT_RIGHT_RE = re.compile(r'^(?:[A-Z]-|-)?\d{1,2}-\d{1,2}$')
LETTER_CODE_FULL_RE = re.compile(r'^[A-Z]{2}-\d{1,2}-\d{1,2}$')
SPLIT_GAP_H = 0.8            # 两截之间的空隙 / 重叠最多这么多个字高
SPLIT_DY_H = 0.5             # 两截的中线最多差这么多个字高


def split_code_joins(tokens):
    """断成两截的编号拼起来 -> [编号]（只拼带字母的编号；纯数字的另有卡片类型标签那条规矩）。"""
    toks = [_tok(t) for t in tokens or ()]
    out = []
    for a in toks:
        if not SPLIT_LEFT_RE.match(a['norm']):
            continue
        left = a['norm'].rstrip('-')
        for b in toks:
            if b is a or not SPLIT_RIGHT_RE.match(b['norm']):
                continue
            h = max(a['h'], b['h'])
            if abs((a['y'] + a['h'] / 2.0) - (b['y'] + b['h'] / 2.0)) > SPLIT_DY_H * h:
                continue
            gap = b['x'] - (a['x'] + a['w'])
            if not -SPLIT_GAP_H * h <= gap <= SPLIT_GAP_H * h:
                continue
            right = b['norm']
            if right[0] == '-':
                code = left + right                     # 'LA' + '-1-8'
            elif right[0].isalpha():
                if right[0] != left[-1] or gap >= 0:
                    continue                            # 'LS' + 'A-1-1'：拿不准是哪个字读错了，不拼
                code = left[:-1] + right                # 'LA' + 'A-1-1'（重叠的 A 读了两遍）
            else:
                code = left + '-' + right               # 'LA' + '1-8'
            if LETTER_CODE_FULL_RE.match(code) and code not in out:
                out.append(code)
    return out


def repair_merged(frame, accepted, h_ref, ocr_fn):
    """裁剪重读之后仍然「又高又宽」的框（高 > 1.4·h_ref 或宽 > 1.3 倍应有宽度）再修一次。

    FIX-C（离线核实时发现）：h_ref 是在裁剪重读**之前**估的，并框会把它抬高（38 而不是 25），
    于是「嶄 + ZS-1-2」这种并框被当成干净编号收下，还读成了「ZS-1-22」、框往左偏了 4 倍字高。
    编号是右对齐的文字、并框总是并在左边，所以只取框的右端重读；读不出来就把框右对齐收窄。
    :return: (accepted, 修了几个)
    """
    run = _Runner(frame, ocr_fn, h_ref, 6)
    H, W = frame.shape[:2]
    n = 0
    for c in accepted:
        x, y, w, h = c['box']
        ew = _code_w(c['code']) * h_ref
        if h <= 1.4 * h_ref and w <= 1.3 * ew:
            continue
        cy = y + h / 2.0
        ew6 = _code_w('ZA-1-2') * h_ref          # 6 个字符的编号：读错的可能多带一个字
        sub_w = max(ew6, min(ew, w))
        sub = (x + w - sub_w, cy - 0.6 * h_ref, sub_w, 1.2 * h_ref)
        win = _pad_box(sub, 0.6 * h_ref, 0.45 * h_ref, W, H)
        toks, win = run.ocr(win, 'fixC')
        got = [c2 for c2 in _hits_from(toks or [], 'crop', h_ref)
               if c2['exact'] and not run.touches_inner_border(c2['box'], win) and c2['box'][3] <= 1.4 * h_ref]
        if got:
            g = max(got, key=lambda c2: c2['box'][0] + c2['box'][2])
            c['code'], c['box'] = g['code'], g['box']
        else:
            c['box'] = anchor_right(c['box'], h_ref, c['code'], k=1.0)
        n += 1
    return accepted, n


def anchor_right(box, ref_h, code, k=1.25):
    """框比这个编号应有的宽度宽出 k 倍以上 —— 左边并进了图形，保留右边缘、收窄到应有宽度。"""
    x, y, w, h = box
    if not ref_h:
        return box
    ew = _code_w(code) * ref_h
    if w > k * ew:
        return (x + w - ew, y, ew, h)
    return box
