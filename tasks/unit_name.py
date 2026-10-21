# -*- coding: utf-8 -*-
"""左下角详情卡（「支援」+ 名字 + Lv.xx）解析，和「是不是同一个角色」判定（ok-gf2 扩展，补人用，见 own_units）。
标 [fixed] 的是对抗验证之后补的。

不 import ok；文本部分只用 re / unicodedata，敌人血条检测（可选）用 numpy / cv2（函数内导入）。

详情卡区域沿用 support_pick.support_detail_box：整屏 (0.02,0.68)-(0.34,0.94)。
实测版面（2560x1440 / 3072x1728 / 3200x1800，另把图缩放到 1920/2240/3840 宽复测；全部按整屏 W/H）：
  - 「支援」角标（只有好友助战才有）：框左沿 x 0.026~0.027，中心 y 0.837~0.838
  - 名字：框左沿 x 0.1034~0.1056，中心 y 0.842~0.844，框高 0.024~0.031（缩到 1920 宽时 2 字名 0.037），
    每个汉字宽 0.0128~0.0139 W（3 字及以上；2 字名带边距，0.0141~0.0161）
  - Lv.xx：框左沿 x 0.1879~0.1898，中心 y 0.845~0.847（名字长时 OCR 会并成「纳美西丝·衍光Lv.45」）
  - 下一行是数值（攻/防/暴击/爆伤），中心 y ≈0.90；再下面是盾值「9/9」
  - 区域上半截（y 0.68~0.82）是地图。敌人头顶的名字条（「精锐·游空兵」+ 血条）是地图上的
    东西，可以落在名字行上：10/02 17:31 OCR 读出过「威玛西娜精锐·游空LV.60」，就是
    助战威玛西娜和背后敌人「精锐·游空兵」的名字（173130 帧里这个敌人名字就在 x 0.151、y 0.837，
    血条在 y 0.862~0.869，正好压在名字条底边）。名字字形被敌人的字盖住时，OCR 会读出
    「感玛娜空兵」这种乱码（合成测试），这时只靠文字救不回来，见 find_enemy_bar。

入口：
  parse_unit_card(texts)                        只有字符串（按从上到下、从左到右排好）
  parse_unit_card_boxes(boxes, W, H, frame=None) 带坐标（ok 的 Box 或 (text,x,y,w,h[,conf])），**推荐**：
                                                按名字条位置挑名字；检查字形（置信度/字高/字宽）；
                                                给了 frame 还会找压在卡上的敌人血条
  same_character(a, b) / card_matches(card, name) / cards_same_character(card_a, card_b)
返回 dict(support, name, level, candidates, ambiguous, reason)。

调用方的保守规则：
  - card['name'] is None -> 读不出是谁（reason 说明为什么）-> 不能出击这个人形。
  - 判断「这个自有人形是不是助战那个角色」用 card_matches(card, 助战名) 或
    cards_same_character(card, 助战card)：任一候选名对上就算同一个角色。
"""
import re
import unicodedata

# ---------------------------------------------------------------- 版面常量（相对整屏）

DETAIL_BOX = (0.02, 0.68, 0.34, 0.94)
# 名字框左沿实测 0.1034~0.1056；「支援」角标在 0.026，立绘在 0.03~0.09。
NAME_X0_RANGE = (0.090, 0.120)
# 名字行中心 y 实测 0.842~0.844（Lv 0.845~0.847）；数值行中心 ≈0.90。
NAME_ROW_YC_RANGE = (0.815, 0.872)
# 同一行：中心 y 差不超过这个值（名字/Lv 实测差 ≤0.0035；数值行差 ≈0.06）。
ROW_DY = 0.015

# 名字框的字形检查。干净读数实测（原生 2560/3072/3200 宽 + 缩放到 2240/3840 宽复测）：
#   置信度 汉字名 ≥0.991、OTs-14 ≥0.971、名字+Lv 并框 ≥0.957；框高 0.0236~0.0310；
#   3 字以上纯汉字名每字宽 0.0128~0.0139。（缩到 1920 宽时 2 字名框高到 0.037、OTs-14 置信度 0.920，
#   会被判 None —— 只损失可用性；本机窗口都 ≥2560 宽。）
NAME_MIN_CONF = 0.90
# [fixed] 下限 0.018 -> 0.0215：敌人头顶名字条的字（区域 OCR）框高实测 0.0161~0.0201
# （防卫兵 0.0167/0.0185、狙击兵I 0.0183、好兵 0.0201、冲锋兵I 0.0179、冲锋兵·燃烧 0.0178、
# 准星圣盾·电导I 0.0174、重机枪模组 0.0194），旧下限 0.018 放得进大部分敌人字；干净名字 ≥0.0236。
NAME_H_RANGE = (0.0215, 0.036)
# [fixed] 上限 0.0152 -> 0.0148（干净读数 0.0128~0.0139；敌人字盖在名字上时 4 字名读成「狙染莱希」「冲染莱希」
# 框变宽到每字 0.0147~0.0151）。
CJK_CHAR_W_RANGE = (0.0120, 0.0148)
# [fixed] 2 字纯汉字名也查字宽（以前整个跳过）：干净 0.0138~0.0161（含 1920 宽缩放），字被挡/被敌人字
# 顶替时 OCR 框跨着原来 3~4 个字的宽度，读出「希烧」「维组」「重塔」「帶雅」这种 2 字，每字 0.0265~0.0275。
CJK2_CHAR_W_RANGE = (0.0125, 0.0175)
# [fixed] 纯汉字名字框（没并进 Lv）干净读数置信度 ≥0.991；被敌人字盖住读坏的「中美齿丝·衍光」0.950、
# 「冲染莱希」0.955、「希烧」0.966 —— 纯汉字名要求 ≥0.97。拉丁名（OTs-14 0.965~0.991）不受影响。
CJK_NAME_MIN_CONF = 0.97
# [fixed] 名字+Lv 并成一个框时干净读数置信度 0.957~0.974；读坏的「纳美画模光Lv.45」0.914、
# 「重纳樊丝·衍光Lv.45」0.933 —— 并框要求 ≥0.945。
MERGED_MIN_CONF = 0.945

SUPPORT_TAG = '支援'
# 敌人精英单位的名字前缀：「精锐·游空兵」「精锐·巡行督导」……人形名字里不会有。
ENEMY_MARKERS = ('精锐',)
# [fixed] 普通敌人名字几乎都带罗马数字等级后缀（防卫兵I、冲锋兵II、重刃圣盾I、超级兵蚁Ⅱ），OCR 读成
# I/l/1/|/!/Ⅱ。人形名字没有「汉字 + 这种后缀」的写法 -> 名字框里出现就当混进了敌人字。
# [fixed] 敌人名字里的常见字词（防卫兵/狙击兵/冲锋兵/好兵/游空兵、X圣盾、重机枪模组、巡行督导、莫普1.5·试作版）。
# 人形名字里没有。名字框（Lv 之前、切掉「精锐…」之后）里出现 -> 名字字形和敌人字混在一个框里，
# 实测这时 OCR 常把真名读坏（合成：「佩里缇亚」->「佩里亚防卫兵」、「OTs-14」->「OT14卫兵」）-> None。
ENEMY_WORDS = ('兵', '圣盾', '模组', '督导', '试作')
_ENEMY_SUFFIX_RE = re.compile(r'[%s][IlΙ1|!！ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*$' % ('%s-%s%s-%s' % (chr(0x3400), chr(0x4DBF), chr(0x4E00), chr(0x9FFF))))

LV_RE = re.compile(r'[Ll][Vv]\s*[.。,，:：·]?\s*(\d{1,3})(?!\d)')
LEVEL_RANGE = (1, 120)

_DOTS = str.maketrans({c: '·' for c in '·•‧∙⋅・･﹒'})
_DASHES = str.maketrans({c: '-' for c in '‐‑‒–—―－﹣'})
_EDGE_JUNK = ' \t|_[]【】「」『』()（）<>《》"\'`~:：;；,，。!！?？*·.-'
_CJK = '%s-%s%s-%s' % (chr(0x3400), chr(0x4DBF), chr(0x4E00), chr(0x9FFF))
_NAME_CHARS_RE = re.compile(r'^[%sA-Za-z0-9·\-.]+$' % _CJK)
_CJK_RE = re.compile(r'[%s]' % _CJK)
_CJK_ONLY_RE = re.compile(r'^[%s·]+$' % _CJK)
_LATIN_RE = re.compile(r'[A-Za-z]')

# 繁->简：只收名字里可能出现、OCR 可能读成繁体的字（可露凱 -> 可露凯）。只用于比较，
# 而且两边都做 —— 只会让「同一个角色」更容易成立（安全方向）。
_T2S = str.maketrans(
    '凱瑪絲維麗婭羅蘭爾諾納華樂薩齊衛銳遊傑亞蓮達倫煙葉嵐雲龍鳳紗緹萊偉韋貝溫賽麥彌瓊頓寧靈蘿夢櫻鈴銀鐵風紅藍綠黃純潔響聲',
    '凯玛丝维丽娅罗兰尔诺纳华乐萨齐卫锐游杰亚莲达伦烟叶岚云龙凤纱缇莱伟韦贝温赛麦弥琼顿宁灵萝梦樱铃银铁风红蓝绿黄纯洁响声')
# 拉丁字母/数字里 OCR 常混的几对，两边都折叠（同样只往「更容易相同」的方向走）。
_LATIN_FOLD = str.maketrans({'0': 'o', '1': 'l', 'i': 'l', '|': 'l'})


# ---------------------------------------------------------------- 文本工具

def _nfkc(s):
    return unicodedata.normalize('NFKC', s or '').translate(_DOTS).translate(_DASHES)


def _squash(s):
    return re.sub(r'\s+', '', _nfkc(s))


def _name_like(s):
    """清洗后的字符串像不像一个人形名字：≥2 字，只含汉字/字母/数字/·/-/.，
    且至少有 1 个汉字或 2 个字母（排除「211」「8/8」「20.0%」这类数值）。"""
    if not s or len(s) < 2 or not _NAME_CHARS_RE.match(s):
        return False
    return bool(_CJK_RE.search(s)) or len(_LATIN_RE.findall(s)) >= 2


def _clean_name(s):
    """从一段 OCR 文字里抠出名字：去掉「支援」前缀、敌人「精锐…」及其后面的字、空白和
    两端符号；不像名字返回 None。"""
    s = _squash(s)
    if s.startswith(SUPPORT_TAG):
        s = s[len(SUPPORT_TAG):]
    for mk in ENEMY_MARKERS:
        i = s.find(mk)
        if i >= 0:
            s = s[:i]
    s = s.strip(_EDGE_JUNK)
    return s if _name_like(s) else None


def _parse_level(m):
    v = int(m.group(1))
    return v if LEVEL_RANGE[0] <= v <= LEVEL_RANGE[1] else None


def _has_support_tag(texts):
    return any(_squash(t).startswith(SUPPORT_TAG) for t in texts)


def _uniq(seq):
    out = []
    for x in seq:
        if x and x not in out:
            out.append(x)
    return out


def _result(support, name=None, level=None, candidates=(), ambiguous=False, reason=None):
    return {'support': bool(support), 'name': name, 'level': level,
            'candidates': list(candidates), 'ambiguous': bool(ambiguous), 'reason': reason}


# ---------------------------------------------------------------- 解析：只有字符串

def parse_unit_card(texts):
    """解析详情卡区域里的 OCR 文字（按从上到下、从左到右排好；ok 的 sort_boxes 同一行按 x）。

    :return: dict(support: bool, name: str|None, level: int|None,
                  candidates: [str], ambiguous: bool, reason: str|None)
      - support: 出现了「支援」角标（好友助战）
      - name: 第一个「Lv.xx」前面的名字（同一个字块里 Lv 前的部分，否则最近的前一个
        像名字的字块）。没有 Lv、或者有两个以上都像名字的字块（ambiguous）-> None。
      - candidates: 所有可能是名字的字块（离 Lv 最近的在前），给 card_matches 用。
    只有字符串时分不清「名字条上的字」和「卡片上方地图里敌人的名字」，也做不了字形检查，
    所以有歧义就给 None；手里有坐标就用 parse_unit_card_boxes。
    """
    texts = [str(t) for t in (texts or []) if t and str(t).strip()]
    support = _has_support_tag(texts)
    for i, t in enumerate(texts):
        m = LV_RE.search(_nfkc(t))
        if m:
            break
    else:
        return _result(support, candidates=_uniq(_clean_name(t) for t in texts), reason='no_lv')
    level = _parse_level(m)
    here = _clean_name(_nfkc(texts[i])[:m.start()])
    before = _uniq(_clean_name(t) for t in reversed(texts[:i]))
    candidates = _uniq([here] + before)
    if len(candidates) > 1:
        return _result(support, None, level, candidates, True, 'ambiguous')
    if not candidates:
        return _result(support, None, level, (), False, 'no_name')
    # [fixed] 只有字符串时也查一遍敌人名字痕迹（罗马数字后缀 / 兵·圣盾·模组…）：
    # 「防卫兵I」「Lv.60」以前直接把敌人名当名字返回。
    raw = _nfkc(texts[i])[:m.start()] if here else next(t for t in reversed(texts[:i]) if _clean_name(t) == candidates[0])
    issue = _text_issue(raw)
    if issue:
        return _result(support, None, level, candidates, False, issue)
    return _result(support, candidates[0], level, candidates)


# ---------------------------------------------------------------- 解析：带坐标（推荐）

def _as_item(b):
    """ok 的 Box（name/x/y/width/height/confidence）或 (text, x, y, w, h[, conf])，整屏像素。"""
    if isinstance(b, (tuple, list)):
        t, x, y, w, h = b[:5]
        conf = b[5] if len(b) > 5 else None
    else:
        t, x, y, w, h = b.name, b.x, b.y, b.width, b.height
        conf = getattr(b, 'confidence', None)
    return str(t or ''), float(x), float(y), float(w), float(h), (None if conf is None else float(conf))


def _text_issue(text):
    """[fixed] 名字框文字（Lv 之前、切掉「精锐…」之后）里有敌人名字的痕迹 -> 原因，否则 None。"""
    pre = LV_RE.split(_squash(text), 1)[0]
    for mk in ENEMY_MARKERS:
        if mk in pre:
            pre = pre[:pre.find(mk)]
    if _ENEMY_SUFFIX_RE.search(pre):
        return 'enemy_suffix'
    if any(wd in pre for wd in ENEMY_WORDS):
        return 'enemy_text'
    return None


def _glyph_issue(text, w, h, conf):
    """名字框的字形和详情卡字体对不上 -> 返回原因（敌人名字条的小字混进来时常见）。"""
    if conf is not None and conf < NAME_MIN_CONF:
        return 'low_conf'
    if not NAME_H_RANGE[0] <= h <= NAME_H_RANGE[1]:
        return 'glyph_height'
    issue = _text_issue(text)
    if issue:
        return issue
    t = _squash(text)
    merged = bool(LV_RE.search(t))
    if merged and _CJK_ONLY_RE.match(LV_RE.split(t, 1)[0] or 'x') and conf is not None and conf < MERGED_MIN_CONF:
        return 'low_conf'
    if not merged and _CJK_ONLY_RE.match(t):
        if conf is not None and conf < CJK_NAME_MIN_CONF:
            return 'low_conf'
        n = len(t.replace('·', ''))
        per = w / (n + 0.45 * t.count('·')) if n else 0
        if n >= 3 and not CJK_CHAR_W_RANGE[0] <= per <= CJK_CHAR_W_RANGE[1]:
            return 'glyph_width'
        if n == 2 and not CJK2_CHAR_W_RANGE[0] <= per <= CJK2_CHAR_W_RANGE[1]:
            return 'glyph_width'
    return None


def parse_unit_card_boxes(boxes, width, height, frame=None):
    """按名字条的位置解析详情卡。boxes 是整屏像素坐标的 OCR 框（区域 OCR 时 ok 已经加回偏移）。

    名字 = 左沿落在名字条起点（NAME_X0_RANGE）、中心在名字行（NAME_ROW_YC_RANGE）的那个框，
    连同它右边同一行的框按 x 拼起来，取第一个 Lv.xx 之前的部分，再去掉「精锐…」敌人名。
    下列情况 name=None（reason 写原因），调用方不得出击：
      no_name_box / two_name_boxes / no_lv / name_unreadable
      low_conf / glyph_height / glyph_width   名字框字形不像详情卡字体（混进了别的字）
      enemy_suffix / enemy_text               [fixed] 名字框里有敌人名字的罗马数字后缀（防卫兵I）或敌人
                                              常用字词（兵/圣盾/模组/督导/试作）—— 字混在一个框里，真名可能已读坏
      enemy_bar                               给了 frame，且卡片附近有敌人血条（名字可能被盖）
    敌人名字条落在名字和 Lv 之间（没盖住名字）时会被拼进名字（「威玛西娜狙击兵」），
    ambiguous=True、name 照给；same_character 的包含规则让它和「威玛西娜」仍算同一个角色。
    """
    W, H = float(width), float(height)
    items = []
    for b in boxes or []:
        t, x, y, w, h, conf = _as_item(b)
        if not t.strip():
            continue
        xc, yc = (x + w / 2) / W, (y + h / 2) / H
        if DETAIL_BOX[0] <= xc <= DETAIL_BOX[2] and DETAIL_BOX[1] <= yc <= DETAIL_BOX[3]:
            items.append((t, x / W, y / H, w / W, h / H, yc, conf))
    support = _has_support_tag([it[0] for it in items])
    starts = [it for it in items
              if NAME_X0_RANGE[0] <= it[1] <= NAME_X0_RANGE[1]
              and NAME_ROW_YC_RANGE[0] <= it[5] <= NAME_ROW_YC_RANGE[1]
              and _clean_name(LV_RE.split(_nfkc(it[0]), 1)[0])]
    if not starts:
        return _result(support, reason='no_name_box')
    if len(starts) > 1:
        return _result(support, candidates=_uniq(_clean_name(it[0]) for it in starts),
                       ambiguous=True, reason='two_name_boxes')
    nb = starts[0]
    row = sorted((it for it in items if it is nb or (abs(it[5] - nb[5]) <= ROW_DY and it[1] > nb[1])),
                 key=lambda it: it[1])
    joined = ''.join(_squash(it[0]) for it in row)
    m = LV_RE.search(joined)
    if not m:
        return _result(support, candidates=_uniq([_clean_name(joined)]), reason='no_lv')
    level = _parse_level(m)
    name = _clean_name(joined[:m.start()])
    # 名字框和 Lv 之间还夹着别的像名字的框（敌人名字条，或名字被拆成两块）-> ambiguous；
    # name 仍取整段（含真名），每一块也放进 candidates。
    parts, pos = [], 0
    for it in row:
        piece = _squash(it[0])
        if pos < m.start():
            parts.append(_clean_name(piece[:m.start() - pos]))
        pos += len(piece)
    parts = _uniq(parts)
    candidates = _uniq([name] + parts)
    reason = None if name else 'name_unreadable'
    reason = reason or _glyph_issue(nb[0], nb[3], nb[4], nb[6])
    if reason is None and frame is not None and find_enemy_bar(frame) is not None:
        reason = 'enemy_bar'
    return _result(support, None if reason else name, level, candidates, len(parts) > 1, reason)


# ---------------------------------------------------------------- 敌人血条（可选的像素检查）

# 在名字条附近找敌人头顶的血条：红（普通，HSV≈(0,175,232)）/ 黄（精锐，≈(21,199,219)），
# 实测厚 0.007~0.008H、满血长 0.079~0.089W。要求「细长横条 + 上下都不是暖色」，
# 地图里的红气球、黄黑警示条这类大块/斜纹不会命中。
BAR_ZONE = (0.06, 0.82, 0.26, 0.90)   # 敌人字盖到名字字形（y 0.829~0.858）时血条落在 y≈0.836~0.883
BAR_MIN_LEN = 0.012          # W：残血时彩色部分变短，取约 15% 血量
BAR_THICK = (0.003, 0.012)   # H
BAR_ALIGN = 0.004            # W：同一条血条逐行起止位置的抖动
BAR_CLEAR = 0.004            # H：血条上下这么远的行在血条横向范围内基本不是暖色


def find_enemy_bar(frame):
    """frame（BGR 整屏）里卡片名字条附近有没有敌人血条。有 -> dict(x0,x1,y0,y1 相对整屏)，没有 -> None。"""
    import cv2
    import numpy as np
    H, W = frame.shape[:2]
    clear = max(2, int(round(BAR_CLEAR * H)))
    x0, x1 = int(BAR_ZONE[0] * W), int(BAR_ZONE[2] * W)
    y0, y1 = int(BAR_ZONE[1] * H), int(BAR_ZONE[3] * H)
    ya, yb = max(0, y0 - clear - 2), min(H, y1 + clear + 2)
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[ya:yb, x0:x1]), cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = (ss >= 110) & (vv >= 130) & ((hh <= 30) | (hh >= 165))
    min_len = max(4, int(round(BAR_MIN_LEN * W)))
    tol = max(2, int(round(BAR_ALIGN * W)))
    tmin, tmax = max(2, int(BAR_THICK[0] * H)), max(3, int(round(BAR_THICK[1] * H)))
    runs = []
    for r in range(warm.shape[0]):
        d = np.diff(np.concatenate(([0], warm[r].astype(np.int8), [0])))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(st):
            k = int(np.argmax(en - st))
            if en[k] - st[k] >= min_len:
                runs.append((int(st[k]), int(en[k])))
                continue
        runs.append(None)
    # [fixed] 以前要求逐行的起点和终点都在 tol 以内，真实血条的一端碰到旁边敌人的等级角标/图标时
    # （敌人扎堆很常见），终点逐行漂 >0.004W，一条血条被切成两段，每段的「上下不是暖色」又被另一段
    # 否掉 -> 满血也漏检（174058「冲锋兵·燃烧」血条贴到卡上 -> None）。现在按「和当前公共段重叠」
    # 往下接行：公共段 = 各行起点取最大、终点取最小，新行和公共段重叠 ≥60% 就算同一条；
    # 起点仍要对齐（tol），上下清空检查在公共段上做。
    r, n = y0 - ya, y1 - ya
    while r < n:
        if runs[r] is None:
            r += 1
            continue
        cs, ce = runs[r]
        s_first = cs
        r2 = r
        while r2 + 1 < len(runs) and runs[r2 + 1] is not None:
            ns, ne = runs[r2 + 1]
            ov = min(ce, ne) - max(cs, ns)
            if ov < 0.6 * (ce - cs) or ov < min_len or (abs(ns - s_first) > tol and abs(ne - ce) > tol):
                break
            cs, ce = max(cs, ns), min(ce, ne)
            r2 += 1
        if tmin <= r2 - r + 1 <= tmax:
            above = warm[max(0, r - 1 - clear):max(0, r - 1), cs:ce]
            below = warm[r2 + 2:r2 + 2 + clear, cs:ce]
            if above.size and below.size and above.mean(axis=1).max() < 0.35 and below.mean(axis=1).max() < 0.35:
                return {'x0': (x0 + cs) / W, 'x1': (x0 + ce) / W, 'y0': (ya + r) / H, 'y1': (ya + r2 + 1) / H}
        r = r2 + 1
    return None


# ---------------------------------------------------------------- 同一个角色？

def _norm(name):
    """比较用的规范化：NFKC、繁->简、小写、去空白和两端符号、汉字之间的 ./-/_ 当成「·」、
    拉丁混淆字折叠（0->o, 1/i/|->l）。"""
    s = _squash(name).translate(_T2S).lower()
    s = re.sub(r'(?<=[%s])[.\-_](?=[%s])' % (_CJK, _CJK), '·', s)
    s = s.strip(_EDGE_JUNK)
    return s.translate(_LATIN_FOLD)


def base_name(name):
    """「纳美西丝·衍光」->「纳美西丝」；没有「·」原样返回（已规范化）。"""
    s = _norm(name)
    head = s.split('·', 1)[0]
    return head if head else s.replace('·', '')


def _edit_distance_le1(a, b):
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def same_character(a, b):
    """a、b 两个名字是不是同一个角色。

    - None/空/规范化后不足 2 字 -> False（「读不出来」由调用方当成不能出击，不在这里兜）
    - 比基础名（「·」之前）和去掉「·」的整名，任一对相同即 True：「纳美西丝」==「纳美西丝·衍光」，
      「威·玛西娜」==「威玛西娜」，「冲锋兵·燃烧威玛西娜」==「威玛西娜」
    - 大小写、全半角、繁简（可露凱/可露凯）不敏感
    - 一方包含另一方（≥2 字）-> 相同：吞了「·」的「纳美西丝衍光」、拼进敌人名的「威玛西娜狙击兵」
    - 短的一方 ≥3 字时容忍 1 个字的 OCR 误差（错字/漏字/多字，或在长串里滑窗错 1 字，
      同 name_matches）：威玛西纳 == 威玛西娜
    - [fixed] 2 字名容忍 1 处错字/漏字/多字（不滑窗）：「奇搭」==「奇塔」，「可凯」==「可露凯」；
      代价是「奇塔」「洛塔」也算同一个（安全方向，只损失可用性）
    """
    if not a or not b:
        return False
    # [fixed] 以前只比「·」前的基础名：OCR 多读出一个点（「威·玛西娜」「奇-塔」，汉字间的 ./-/_ 也会
    # 变成「·」）基础名就只剩 1 个字 -> False；敌人名带「·」并进名字框（「冲锋兵·燃烧威玛西娜」）
    # 基础名变成「冲锋兵」-> False。这都是危险方向（同一个角色判成不同）。现在基础名和去掉「·」的整名
    # 两两比，任一对相同就算同一个角色 —— 只会多给 True。（不单独比「·」后的段：「X·衍光」「Y·衍光」
    # 这种同后缀不同角色会被误并。）
    na, nb = _norm(a), _norm(b)
    fa, fb = _forms(na), _forms(nb)
    return any(_same_form(x, y) for x in fa for y in fb)


def _forms(s):
    out = []
    # 基础名、去「·」整名、再去掉 ./-/_ 的整名（「OTs-14」读成「OT14」时 - 和 s 都丢了）
    for f in (s.split('·', 1)[0], s.replace('·', ''), re.sub(r'[·.\-_]', '', s)):
        f = f.strip(_EDGE_JUNK)
        if len(f) >= 2 and f not in out:
            out.append(f)
    return out


def _same_form(na, nb):
    if na == nb:
        return True
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if short in long_:
        return True
    if len(short) < 3:
        # [fixed] 2 字名以前必须精确匹配，于是 OCR 错 1 个字（实测合成里「奇塔」读成「奇搭」、
        # 「可露凯」中间字被挡读成「可凯」）就判成不同角色 = 危险方向。现在 2 字名也容忍 1 处
        # 错字/漏字/多字（只和等长或长 1 的比，不做长串滑窗）。代价：「奇塔」和「洛塔」这类
        # 真的不同、但只差 1 个字的 2 字名会被当成同一个（只损失可用性）。
        return _edit_distance_le1(short, long_)
    if _edit_distance_le1(short, long_):
        return True
    # [fixed] 长串里找「和短名差 ≤1 处编辑」的片段：以前只滑等长窗口（只容错字），真名漏一个字又拼上
    # 敌人名时（「佩里亚防卫兵」vs「佩里缇亚」）对不上。现在窗口长 n-1 / n / n+1 都试。
    n = len(short)
    for k in (n - 1, n, n + 1):
        for i in range(len(long_) - k + 1):
            if _edit_distance_le1(long_[i:i + k], short):
                return True
    return False


def card_matches(card, name):
    """解析出的详情卡「可能是」name 这个角色吗（用来决定「这个自有人形要不要跳过」）。

    [fixed] 读不出来的一方当成「可能相同」：card 为空 / card['name'] 为 None，或 name 为空 -> True。
    以前这两种情况返回 False（= 不同角色 = 可以出击），助战名没读出来时每个自有人形都会被放行。
    其余：card['name'] 或任一候选名 same_character(name) 即 True。"""
    if not name or not card or not card.get('name'):
        return True
    cands = _uniq([card.get('name')] + list(card.get('candidates') or []))
    return any(same_character(c, name) for c in cands)


def cards_same_character(card_a, card_b):
    """两张详情卡「可能是」同一个角色吗。[fixed] 任一张为空或 name 为 None -> True（读不出来当成相同）；
    否则任一对候选名相同即 True。"""
    if not card_a or not card_b or not card_a.get('name') or not card_b.get('name'):
        return True
    cb = _uniq([card_b.get('name')] + list(card_b.get('candidates') or []))
    return any(card_matches(card_a, c) for c in cb)
