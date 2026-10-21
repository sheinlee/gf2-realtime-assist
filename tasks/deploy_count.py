# -*- coding: utf-8 -*-
"""顶部「部署人形 x/y」计数的读取（纯函数，只依赖 re，不 import ok）。

用法（在 support_pick / own_units 里）：
    boxes = self.ocr(box=self.deploy_count_box())        # 0.30~0.62W x 0.015~0.10H
    count = read_deploy_count_boxes(boxes, self.width, self.height)

boxes 只要有 .name/.x/.y/.width/.height（ok-script 的 Box，整帧坐标）。

和 support_pick.read_deploy_count（DEPLOY_COUNT_RE 那一版）比，规则是**它的超集再加两道闸**：

  旧规则照留：
    - 只认「部署人形」标签；标签和计数并成一个框（「部署人形3/4」）时从标签后面截出来解析；
    - 否则只看标签**同一行**（中心高差 < 0.6×标签高）、x >= 标签.x + 0.5×标签宽 的框，按 x 从左往右；
    - 斜杠必须有，而且只能是长得像斜杠的字符（/ ／ \\ | 丨 l I i 1）；x、y 都只有一位；
    - 第一个「长得像计数」的框说了算：合法就返回，不合法（x>y 之类）就返回 None，不再往后找。

  放宽（这次要修的）：
    - x/y 后面允许跟着并进来的「作战效能」：「3/4作战效能」「3/4 作战效能」「3/4曜石作战效能」「3/4效能」
      （只认这个，别的尾巴一律不认，见下面「对抗验证后的两处收紧」）；
    - x/y 后面紧跟**一个**竖线样的字符（1 l I i | 丨）当成计数和「作战效能」之间那根分隔竖线：
      「3/41」「3/4 1」「1/41」→ 3/4、1/4。依据：y 只有一位（日志里见过的上限只有 1/3/4/5），
      「41」不可能是上限。旁证：2560 下「3/41」的框宽 108px，同分辨率正常「3/4」57~83px（宽度只当旁证，
      别的分辨率两者有重叠：d/d 0.038~0.071H，d/dd 0.070~0.077H，所以代码里不用框宽判断）。
    - 标签认错一个字也算：「部署大形」「部署火形」（部署.?形）。

  加的闸（放宽之后才需要）：
    - **位置闸**：计数框的左边必须落在标签右边沿往右 -0.03H ~ +0.09H 之内（实测 0.012~0.061H）。
      作战效能数值框在 +0.20H 以后，所以「3141心」「414」这种战力数字在计数没认出来时也不会被当成 3/4、4/4
      （放宽之后「3141心」会被解析成 3/4，这道闸是必须的）。
    - x/y 后面（去掉那根竖线后）紧跟数字，或者「标点+数字」（「1.5」「3/4 2615」「3/412」）一律不认，并且就此停手。
    - 从标签框里截出来的尾巴（「部署人形3/4…」）没法用位置闸，所以尾巴里 x/y 后面只允许空着/竖线/紧跟「作战效能」，
      其余（比如标签和敌人名字并框成「部署人形1l5·试作版」）当「不像计数」，接着去同一行找。
    - 上限 y 必须在 1~5 之间（原版 1~6）。**0/0 不认**（和旧版不同）：10/10 3072 的多张图，进部署界面刚加载那一两秒
      整帧 OCR 读到的是「部署人形 0/0 作战效能 0」，约 5 秒后才变成 3/4（192620：19:25:11 整帧读 0/0，日志 19:25:16 读 3/4）。
      旧版会把 0/0 当成「阵容已满 0/0，不借助战」。

  对抗验证后的两处收紧：
    - **x/y 后面跟的字必须是「作战效能」**：去掉那根竖线之后，要么什么都不剩，要么「作战 / 战效 / 效能」
      出现在剩下的字的前 2 个字符之内（容得下实测的「暉石作战效能」「」作战效能」「|作找效能」）。
      行内框和标签尾巴都用这条（行内框不过 -> 'bad' 停手；尾巴不过 -> 当「不像计数」去行里找）。
      原版对行内框只要求「后面不是数字开头」，于是这些都会读成计数（旧版全是 None）：
        「111.」（1/4 下半截被挡）-> 1/1（= 阵容已满）；「4/4+」-> 4/4；「2|3号机」-> 2/3；
        敌人名牌压在计数上和「作战效能」并框「115·试作版作战效能」-> 1/5（只查「含作战效能」也拦不住这条）。
      代价：真实裁切 OCR 里 0 条读数受影响；只有整帧 json 里一条过期帧的「3/4Z7」从 3/4 变成 None。
    - **上限 MAX_CAP 从 6 降到 5**：对话框背景那种「压暗 + 高斯模糊」下，OCR 会把 4 认成 6（实测 5 例：
      3/4->3/6、4/4->4/6、1/4->1/6，置信度 0.77~0.93，和正确读数的置信度分布重叠，按置信度拦不住）。
      日志 1,322 条读数和全部截图里上限只出现过 1/3/4/5。若真有上限 6 的关卡，这里会返回 None（安全方向）。

  双计数变体（「人形 3/4 | 效能 …」下面一行「支援 0/2」，见 20261002/150258、151303、151317、153443）：
    - 默认**不读**（allow_short_label=False，和旧版一样返回 None）。
    - 打开后只认：文字**恰好**是「人形」的标签、标签中心在 0.07H 以上（实测 0.041~0.045H；「支援」那一行 0.100~0.101H）、
      计数框在标签右边沿 +0.06H ~ +0.14H（实测 0.097~0.103H）、同一行计数右边还要有「效能」。
      「支援 0/2」那一行：标签不是「人形」；和「人形」行中心差 ~0.056H，同行阈值只有 0.6×标签高 ≈ 0.014~0.023H；
      「0/2」在 ~0.596W，离「人形」右边沿 ~0.33H，窗口只到 0.14H —— 三道都过不去。
      语义上的旁证：变体里「人形 3/4」旁边的效能 2615 / 8152 和同一天普通布局「部署人形 3/4」的效能完全一样（同一队预设），
      右下角也是「部署人形」+「部署支援」两个按钮，借来的助战走的是「部署人形」。但**没有一张图拍到变体里放上助战之后
      「人形」计数从 3/4 涨到 4/4**，所以默认关着。
"""
import re

DEPLOY_LABEL = '部署人形'
DEPLOY_LABEL_FUZZY_RE = re.compile(r'部署.?形')       # 「部署大形」「部署火形」「部署形」
SHORT_LABEL = '人形'                                   # 双计数变体
SHORT_ROW_TAG = '效能'

SEP_CHARS = '/／\\|丨lIi1'                              # 和旧版 DEPLOY_COUNT_RE 中间那一位一样
BAR_CHARS = '1lIi|丨'                                   # 计数和「作战效能」之间的分隔竖线常被认成这些
COUNT_HEAD_RE = re.compile(r'^(\d)\s*[' + re.escape(SEP_CHARS) + r']\s*(\d)')
# 去掉竖线之后，如果紧跟（可夹标点/空白）数字，就是「1.5」「3/412」「3/4 2615」之类，不认
TRAILING_NUMBER_RE = re.compile(r'^[^\w]*\d|^_')

# x/y（去掉竖线）后面如果还有字，必须是并进来的「作战效能」：这几个片段之一出现在前 TAG_MAX_PREFIX 个字符里
TAG_FRAGMENTS = ('作战', '战效', '效能')
TAG_MAX_PREFIX = 2

MIN_CAP = 1
MAX_CAP = 5                        # 原版 6；6 只会来自模糊时 4->6 的误认（见模块说明）

ROW_DY_RATIO = 0.6                 # 旧版：同一行 = 中心高差 < 0.6 × 标签高
RIGHT_OF_LABEL_RATIO = 0.5         # 旧版：x >= 标签.x + 0.5 × 标签宽
COUNT_GAP_MIN_H = -0.03            # 计数框左边 - 标签右边沿，单位 H（实测 0.012~0.061）
COUNT_GAP_MAX_H = 0.09             # 战力数值框最早 0.2046H；「作战效能」字样最早 0.089H（本身不会被解析成计数）
SHORT_GAP_MIN_H = 0.06             # 变体：实测 0.097~0.103
SHORT_GAP_MAX_H = 0.14
SHORT_LABEL_MAX_CY_H = 0.07        # 变体标签中心实测 0.041~0.045H；「支援」那一行 0.100~0.101H
LABEL_W_H = 0.092                  # 「部署人形」四个字单独成框时的宽度，实测 0.090~0.095H


def _rest_is_power_label(rest):
    """x/y（和竖线）之后剩下的字是不是并进来的「作战效能」：空着，或者片段出现在前 TAG_MAX_PREFIX 个字符里。"""
    r = re.sub(r'\s+', '', rest or '')
    if not r:
        return True
    pos = [p for p in (r.find(f) for f in TAG_FRAGMENTS) if p >= 0]
    return bool(pos) and min(pos) <= TAG_MAX_PREFIX


def _parse(text, max_cap=MAX_CAP, min_cap=MIN_CAP, tail=False):
    """-> ('none', None) 不像计数 / ('bad', 原因) 像计数但不能信 / ('ok', (x, y))。

    x/y 后面（去掉竖线后）只允许空着、或者紧跟着并进来的「作战效能」（_rest_is_power_label）。
    tail=False（行内的框）：不满足就 'bad'，停手 —— 「111.」「4/4+」「2|3号机」「115·试作版作战效能」都不认。
    tail=True：这段字是从「部署人形」标签框里截出来的尾巴，不满足就当「不像计数」，
    接着去同一行找（防「部署人形」和敌人名字并框，名字恰好以「数字+竖线样字符+数字」开头）。"""
    t = (text or '').strip()
    m = COUNT_HEAD_RE.match(t)
    if not m:
        return 'none', None
    cur, cap = int(m.group(1)), int(m.group(2))
    rest = t[m.end():].lstrip()
    if rest[:1] and rest[0] in BAR_CHARS:
        rest = rest[1:].lstrip()
    if rest and TRAILING_NUMBER_RE.match(rest):
        return 'bad', 'digits after x/y: %r' % t
    if not _rest_is_power_label(rest):
        if tail:
            return 'none', None
        return 'bad', 'text after x/y is not 作战效能: %r' % t
    if not (min_cap <= cap <= max_cap):
        return 'bad', 'cap %d out of %d..%d: %r' % (cap, min_cap, max_cap, t)
    if cur > cap:
        return 'bad', 'cur %d > cap %d: %r' % (cur, cap, t)
    return 'ok', (cur, cap)


def parse_count_text(text, max_cap=MAX_CAP, min_cap=MIN_CAP):
    """一段 OCR 文字 -> (已部署, 上限) 或 None。文字必须**以** x/y 开头。

    >>> parse_count_text('3/4'), parse_count_text('3/4作战效能'), parse_count_text('3/41')
    ((3, 4), (3, 4), (3, 4))
    >>> parse_count_text('314作战效能'), parse_count_text('3/4 1'), parse_count_text('3/4曜石作战效能')
    ((3, 4), (3, 4), (3, 4))
    >>> [parse_count_text(s) for s in ('15', '1.5', '莫普1.5·试作版', '1.5·试作版', '3/412', '3/4 2615', '0/0', '5/4', '3/7')]
    [None, None, None, None, None, None, None, None, None]
    >>> [parse_count_text(s) for s in ('111.', '4/4+', '2|3号机', '115·试作版作战效能', '3/6', '4/6', '3/4Z7')]
    [None, None, None, None, None, None, None]
    >>> [parse_count_text(s) for s in ('3/4暉石作战效能', '1/4」作战效能', '3/4|作找效能', '3/4作战效能14154', '3/4效能')]
    [(3, 4), (1, 4), (3, 4), (3, 4), (3, 4)]
    """
    status, val = _parse(text, max_cap, min_cap)
    return val if status == 'ok' else None


def _name(b):
    return (getattr(b, 'name', None) or '').strip()


def _cy(b):
    return b.y + b.height / 2.0


def _label_end_x(b, start_index, end_index, H):
    """标签框里「部署人形」这几个字的右边沿。
    - 字后面没别的字（含「徒部署人形」「重机枪模组鳄面部署人形」这种前面并了敌人名字的）：框的右边；
    - 字在最前面、后面跟着别的字：框左边 + 标签本身的宽度（实测 0.090~0.095H），
      不按字数比例估 —— 尾巴是数字/小字号的敌人名字时比例会偏得很远；
    - 前后都有别的字（没见过）：按字数比例估。"""
    n = len(_name(b))
    if n <= 0 or end_index >= n:
        return b.x + b.width
    if start_index == 0:
        return b.x + min(b.width, LABEL_W_H * H)
    return b.x + b.width * end_index / float(n)


def _find_label(boxes, H):
    """-> (box, 标签之后的文字, 标签右边沿 x, 'exact'/'fuzzy') 或 None。精确的优先，各自取列表里第一个。"""
    for b in boxes:
        t = _name(b)
        i = t.find(DEPLOY_LABEL)
        if i >= 0:
            j = i + len(DEPLOY_LABEL)
            return b, t[j:].strip(), _label_end_x(b, i, j, H), 'exact'
    for b in boxes:
        t = _name(b)
        m = DEPLOY_LABEL_FUZZY_RE.search(t)
        if m:
            return b, t[m.end():].strip(), _label_end_x(b, m.start(), m.end(), H), 'fuzzy'
    return None


def _row_candidates(boxes, label, label_end, H, gap_min_h, gap_max_h):
    lcy, lh = _cy(label), max(label.height, 1)
    row = []
    for b in boxes:
        if b is label:
            continue
        if b.x < label.x + label.width * RIGHT_OF_LABEL_RATIO:
            continue
        if abs(_cy(b) - lcy) >= ROW_DY_RATIO * lh:
            continue
        gap = (b.x - label_end) / float(H)
        if not (gap_min_h <= gap <= gap_max_h):
            continue
        row.append(b)
    row.sort(key=lambda b: b.x)
    return row


def explain_deploy_count_boxes(boxes, W, H, allow_short_label=False):
    """和 read_deploy_count_boxes 一样，但返回 dict：value / path / text / reason，方便写日志。"""
    boxes = [b for b in (boxes or []) if _name(b)]
    found = _find_label(boxes, H)
    if found:
        label, tail, label_end, kind = found
        cands = []
        if tail:
            cands.append(('tail', tail))
        for b in _row_candidates(boxes, label, label_end, H, COUNT_GAP_MIN_H, COUNT_GAP_MAX_H):
            cands.append(('row', _name(b)))
        for where, text in cands:
            status, val = _parse(text, tail=(where == 'tail'))
            if status == 'none':
                continue
            if status == 'bad':
                return {'value': None, 'path': 'label_' + kind, 'text': text, 'reason': val}
            return {'value': val, 'path': 'label_%s/%s' % (kind, where), 'text': text,
                    'reason': 'label %r' % _name(label)}
        return {'value': None, 'path': 'label_' + kind, 'text': None,
                'reason': 'no count-shaped box next to %r (row: %s)' % (
                    _name(label), [t for _, t in cands])}

    if allow_short_label:
        return _explain_short_label(boxes, W, H)
    return {'value': None, 'path': None, 'text': None, 'reason': 'no 部署人形 label'}


def _explain_short_label(boxes, W, H):
    labels = [b for b in boxes if _name(b) == SHORT_LABEL and _cy(b) < SHORT_LABEL_MAX_CY_H * H]
    if not labels:
        return {'value': None, 'path': None, 'text': None, 'reason': 'no label'}
    label = labels[0]
    label_end = label.x + label.width
    for b in _row_candidates(boxes, label, label_end, H, SHORT_GAP_MIN_H, SHORT_GAP_MAX_H):
        text = _name(b)
        status, val = _parse(text)
        if status == 'none':
            continue
        if status == 'bad':
            return {'value': None, 'path': 'short', 'text': text, 'reason': val}
        # 同一行、计数右边要有「效能」（并在计数框里也算）——确认这一行是部署条，不是别的「人形」字样
        lcy, lh = _cy(label), max(label.height, 1)
        tagged = SHORT_ROW_TAG in text or any(
            SHORT_ROW_TAG in _name(o) and o.x > b.x and abs(_cy(o) - lcy) < ROW_DY_RATIO * lh
            for o in boxes if o is not b)
        if not tagged:
            return {'value': None, 'path': 'short', 'text': text, 'reason': 'no 效能 on the row'}
        return {'value': val, 'path': 'short/row', 'text': text, 'reason': 'label 人形'}
    return {'value': None, 'path': 'short', 'text': None, 'reason': 'no count next to 人形'}


def read_deploy_count_boxes(boxes, W, H, allow_short_label=False):
    """顶部那一条的 OCR 框 -> (已部署, 上限) 或 None。W/H 是整帧宽高，框是整帧坐标。"""
    return explain_deploy_count_boxes(boxes, W, H, allow_short_label)['value']

