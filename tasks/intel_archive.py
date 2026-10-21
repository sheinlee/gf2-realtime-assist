"""机要密报的 OCR 判定；坐标始终来自游戏客户区。"""
import re


def progress_crop(frame, target_height=160):
    """隔离右上角数字，放大至固定高度，避免整屏缩放使斜杠变成 1。"""
    import cv2
    height, width = frame.shape[:2]
    crop = frame[int(.015*height):int(.085*height), int(.95*width):int(.99*width)]
    scale = target_height / crop.shape[0]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def text(box):
    return re.sub(r'\s+', '', box.name)


def center(box):
    return box.x + box.width / 2, box.y + box.height / 2


def counter(value):
    m = re.search(r'(\d+)[/／](\d+)', value)
    return tuple(map(int, m.groups())) if m else None


def counter_loose(value):
    """斜杠被读成「1」的进度（「016」-> 0/6、「919」-> 9/9）：把其中一个 1 当成斜杠，只有一种拆法说得通才返回。
    拆不出 / 有两种说法都返回 None。用的地方自己决定信不信（列表上只决定点不点开，详情页要和已知的总数对得上）。"""
    v = re.sub(r'\s+', '', value or '')
    if not re.fullmatch(r'\d{3,5}', v):
        return None
    found = set()
    for i, ch in enumerate(v):
        if ch != '1' or i == 0 or i == len(v) - 1:
            continue
        a, b = v[:i], v[i + 1:]
        if (len(a) > 1 and a[0] == '0') or (len(b) > 1 and b[0] == '0'):
            continue
        a, b = int(a), int(b)
        if 0 <= a <= b <= 99 and b > 0:
            found.add((a, b))
    return found.pop() if len(found) == 1 else None


def detail_progress(boxes, width, height):
    for b in boxes:
        x, y = center(b)
        if x > .85 * width and y < .12 * height:
            p = counter(text(b))
            if p and 0 <= p[0] <= p[1] <= 99:
                return p


def pending_cards_full(boxes, width, height):
    """进度锚点 -> 下方进度与卡片标题 -> [(标题, 位置, 进度)]。裁掉的卡片留给下一次滚动。"""
    result = []
    for anchor in boxes:
        if '解锁进度' not in text(anchor):
            continue
        ax, ay = center(anchor)
        if not (.12 * width < ax < .65 * width and .1 * height < ay < .78 * height):
            continue
        nearby = [b for b in boxes if abs(center(b)[0] - ax) < .07 * width
                  and -.015 * height < center(b)[1] - ay < .09 * height]
        if any('归档' in text(b) for b in nearby):
            continue
        progress = next((counter(text(b)) for b in nearby if counter(text(b))), None)
        if not progress:
            # 「0/6」读成「016」（10/14 14:15「未至的祷告」）：列表上只决定点不点开，点进去还会核对详情页的计数
            progress = next((counter_loose(text(b)) for b in nearby if counter_loose(text(b))), None)
        if not progress or not 0 <= progress[0] < progress[1] <= 99:
            continue
        # 标题至少两个字（「裂痕」只有两个字，以前要求三个字，整张卡一直被跳过 —— 10/14 你自己翻回去点开的），
        # 编号（FZ-015）、纯数字不算
        titles = [b for b in boxes if ax - .06 * width < b.x < ax + .20 * width
                  and .12 * height < center(b)[1] - ay < .22 * height
                  and len(text(b)) >= 2 and re.search(r'[\u4e00-\u9fffA-Za-z]', text(b))
                  and not re.fullmatch(r'[A-Z0-9-]+', text(b))]
        if titles:
            title = min(titles, key=lambda b: b.y)
            result.append((text(title), center(title), progress))
    return sorted(result, key=lambda item: (round(item[1][1] / (height * .08)), item[1][0]))


def pending_cards(boxes, width, height):
    """[(标题, 位置)]：还没完成的卡片，从上到下、从左到右。"""
    return [(n, pt) for n, pt, _ in pending_cards_full(boxes, width, height)]


def card_progress(boxes, width, height, name):
    """列表上这张卡读到的进度（pending_cards 那一套），读不到 None。"""
    return next((p for n, _, p in pending_cards_full(boxes, width, height) if n == name), None)
