# -*- coding: utf-8 -*-
"""缺人时从「我的人形」补人（ok-gf2 扩展）。不定义 BaseTask 子类。

你 10/10 说：「缺人时让它从「我的人形」里自动补人，注意不要上助战借用相同的角色」「放入空蓝格就行了」。
10/01～10/10 七次：放完助战点「作战开始」，游戏弹「本关至少需要出击4人！」，它照样再点 3 次、停手，
5 分钟后重试还去借一次已经在场上的助战。现在（stage_runner._sr_fill 调 fill_own_units）：

  1. 选一个空蓝格（沿用借助战那一套 pick_empty_tile）；点完地格右边面板会自己跳回「我的人形」
  2. 认准是「我的人形」页（页签下划线 +「使用整备室配置」下拉框）。下拉框成了「编队N」、列表写「暂无可选择人形」：
     不补、停手（10/02 08:17 误点那个下拉框，阵容从 3/4 变成 0/4 —— **那一块一下都不点**，own_panel.in_danger_zone）
  3. 元素筛选切回「全部」：好友助战页选的筛选会带过来（10/10 三张图只剩和助战同元素的 3 张卡，正好最容易撞同一个角色）
  4. 从没打勾（没上场）的卡里挑作战效能最高的，点开读左下角详情卡的名字：认准了是和助战同一个角色（含「·衍光」这种
     不同版本）就换下一张；详情卡认不准（没刷新 / 读不出 / 还是助战）就不往下点了 —— 换下一张时读到的可能是上一张迟到的名字
  5. 按右下角「部署人形」；部署数涨了算补上。部署数读不到就看这张卡有没有打上勾
  6. 不够就再来一轮；补不上就把原因交回去（stage_runner 抛 LineupShortError，轮询停手提醒你）

**只点了卡、还没按「部署人形」，场上就该不变**（借助战一直是这样：选卡、再按按钮）。点完卡部署数涨了 / 右下角成了「撤回人形」/
读不准，都是没见过的情况：场上可能多了个没核对的人 —— 马上停手（risky），轮询不再碰这个部署界面，请你看一眼、自己开打。
（10/11 两轮对抗复查：先前「撤下来再接着补」「停手后一看够了就替你开打」那套，在点卡就上场 + 撤不下来 / 详情卡慢 /
读数一闪的组合下都会带着重复角色开打，所以改成这样。）
认卡的几块（own_panel / unit_name / deploy_count）都在真截图上量过，判不准一律往「不点 / 不上」那边报。
"""
import collections

from own_panel import in_danger_zone, read_own_panel
from support_pick import DEPLOY_BTN, TILE_EMPTY, TILE_OCCUPIED, TILE_SEARCH_GIVE_UP, UserTakeoverError
from unit_name import card_matches, parse_unit_card_boxes

OWN_FILL_SECONDS = 150.0        # 一次补人最多花这么久（找地格最多 24 下 + 每张卡读名字约 3 秒）
OWN_CARD_TRIES = 6              # 补一个人最多点开几张卡（认准了是同一个角色就换下一张）
OWN_CLICK_SLEEP = 0.9           # 点卡 / 页签 / 筛选之后等这么久再读（再加界面等待）
CARD_SETTLE_READS = 4           # 点卡之前：详情卡连着两次读到同一张才算稳了（最多读这么多次，隔 0.6 秒）
CARD_REFRESH_READS = 4          # 点卡之后：还是点之前那张 / 空的就再等 0.6 秒重读，最多这么多次
OCCUPIED_SETTLE_MAX = 8.0       # 找地格时点到过站人的格子：详情卡要换成那个人 —— 等它换过来（最多这么久），换不过来就不往下点
UNIT_READ_REASONS = {'no_name_box': '没读到名字', 'two_name_boxes': '读到两个名字', 'no_lv': '没读到等级',
                     'name_unreadable': '名字读不出', 'low_conf': '名字读得不清楚', 'glyph_height': '字的大小不对',
                     'glyph_width': '字的宽度不对', 'enemy_suffix': '混进了敌人名字', 'enemy_text': '混进了敌人名字',
                     'enemy_bar': '旁边有敌人血条', 'unreadable': '读不出'}
OWN_PANEL_BOX = (0.74, 0.0, 1.0, 0.93)   # 页签 + 下拉框 + 筛选条 + 卡片（support_panel_box 从 0.12 起，看不到页签）


class LineupShortError(Exception):
    """「本关至少需要出击N人」，从「我的人形」补不上（或补人关着）—— 停手。

    risky = None：场上的人都核对过，等你放够 N 个、替你点「作战开始」。
    risky = 说明文字：场上可能站着个没核对 / 不能上的人（点卡之后场上像是变了、点地格时放上去了谁、你在刚点完卡时接手……）。
    轮询就不再碰这个部署界面、也不替你开打，直到这一关往下走了（你开打 / 退出）。
    """

    def __init__(self, need, count, why, added=(), skipped=(), risky=None):
        self.need, self.count, self.why = need, count, why
        self.added, self.skipped = list(added), list(skipped)
        self.risky = risky
        super().__init__(why)


class OwnUnitsMixin:
    """补人。用到 SupportBattleMixin 的 pick_empty_tile / tile_state / read_deploy_count / support_detail_box，
    键鼠都走 stage_runner 的输入关口（_sr_guarding 时你一动就抛 UserTakeoverError）。"""

    def read_unit_card(self):
        """左下角详情卡 -> unit_name.parse_unit_card_boxes 的结果（support / name / level / candidates / reason）。"""
        self.next_frame()
        return parse_unit_card_boxes(self.ocr(box=self.support_detail_box()), self.width, self.height,
                                     frame=self.frame)

    @staticmethod
    def _card_key(card):
        """详情卡「是哪张」：名字 + 有没有「支援」（自己的可露凯和助战可露凯名字一样，靠「支援」分开）。"""
        return card['name'], bool(card['support'])

    def _own_card_settled(self):
        """点卡之前详情卡上是哪张 -> (card, 稳了没有)。连着两次读到同一张才算稳（详情卡在换的时候读到的是哪张说不准）；
        这一次补人已经认出过名字，现在却读成空的，也不算稳（多半是一次没读出来，不是真空了）。"""
        card = self.read_unit_card()
        for _ in range(CARD_SETTLE_READS - 1):
            self.sleep(0.6)
            again = self.read_unit_card()
            if self._card_key(again) == self._card_key(card):
                blank = not (again['name'] or again['support'])
                return again, not (blank and self._fill_seen)
            card = again
        return card, False

    def _click_tile(self, tile, why=''):
        """找地格时点到站人的格子：详情卡会换成那个人（可能晚一点）—— 记下来，认「点之前是谁」之前多等一会。"""
        st = super()._click_tile(tile, why)
        if st == TILE_OCCUPIED and getattr(self, '_fill_active', False):
            self._fill_occupied_hits = getattr(self, '_fill_occupied_hits', 0) + 1
        return st

    def _own_card_after(self, pre):
        """点卡之后详情卡上是哪张 -> (card, 认不准)。

        还是点之前那张（pre，空的也算）就再等等，一直是 = 没刷新；换了也要再读一次、两次一样才算数
        （还在变 = 读到的可能是刷到一半 / 上一张迟到的）。"""
        key = self._card_key(pre)

        def waiting(c):
            # 还是点之前那张，或者读成空的（刷到一半 / 一次没读出来）：再等等
            return self._card_key(c) == key or not (c['name'] or c['support'])
        card = self.read_unit_card()
        n = 1
        while waiting(card) and n < CARD_REFRESH_READS:
            self.sleep(0.6)
            card = self.read_unit_card()
            n += 1
        if waiting(card):
            if self._card_key(card) == key and (card['name'] or card['support']):
                return card, True                      # 一直是点之前那张：没刷新
            return card, False                         # 一直是空的：读不出名字（交给「名字读不出」那条）
        # 换了：再读最多两次，要连着两次一样、而且不是空的才算数（还在变 = 读到的可能是刷到一半 / 上一张迟到的）
        for _ in range(2):
            self.sleep(0.6)
            again = self.read_unit_card()
            if self._card_key(again) == self._card_key(card):
                return again, False
            card = again
        return card, True

    def _own_panel(self):
        self.next_frame()
        x0, y0, x1, y1 = OWN_PANEL_BOX
        boxes = self.ocr(box=self.box_of_screen(x0, y0, x1, y1, name='own_panel'))
        return read_own_panel(self.frame, boxes, self.width, self.height)

    def _own_click(self, x, y, name, risky=None):
        """点面板上的一个点。下拉框那一块（own_panel.DANGER_ZONE）一律不点 —— 返回 False。
        risky：点出去之后（等界面反应的那段里你接手的话）场上可能有什么没核对 —— 点出去那一刻才记上。"""
        if in_danger_zone(x, y, self.width, self.height):
            self.log_info('「%s」的位置 (%d,%d) 落在「使用整备室配置」下拉框上，不点' % (name, x, y))
            return False
        self.click(int(x), int(y), name=name)
        if risky:
            self._fill_risky = risky
        self.sleep(OWN_CLICK_SLEEP + self.ui_delay)
        return True

    def _own_ready_panel(self):
        """确认右边是「我的人形」页、筛选是「全部」-> (panel, None)；不行 -> (panel, 原因)。最多点一次页签、一次「全部」。"""
        p = self._own_panel()
        if p['own_tab'] is False and p['my_tab_box']:
            _, x, y, w, h = p['my_tab_box']
            self.log_info('右边面板是「好友助战」页，切到「我的人形」')
            self._own_click(x + w / 2.0, y + h / 2.0, '我的人形')
            p = self._own_panel()
        if p['own_tab'] is not True:
            return p, '右边的人形列表没切到「我的人形」（页签认不准）'
        if p['empty'] or p['dropdown'] == 'formation':
            return p, '「我的人形」列表是空的（%s）—— 那个下拉框我不动' % (
                '暂无可选择人形' if p['empty'] else '下拉框是「%s」' % (p['dropdown_box'] or ('编队',))[0])
        if p['dropdown'] is None:
            return p, '没认出「我的人形」页签下面的「使用整备室配置」'
        if p['filter'] is None:
            self.sleep(0.5)                        # 可能正在切换：再看一眼
            p = self._own_panel()
            if p['own_tab'] is not True or p['filter'] is None:
                return p, '「我的人形」的元素筛选认不准（条子可能被拖动过），不点'
        if p['filter'] != 0:
            self.log_info('「我的人形」的元素筛选不是「全部」（借助战时选的会带过来），切回「全部」')
            self._own_click(p['all_click'][0], p['all_click'][1], '筛选_全部')
            p = self._own_panel()
            if p['own_tab'] is not True or p['filter'] != 0:
                return p, '元素筛选切不回「全部」'
        return p, None

    def _own_checked(self, panel):
        return collections.Counter(c['power'] for c in panel['cards'] if c['checked'] and c['sure'])

    def _own_no_support_check(self, count, panel):
        """没认出场上有助战时：部署数和「我的人形」里打勾的人数对得上 -> None；对不上 / 读不到 -> 不补的原因。"""
        if count is None:
            return '读不到部署数，确认不了场上有没有助战（怕补上和助战同一个角色）'
        sure = sum(self._own_checked(panel).values())
        unsure = sum(1 for c in panel['cards'] if c['checked'] and not c['sure'])
        if count[0] != sure or unsure:
            return ('部署数 %d，「我的人形」里认准打勾的只有 %d 个%s —— 可能有助战（读不出是谁），也可能勾没认准，'
                    '怕补上和助战同一个角色，先不补' % (count[0], sure, ('（还有 %d 个拿不准）' % unsure) if unsure else ''))
        return None

    def _own_confirm(self, ref, power, checked_before):
        """按完「部署人形」：部署数涨了 / （读不到部署数时）这张卡打上了勾 -> True。
        部署数读得到、又没涨，就不看勾（勾是退路，不能拿来推翻读得到的部署数）。"""
        readable = False
        for i in range(3):
            if i:
                self.sleep(0.6)
            self.next_frame()
            after = self.read_deploy_count()
            if ref and after:
                if after[0] > ref[0]:
                    return True
                readable = True
        if readable:
            return False
        p = self._own_panel()
        return bool(p['own_tab'] and self._own_checked(p)[power] > checked_before[power])

    def _own_user_touched(self, t0):
        """补人开始之后你动过游戏（不光是最近 2.5 秒）-> 抛 UserTakeoverError（risky）：你可能点了别的卡、放了人，
        场上的人我核对不了。输入关口只看最近 2.5 秒，按「部署人形」之前那段等按钮的空当里你点的会漏掉（10/11 第三轮复查）。"""
        ops = getattr(self, '_user_ops_since', None)
        got = ops(t0) if ops else None
        if got is not None:
            touched = bool(got)          # 分得出来：只看你操作没操作游戏（点开游戏看、挪鼠标不算，I-52）
        else:
            recent = getattr(self, '_user_recent_seconds', None)
            idle = recent() if recent else None
            touched = idle is not None and idle < self._now() - t0
        if touched:
            self._fill_risky = '补人途中你动过游戏（场上的人我没法核对）'
            raise UserTakeoverError('补人途中你动了键鼠')

    def fill_unexplained(self, final):
        """部署数比我给核对过的人放的还多 -> 原因；对得上 / 核对不了 -> None。
        fill_own_units 正常结束时调；你中途接手 / 出错时 stage_runner 也调（不然那条路上多出来的人没人管）。"""
        start = getattr(self, '_fill_start', None)
        pressed = getattr(self, '_fill_pressed', 0)
        if start and final and final[0] - start[0] > pressed:
            return '部署数从 %d/%d 涨到 %d/%d，比我放的 %d 个多' % (start[0], start[1], final[0], final[1], pressed)
        return None

    def fill_own_units(self, need, avoid=None):
        """从「我的人形」补到 need 个人。

        :param avoid: 场上助战的名字（不上同一个角色）；None = 没认出场上有助战 —— 这时要拿场上的人数核对：
                      部署数比「我的人形」里打勾的多，就是有人不在「我的人形」里（多半是你自己借的助战，读不出是谁），不补
        :return: dict(added=[补上的名字], skipped=[(谁, 为什么没上)], count=最后读到的部署数,
                      why=停下的原因（补够了是 None）)
        :raises LineupShortError: risky —— 场上可能多了个没核对的人（见模块说明），一定要停手叫你
        :raises UserTakeoverError: 你中途动了键鼠（输入关口抛的，原样往外；刚点完卡时 self._fill_risky 有值）
        """
        t0 = self._now()
        added, skipped, tried = [], [], set()
        self._fill_added, self._fill_skipped = added, skipped      # 你中途接手时 stage_runner 从这里拿
        self._fill_pressed = 0                              # 给核对过名字的人放了几个（按「部署人形」/ 点卡直接上场）
        self._fill_risky = None
        self._fill_seen = set()                             # 这一次补人详情卡上出现过的（名字, 支援）
        self._fill_active = True
        no_clicked = False                                  # 这一次补人点过「不能上」的卡（之后部署数无故涨了不能算到别人头上）
        count = self.read_deploy_count()
        for _ in range(2):
            if count:
                break
            self.sleep(0.5)                                 # 读不到：再看两眼（读得到才能在最后核对人数）
            count = self.read_deploy_count()
        start = self._fill_start = count
        todo = max(1, need - count[0]) if count else 1      # 读不到部署数：先补 1 个，点「作战开始」还提示再补
        why = None
        for _ in range(todo + 2):                           # 多给两次：地格丢了 / 按了没放上去重来
            if len(added) >= todo or (count and count[0] >= need):
                break
            if self._now() - t0 > OWN_FILL_SECONDS:
                why = '补人用了 %d 秒还没补够' % (self._now() - t0)
                break
            # 1) 空蓝格（「找不到地格时挪视角」关掉、或者连着几关都找不到，就只在当前画面找 —— 和借助战一样）
            self._count_ref = count
            self._lineup_changed = self._lineup_full = False
            self._fill_occupied_hits = 0
            d0 = self._card_key(self.read_unit_card()) if avoid else None   # 找地格之前详情卡上是哪张
            if d0 and (d0[0] or d0[1]):
                self._fill_seen.add(d0)      # 之后点卡读到它 = 多半是迟到的旧内容；之后读成空的「点之前」也不算稳
            move = (getattr(self, '_move_view_for_tile', True)
                    and getattr(self, '_tile_search_failures', 0) < TILE_SEARCH_GIVE_UP)
            if not self.pick_empty_tile(move_view=move):
                if getattr(self, '_lineup_full', False):
                    why = '阵容已经满了'
                    break
                if getattr(self, '_lineup_changed', False):
                    # 点地格时部署数自己涨了：多半是把选着的人放上去了（9/26 见过），是谁没核对
                    raise LineupShortError(need, self.read_deploy_count(), '找空地格时部署数自己涨了', added, skipped,
                                           risky='点地格时放上去了一个人（没核对是谁）')
                why = '找不到能放人的空地格'
                break
            # 2) 「我的人形」页、筛选「全部」
            panel, bad = self._own_ready_panel()
            if bad:
                why = bad
                break
            checked_before = self._own_checked(panel)
            if avoid is None:
                # 没认出场上有助战：拿人数核对。列表里打勾的（已上场的）排在最前面，筛选是「全部」时都看得见。
                # 对不上先等半秒再看一次（列表淡入时勾会读成「拿不准」）
                bad = self._own_no_support_check(count, panel)
                if bad:
                    self.sleep(0.5)
                    count = self.read_deploy_count() or count
                    panel = self._own_panel()
                    checked_before = self._own_checked(panel)
                    bad = self._own_no_support_check(count, panel)
                if bad:
                    why = bad
                    break
            cands = sorted((c for c in panel['cards'] if not c['checked'] and c['power'] not in tried),
                           key=lambda c: -c['power'])
            if not cands and any(c['checked'] and not c['sure'] for c in panel['cards']):
                self.sleep(0.8)                              # 列表可能还在淡入（勾读成「拿不准」）：再看一次
                panel = self._own_panel()
                checked_before = self._own_checked(panel)
                cands = sorted((c for c in panel['cards'] if not c['checked'] and c['power'] not in tried),
                               key=lambda c: -c['power'])
            if not cands:
                why = ('「我的人形」这一页没有能上的人了（没打勾的都试过了）' if tried
                       else '「我的人形」这一页没认出没上场的人（勾都拿不准 / 都打勾了）')
                break
            if avoid and self._fill_occupied_hits:
                # 找地格时点到过站人的格子：详情卡会换成那个人。等它真换过来再认「点之前是谁」——
                # 不然它晚一点换过来，会被当成接下来点的那张卡（10/11 第三轮复查）
                # 「换过来了」要连着两次读到同一张、不是空的、也不是找地格之前那张（读错一次不算）
                waited, last, changed = 0.0, None, False
                while waited < OCCUPIED_SETTLE_MAX:
                    k = self._card_key(self.read_unit_card())
                    if k != d0 and (k[0] or k[1]) and k == last:
                        changed = True
                        break
                    last = k
                    self.sleep(0.6)
                    waited += 0.6
                if not changed:
                    why = '找地格时点到了站人的格子，详情卡一直没换过来，认不准之后点的是谁'
                    break
            placed, lost_tile = False, False
            for c in cands[:OWN_CARD_TRIES]:
                tried.add(c['power'])
                x, y = c['click']
                ref = self.read_deploy_count() or count          # 每张卡点之前重读：和点之后比
                pre = None
                if avoid:                                         # 场上有助战才要认名字
                    pre, settled = self._own_card_settled()
                    if not settled:
                        why = '点卡之前详情卡一直在变 / 读不准（认不出点之前是哪张），不往下点了'
                        break
                    self._fill_seen.add(self._card_key(pre))
                if not self._own_click(x, y, 'own_card_%d' % c['power'],
                                       risky='作战效能 %d 那张（刚点了，还没核对上没上场、是谁）' % c['power']):
                    continue
                if avoid:
                    card, unsure = self._own_card_after(pre)
                else:
                    card, unsure = self.read_unit_card(), False
                # 这张能不能上（名字核对过）：场上没助战 = 都能上；有助战 = 认准了、不是助战、和助战不是同一个角色
                stop, no = None, None
                if avoid:
                    key = self._card_key(card)
                    if card['support']:
                        stop = '详情卡上还是助战（「%s」），认不准点的是谁' % (card['name'] or '读不出名字')
                    elif unsure:
                        stop = '详情卡没刷新 / 还在变（点之前是「%s%s」），认不准点的是谁' % (
                            pre['name'] or '空的', '（支援）' if pre['support'] else '')
                    elif not card['name']:
                        stop = '名字读不出（%s），怕和助战「%s」是同一个角色' % (
                            UNIT_READ_REASONS.get(card['reason'], '读不出'), avoid)
                    elif key in self._fill_seen:
                        # 这一次补人里详情卡上出现过的名字又出现了：多半是之前那张迟到的，不是刚点的这张
                        stop = '详情卡上的「%s」之前出现过，认不准是不是刚点的这张' % card['name']
                    elif card_matches(card, avoid):
                        no = '和助战「%s」是同一个角色' % avoid
                    self._fill_seen.add(key)
                # 名字靠不住时不拿读到的名字称呼它（读到的多半是助战 / 上一张）
                label = ('作战效能 %d 那张' % c['power']) if stop else (card['name'] or '作战效能 %d 那张' % c['power'])
                ok_to_go = not (stop or no)
                now = self.read_deploy_count()
                tile = self.tile_state(rereads=2)
                readable = bool(ref and now)
                rose = bool(readable and now[0] > ref[0])
                # 只点了卡、还没按「部署人形」：部署数不该涨、右下角应该还是选着空地格（借助战一直是这样）
                if rose or tile == TILE_OCCUPIED or (tile != TILE_EMPTY and not readable):
                    if not ok_to_go and not rose and not readable and tile != TILE_OCCUPIED:
                        # 部署数读不到、右下角读不出：拿勾核对 —— 这张没打上勾、打勾的人数也没变，就是场上没变
                        p2 = self._own_panel()
                        if p2['own_tab'] and self._own_checked(p2) == checked_before and not any(
                                x['power'] == c['power'] and x['checked'] for x in p2['cards']):
                            self._fill_risky = None
                            why = '点了「%s」之后右下角读不出、部署数也读不到（场上没变），不往下点了' % label
                            break
                    if ok_to_go and rose and no_clicked:
                        raise LineupShortError(
                            need, now, '点了「%s」部署数涨了，可之前点过不能上的卡 —— 涨的可能是那张' % label, added, skipped,
                            risky='之前点过的不能上的那张（可能晚一点上场了）')
                    if ok_to_go and rose:
                        # 点卡就直接上场了（没见过），好在是核对过能上的人：算补上
                        added.append(label)
                        self._fill_pressed += 1
                        count = now
                        self._fill_risky = None
                        self.log_info('补上了「%s」（点卡就直接上场了）：部署人形 %d/%d' % ((label,) + now))
                        placed = True
                        break
                    if not ok_to_go:
                        raise LineupShortError(
                            need, now or ref, '点了「%s」之后场上像是变了（部署人形 %s -> %s，右下角「%s」），没见过这种情况' % (
                                label, '%d/%d' % ref if ref else '?', '%d/%d' % now if now else '?',
                                '、'.join(getattr(self, '_last_br', []) or ['读不出'])),
                            added, skipped, risky='「%s」可能已经上场了（%s）' % (label, stop or no))
                    self._fill_risky = None
                    self._fill_pressed += 1                        # 核对过能上的：就算上场了也不是没核对的人
                    why = '点了「%s」之后右下角是「%s」、部署数%s，看不准它上没上场' % (
                        label, '、'.join(getattr(self, '_last_br', []) or ['读不出']), '没涨' if readable else '读不到')
                    break
                self._fill_risky = None                          # 场上没变：点的这张还没上场
                if stop:
                    # 认不准这张是谁：不往下点了（换下一张时读到的可能是这张迟到的名字）—— 地格丢没丢都一样
                    skipped.append((label, stop))
                    self.log_info('点开「%s」：%s —— 不往下点了' % (label, stop))
                    why = '点开「%s」%s' % (label, stop)
                    break
                if no:
                    no_clicked = True
                    skipped.append((label, no))
                    self.log_info('「%s」不上：%s' % (label, no))
                    if tile != TILE_EMPTY:
                        lost_tile = True                         # 选中的空地格丢了：回去重选
                        break
                    continue
                if tile != TILE_EMPTY:
                    skipped.append((label, '点完卡选中的空地格丢了'))
                    lost_tile = True                             # 部署数没涨、选中的空地格丢了：回去重选
                    break
                self._own_user_touched(t0)
                if avoid:
                    # 按按钮之前再看一眼详情卡：还是刚核对的那张（中间你点了别的卡、或者它自己变了，就不按）
                    last = self.read_unit_card()
                    if self._card_key(last) != self._card_key(card):
                        why = '要按「部署人形」时详情卡变成了「%s」（刚核对的是「%s」），不按了' % (
                            last['name'] or '读不出', card['name'])
                        break
                if not self.wait_click_ocr(match=DEPLOY_BTN, box=self.box.bottom_right, time_out=4,
                                           after_sleep=1.0 + self.ui_delay, raise_if_not_found=False, log=True):
                    skipped.append((label, '右下角没找到「%s」按钮' % DEPLOY_BTN))
                    lost_tile = True
                    break
                self._fill_pressed += 1
                if self._own_confirm(ref, c['power'], checked_before):
                    added.append(label)
                    count = self.read_deploy_count() or ((count[0] + 1, count[1]) if count else None)
                    self.log_info('补上了「%s」%s' % (label, ('：部署人形 %d/%d' % count) if count else ''))
                    placed = True
                else:
                    skipped.append((label, '按了「%s」没放上去' % DEPLOY_BTN))
                    lost_tile = True
                break
            if why:
                break
            if not placed and not lost_tile:
                why = '点开的 %d 张卡都不能上' % len(cands[:OWN_CARD_TRIES])   # 每张为什么不上在 skipped 里
                break
        self._fill_risky = None
        self._fill_active = False
        if added:
            self._own_user_touched(t0)                       # 补上了人：你中途动过游戏的话场上的人不可信
        final = self.read_deploy_count()
        extra = self.fill_unexplained(final)
        if extra:
            raise LineupShortError(need, final, extra, added, skipped, risky='多出来的那个人（没核对是谁）')
        count = final or count
        if why is None and len(added) < todo and not (count and count[0] >= need):
            why = ('试了几轮只补上 %d 个' % len(added)) if added else '没补上人'
        return dict(added=added, skipped=skipped, count=count, why=why)
