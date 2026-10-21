# -*- coding: utf-8 -*-
"""测试 / 调试用：只跑一次「借好友助战」。

用法：手动在游戏里进到某一关的**部署界面**（能看到「作战开始」和右上角
「我的人形 / 好友助战」），再点这个任务的「开始」。全程不按「作战开始」。

- 「助战角色名」留空  → 只普查：逐张点开助战卡片，把 作战效能 -> 角色名 打到日志里，
  用来确认她在游戏内叫什么、排在第几个。
- 「助战角色名」填了  → 走和推图里完全相同的借人流程（含元素筛选、可选退出重进）。

日志里会先把蓝色区域切成一格一格、打出每格的内接半径，再打元素筛选前后的候选
数量 —— 调「助战元素筛选」和排查「点不中地格」就看这两处。
"""

from src.core.BaseGfTask import BaseGfTask

from support_pick import (SupportPickMixin, MIN_TILE_RADIUS_RATIO,
                          TILE_NEAR_UNIT_RATIO, PAN_METHOD_NAMES, TILE_STATE_NAMES,
                          SupportTabLostError, TileNotSelectedError, UserTakeoverError)


class TestSupportPickTask(SupportPickMixin, BaseGfTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '测试:好友助战'
        self.description = '在部署界面单独跑一次借助战，不开战'
        self._support_power_hint = None
        self.default_config.update({
            '助战角色名': '威玛西娜',
            '备选助战角色名': '可露凯',
            '助战元素筛选': 2,
            '备选助战元素筛选': 6,
            '助战最低战力': 10000,
            '助战后自动部署': False,
            '助战列表翻页次数': 3,
            '找不到时重进关卡次数': 0,
            '界面等待(秒)': 1.5,
            '探测镜头平移方式': False,
        })
        self.config_description.update({
            '助战角色名': (
                '留空 = 只普查：把好友助战列表里每张卡对应的角色名打到日志\n'
                '填了 = 按名字找人并选中（不会按「作战开始」）'
            ),
            '备选助战角色名': (
                '「助战角色名」在这份好友列表里找不到（或战力不到门槛）时，改借这个角色（默认：可露凯）\n'
                '就在同一份好友列表里找（按「备选助战元素筛选」那一列），不会为了主选去退出重进；'
                '两个都没有才退出重进换一批好友\n'
                '可以填多个，用逗号隔开，按先后优先；留空 = 不用备选'
            ),
            '助战元素筛选': (
                '「好友助战」下方元素条点第几个图标（1~6，从左数），0 = 不筛选\n'
                '普查模式（角色名留空）下也会生效，方便确认第几个是对的'
            ),
            '备选助战元素筛选': (
                '备选角色在元素条的第几列（1~6，从左数）。主选那一列里没有够强的主选时，换到这一列找备选\n'
                '默认 6 = 可露凯那一列；填 0 = 和主选在同一列里一起找（找不到再全量查找）'
            ),
            '助战最低战力': '作战效能低于这个数就换下一张，0 = 不看战力',
            '助战后自动部署': '选中后是否顺手点一个空地格把她放上场',
            '助战列表翻页次数': '最多往下翻几页',
            '找不到时重进关卡次数': (
                '默认 0 = 不退出关卡，只在当前界面找一遍（测试时更安全）\n'
                '想单独验证「左上角 -> 退出作战 -> 重新挑战」这条路径时才调成 1'
            ),
            '界面等待(秒)': '每次点击后额外等多久，给界面动画留时间；点太快会误触把面板收回去',
            '探测镜头平移方式': (
                '依次试 WASD / 右键拖拽（每种两个方向都试），再试一次滚轮缩放，'
                '每一下的实测位移都打到日志。「关卡内移动不生效」时打开它跑一次就知道该用哪种'
            ),
        })
        # 不给 min 的话 SpinBox 对正数默认值会把下限定成 1，「填 0」就点不到了
        self.config_type.update({
            '助战元素筛选': {'min': 0, 'max': 6},
            '备选助战元素筛选': {'min': 0, 'max': 6},
            '助战最低战力': {'min': 0, 'max': 999999},
        })

    def _int_config(self, key, default):
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def run(self):
        self.next_frame()
        pages = self._int_config('助战列表翻页次数', 3)
        element = self._int_config('助战元素筛选', 2)
        reenter = self._int_config('找不到时重进关卡次数', 0)

        count = self.read_deploy_count()
        self.log_info('部署人形 = ' + (str(count) if count else '读不到'))

        # 右下角原文：排查「蓝格点不中」时最有用的一行 ——
        # 「部署人形」= 选中了空格；「撤回人形 / 编入队伍」= 选中的是站着人的格子；
        # 「请点击地格」= 没选中；都没有 = OCR 没读到
        st = self.tile_state(rereads=0)
        self.log_info('右下角文字 = %s -> 地格选中状态 = %s'
                      % (getattr(self, '_last_br', []) or '(空)', TILE_STATE_NAMES[st]))

        w = self.width
        need = MIN_TILE_RADIUS_RATIO * w
        cand = self.find_deploy_tiles(empty_only=False)
        for t in cand:
            r = t.width / 2.0
            self.log_info('切格 中心=(%d,%d) 内接半径=%.0fpx (门槛 %.0fpx)  <- %s' % (
                t.x + t.width // 2, t.y + t.height // 2, r, need,
                '进候选' if r >= need else '太小，多半不是格子'))
        pts = self.unit_badge_points()
        self.log_info('人形编号牌: %s（锚点半径 %.0fpx，只影响先后顺序）' % (
            str(pts) if pts else '一个都没认出来', TILE_NEAR_UNIT_RATIO * w))
        order = self.tile_candidates(skip_tried=False)
        self.log_info('正式流程会按这个顺序去点（hole=边框围住的洞 / tile=切格 / blob=切块 / grid=撒点）：')
        for i, c in enumerate(order):
            self.log_info('  %2d. (%d,%d) %s' % (i + 1, c.x + c.width // 2, c.y + c.height // 2, c.name))
        if not order:
            self.log_info('!! 一个候选都没有 —— 当前视野里多半没有空地格')
        blink = self.find_blinking_tile(order[:8])
        self.log_info('闪烁判据: %s' % (
            '认出 (%d,%d)' % (blink.x + blink.width // 2, blink.y + blink.height // 2)
            if blink is not None else '判不出来（选中助战之前本来就不闪；或候选不足 2 个）'))
        self._save_tile_debug('test')   # 原图 + 标注图存到 debug_tiles/，对照着看候选落点

        if self.config.get('探测镜头平移方式', False):
            # 横纵分开探测 —— 实测有的游戏 WASD 只能上下、不能左右
            for axis, label in (('x', '横向'), ('y', '纵向')):
                m = self.detect_pan_method(axis)
                self.log_info('%s平移方式：%s' % (
                    label, PAN_METHOD_NAMES[m] if m else '怎么都挪不动'), notify=True)
            # 滚轮：两个方向各试一下（镜头可能刚好顶在一侧的极限上），生效了就滚回来
            result = '没效果'
            for count in (-2, 2):
                r = self._zoom(count)
                if r:
                    self._zoom(-count)
                    result = '能用'
                    break
                if r is None:
                    result = '判不了（画面自己在大幅变化）'
                    break
            self.log_info('滚轮缩放：%s' % result, notify=True)

        try:
            if not self.switch_to_support_tab():
                raise Exception('没找到「好友助战」页签，请先手动进到关卡的部署界面')
        except SupportTabLostError as e:
            self.log_info('%s，停下' % e, notify=True)
            return

        target = (self.config.get('助战角色名') or '').strip()
        if not target:
            try:
                if element:
                    self.select_element_filter(element)
                found = self.scan_support_cards(max_pages=pages)
            except SupportTabLostError as e:
                self.log_info('%s，停下' % e, notify=True)
                return
            self.log_info('普查完成，共 ' + str(len(found)) + ' 张卡；对照日志找出她的名字后，'
                          '填进「推图(借好友助战)」的「助战角色名」', notify=True)
            return

        try:
            delay = float(self.config.get('界面等待(秒)', 1.5))
        except (TypeError, ValueError):
            delay = 1.5
        try:
            ok = self.borrow_support(target, max_pages=pages, ui_delay=delay,
                                     deploy=bool(self.config.get('助战后自动部署', False)),
                                     element_index=element, reenter_times=reenter,
                                     backup_names=self.config.get('备选助战角色名') or '',
                                     backup_element_index=self._int_config('备选助战元素筛选', 6),
                                     min_power=self._int_config('助战最低战力', 10000))
        except TileNotSelectedError as e:
            # 正式流程里这会让实时辅助停手等人接管；测试任务只报一声就好
            self.log_info('选不中空地格，没有往下选人：%s' % e, notify=True)
            return
        except UserTakeoverError as e:
            self.log_info('%s，停下' % e, notify=True)
            return
        self.log_info(('已选中「' + target + '」') if ok else ('没找到「' + target + '」'), notify=True)
