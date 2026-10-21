# -*- coding: utf-8 -*-
"""借「好友助战」的共用逻辑（ok-gf2 扩展）。

本文件只提供 ``SupportPickMixin`` 和一个异常类，**不定义任何 BaseTask 子类**：
ok 的任务扫描器会给 ok_import 下每个 .py 找第一个 BaseTask 子类并实例化成
一个任务条目，这里没有这样的类，所以不会多出一个空任务。

界面依据（全部用相对坐标，不写死像素）：
  - 部署界面右上角有「我的人形 / 好友助战」两个页签
  - 页签下面一条 6 个元素图标的筛选条，点一下只剩该元素的助战，找人快很多
  - 好友助战列表每行 3 张卡片，卡片下方是「作战效能」数字 —— 用它当锚点
  - 点中某张卡后，左下角详情卡会显示「支援」标签 + 角色名 + Lv.xx
  - 可部署地格是青色描边的方块，右下角提示「请点击地格或拖拽以部署单位」
  - 顶部有「部署人形 3/4」，用来确认部署是否真的生效
  - 左上角 ☰ 打开「关卡信息」，左下角「退出作战」、右下角「继续作战」
"""

import collections
import inspect
import json
import os
import re
import time

import cv2
import numpy as np
from ok import Box, Logger

from deploy_count import read_deploy_count_boxes
from own_panel import own_tab_active

try:
    from ok import FinishedException, TaskDisabledException
except ImportError:            # 离线回放截图时 ok 是个桩模块
    class TaskDisabledException(Exception):
        pass

    class FinishedException(Exception):
        pass

# 用户在界面上点了「停止」时 ok 抛的异常。**绝不能被 except Exception 吞掉**：
# ok 在抛之前就把 current_task 清空了，吞掉之后后面的 sleep/next_frame 不会再抛，
# 任务会在用户按了停止之后继续点来点去。
STOP_EXCEPTIONS = (TaskDisabledException, FinishedException)

logger = Logger.get_logger(__name__)

# 好友助战卡片下方的「作战效能」。1~2 位数字（元素角标、页码）会被排除。
POWER_RE = re.compile(r'^\d{3,7}$')
# 顶部「部署人形 3/4」的计数。OCR 经常把那个斜杠读成 1 / l / I / |（实测「3/4」->「314」），
# 所以中间那一位按「任意单个非数字，或数字 1」来容忍，两头用 fullmatch 卡死只能是一位数。
# **分隔符必须有，而且只能是长得像斜杠的字符**：以前是「可选的任意非数字」，于是同一条横带里
# 敌人名字「莫普1.5·试作版」的「1.5」、甚至裸的「15」都会被当成「1/5」（2026-09-26 13:18 实测误判过）。
# 光靠正则还不够，read_deploy_count 另外只认「部署人形」标签**同一行右侧**的那个框。
DEPLOY_COUNT_RE = re.compile(r'^(\d)\s*[/／\\|丨lIi1]\s*(\d)$')   # 10/11 起 read_deploy_count 改用 deploy_count.py（放宽 + 位置闸）

# 可部署地格的蓝色，OpenCV HSV 口径（H: 0-180）。
#
# **色相是唯一可靠的判据**，而且下界卡得很死。八张真实截图上量到的中位色相：
#     能部署的格子   H = 102 / 104 / 104 / 104 / 105 / 107 / 108
#     雾天地图的景物 H = 100 / 100 / 100 / 101     <- 蓝灰色的雾和地面
#     浅青色的假格子 H =  94
# 所以下界取 **102**：再松一格（101）雾就开始漏进来，蒙出比真格子还大的假峰
# （图14 里 4 个候选全是雾）；再紧一格（103）又会把图12 那种 H=102 的格子整片切没。
# 饱和度不能当主判据 —— 真格子 S 从 79 到 248 都有（图7 只有 79，图12 有 248），
# 和雾（S≈80）重叠得厉害；亮度同理。所以只卡死色相，S/V 放得很宽。
TILE_HSV_LOWER = (102, 60, 60)
TILE_HSV_UPPER = (115, 255, 255)

# 空地格怎么找：先把连成一片的蓝色按「格子亮边框」切成一格一格，再逐格取最大内接圆。
#
# 不能直接对整片蓝色做距离变换：田字格那种 2x2 连排时整片蓝是**实心**的 ——
# 人形画在格子上方，脚底只压住一点点，几乎不在蓝色里挖洞。实测 3052px 宽的画面上，
# 站了人的格子内接半径 45.3px、空格 46.5px，根本分不开。
#
# 格子之间那圈亮边框是分界：实测边框 V≈250，格子内部 V≈146~155。
# 绝对阈值不通用（不同关卡整体亮度差很多），所以取**蓝色像素自身亮度的 92 分位数**
# 当边框阈值，切完再按连通域拆格子。
# 注意这条**在雾大的地图上不灵**（图14 边框 V=154 反而比内部 V=159 还暗），
# 那种图上一整排格子会连成一块、只给出一个候选 —— 靠后面「部署计数当裁判」兜住。
# 阈值**按每个蓝色连通块各自算**，不能全图一个。图18 里中间那条明亮走廊也是蓝色
# （H=105，和真格子一模一样，色相根本分不开），它一大，全局分位数被抬高，
# 真格子内部（V=169）反而高过阈值、被当成边框整格抹掉 —— 真格子直接消失。
TILE_RIDGE_PERCENTILE = 92
TILE_MIN_AREA_RATIO = 0.0004      # 面积小于全屏这个比例的碎片不当格子
# 切开之后单格的内接半径小了很多（不再跨格连通），门槛相应下调。
# 实测：有空格时 46 / 58 / 65 / 97 px，整片都站满人时最大只有 40 / 18 px。
# **排序用几何量，不用亮度**。地格是叠加在 3D 场景上的发光覆盖层，颜色取决于底下渲染
# 的是什么，所以「哪块蓝色更亮/更开阔」这种光度判据换张地图就翻车（H 从 99 调到 102、
# 全局分位数改按块算、都是被新地图逼出来的）。真正跨地图不变的是**形状**：
# 地格是等轴投影的菱形 —— 轮廓多边形逼近恰好 4 个顶点，且面积几乎填满自己的最小外接
# 旋转矩形。实测九张截图（含雾天、明亮走廊）：
#     真空格      顶点=4（9 张里 8 张），矩形度 0.73~0.88
#     雾/走廊/杂块 顶点 6~11，矩形度 0.20~0.85
# 所以排序键 = (是不是四边形, 矩形度)，半径只用来卡上下限、不参与排序。
TILE_QUAD_EPS = 0.03          # approxPolyDP 的 epsilon，取周长的比例
MIN_TILE_RADIUS_RATIO = 0.014
# 也要卡**上限**：地格在屏幕上有固定的大小范围（镜头缩放有限），大过这个的一定不是格子。
# 图18 那条走廊的内接半径是 0.070*屏宽，而九张截图里最大的真格子只有 0.042*屏宽。
MAX_TILE_RADIUS_RATIO = 0.05
# 切出来的块只差一点点够格（≥ 门槛的这个比例）时存一张图（每次找地格最多一张）。9/30 23:44 DF-1-1 第 3 阶段：
# 镜头离得远、格子只有第 1 阶段的一半大，挪了几个视野都是「最大 38~44px（要求 45px）」，最后放不上人 ——
# 那几块是不是空格、要不要按地图比例放宽门槛，没有那几帧的图判断不了（ISSUES.md I-16）
TILE_NEAR_RATIO = 0.75
MAX_TILE_CANDIDATES = 4          # 「切格」这一级最多给几个（在排除已试过的格子**之后**才截断）
# 切格子失败时的兜底：直接在蓝色区域上撒一批落点逐个试。
# 切缝这一步已经被四种地图分别坑过（雾、明亮走廊、暗边框、格内亮三角把格子切碎），
# 与其继续给它打补丁，不如承认「有时候就是切不开」，改成**不求认准、只求覆盖到** ——
# 反正点一下读一次右下角就 1.5 秒，撒 8 个点也才十几秒，比挪视角便宜得多。
GRID_FALLBACK_POINTS = 8
GRID_SPACING_RATIO = 0.03       # 落点之间至少隔这么远（屏宽占比），约半个格子
# 同一个视野里最多点几下。四级候选（围洞 / 切格 / 切块 / 撒点）去重之后可能有二十来个，
# 每点一下约 2.7s；实测真空格最晚在第 5 下命中，留一倍余量。
MAX_TILE_CLICKS_PER_VIEW = 10
# 缩放 / 平移之后的新视野里只点前几个（多数是和上一个视野重叠的同一批地方）
MOVED_VIEW_CLICKS = 6
# 一次找地格**总共**最多点几下。每换一个视野「已试过」就清空，不设总数的话
# 1 个初始视野 + 3 档缩放 + 12 步平移最多能点 160 下（约 9 分钟）。
TILE_CLICK_BUDGET = 24
# 挪了一圈（或者不挪）还没点中：最后这个视野连看几帧，把冒出来、还没点过的 hole / blob / tile 候选点一遍再「先选人」（I-47）。
# 10/12 01:58 SF-1-2：右边那格空格一直在画面上，hole / blob 每帧时有时无（门槛边上），22 下里没有一下点到它；
# 唯一认出它的那一帧（01:58:23）只拿去画了 no_tile 标注图。挪镜头时「点过的」清掉了，最后这个视野一次都没试过
FINAL_VIEW_FRAMES = 3
FINAL_VIEW_GAP = 0.3
FINAL_VIEW_CLICKS = 4
FINAL_VIEW_KINDS = ('hole', 'blob', 'tile')
# 选好人之后重新找地格（闪烁 / 换格）总共最多点几下
RESELECT_CLICK_BUDGET = 16
# 连着几个部署界面滚轮两个方向都没反应，就这次运行里不再试缩放
WHEEL_GIVE_UP = 1                 # 10/14 巡检：所有日志里「滚轮 ±4 格」83 次全没反应，每个部署界面白花约 9 秒（I-60）
# 同一片里几格的内接半径非常接近，**排序靠不住**（图12 里空格只排第 2）。
# 所以不指望一次点对：选完人按下「部署人形」看顶部计数涨没涨，没涨就换一格重来。
DEPLOY_RETRY_TILES = 3
DEPLOY_COUNT_REREADS = 3          # 进部署界面读不到「部署人形 x/y」时再看几眼（每眼隔 0.5 秒）

# ---- 阳光地图上的空格（2026-09-26 离线诊断）----
#
# 上面那套 HSV 掩码卡了 S>=60。晴天地图（广场那张）上**空格内部几乎是白的**：
# 实测中位饱和度 S=2 / 23 / 57，掩码只覆盖到空格的 0% / 6% / 43%；
# 反倒是被天光照蓝的阴影（H104 S81）有 52~73% 进了掩码 —— 候选和撒点全落在阴影上。
# 「发白 = 空格」也不成立：站了人的格子在阳光下同样发白（S 31~62），在阴影里反而很饱和。
#
# 跨光照不变的是「**比周围更蓝**」：阴影和雾是整片均匀地偏蓝，地格覆盖层是局部偏蓝。
# 所以用 B-R 减去大窗口中值（top-hat）：T = (B-R) - median_{0.08W}(B-R)，把均匀的偏蓝减掉。
# 在 21 张手工标注的截图上（13 张有未选中的空格）：
#     空格排第 1     现在 8/13  ->  10/13
#     空格在候选里   现在 10/13 ->  12/13
# 候选按四级给出：① 被格子亮边框围住的「洞」（空格内部干净，被边框围成闭合区域；
# 站了人的格子被人形打断，围不成）② 原来的切格 ③ top-hat 上切出来的块 ④ 在新掩码上撒点。
TOPHAT_K = 0.08                       # 中值窗口，屏宽占比（在 1/4 分辨率上算）
# 外圈色相上界 125（以前 113）：格子覆盖层是半透明的，铺在红色 / 暖色地面上时外圈的蓝被底色拉向紫，
# 实测 112~128（10/14 DF-1-3 红落叶、10/13 22:42 红土、9/29 SA-1-8 红地毯），113 把它们全挡掉了（I-61）。
# 122 张现场图上放宽后多出来的 hole / blob 全落在真空格上。下界不动：它是用来排除青色「V」格的。
HOLE_PARAMS = dict(t_edge=22, close_px=7, min_r=0.006, max_r=0.05, min_solidity=0.85, min_rect=0.65,
                   ring_h=(99, 125), ring_min_v=110, occ_max=0.06)
BLOB_PARAMS = dict(t_low=16, close_px=15, ridge_pct_T=70, ridge_min_T=22, ridge_pct_V=92,
                   min_r=0.006, max_r=0.05, min_rect=0.6, min_sol=0.8,
                   rim_h=(101, 125),   # 青色带「V」标记的格子 H=93~100，不是可部署地格，排除
                   occ_max=0.08, limit=6)
HUE_MISS_MARGIN = 12                  # 色相只差这么多没过的 hole / blob 记下来（存现场图时写进 json，I-61 取证）
GRID_PARAMS = dict(h=(101, 115), s_sat=60, v_sat=60, s_pale=12, v_pale=140, t_fill=6)

# ---- 右下角：点完地格之后它显示什么（2026-09-26 日志 + 用户截图核实）----
#   「部署人形」按钮                 -> 点中了能放人的空格
#   「请点击地格或拖拽以部署单位」     -> 没点中（点在地面上）
#   「撤回人形」/ 选完助战后「编入队伍」 -> 点到了**站着人的格子**，选中的是那个人形
#   什么都没读到                     -> 说不准
# 以前只认前两种，第三种落进「读不到」、又被当成「已选中」放行 —— 这是「放不上人」的头号原因：
# 9/26 12:49 之后 14 次确定失败里有 9 次是它。点击到判定耗时也证明它是持续状态而不是动画帧：
# 真/假两态 2.62~2.74s，这一态 3.34~3.53s（多了那次 0.6s 重读），17 次里没有一次重读后变成真。
OCCUPIED_RE = re.compile('撤回人形|编入队伍')
TILE_EMPTY = 'empty'          # 选中了空地格，可以部署
TILE_HINT = 'hint'            # 没选中
TILE_OCCUPIED = 'occupied'    # 选中的是一个已经上场的人形
TILE_UNKNOWN = 'unknown'      # 读不出来
TILE_STATE_NAMES = {TILE_EMPTY: '空地格', TILE_HINT: '没点中', TILE_OCCUPIED: '有人的格子',
                    TILE_UNKNOWN: '读不出'}

# 部署失败时自动存图，下次失败就有现场可看（相对工作目录，独立版在 standalone/ 下）
DEBUG_TILE_DIR = 'debug_tiles'
DEBUG_TILE_KEEP = 120
# 按下「部署人形」之后在这些时刻（秒）各抓一帧；没放上时 OCR 画面中间找提示字、存下来（I-62 取证：
# 10/14 11:57 FZ-1-1 按完格子被取消，原因分不出是点击时序还是游戏弹了提示）
PRESS_FRAME_TIMES = (0.15, 0.4, 0.8, 1.3, 2.0)
PRESS_TOAST_BOX = (0.2, 0.3, 0.8, 0.7)
# 每张现场图旁边一份同名 .json（见 _save_case_note）：光有图和一句日志，常常分不清当时它以为自己在哪、
# 前面做了什么（9/29 SZ-1-x「点了没打开」到现在都查不出来）。记最近这么多条动作 / 日志
CASE_TRAIL = 30


def _is_chapter_image(name):
    # 和 stage_select.is_chapter_image 一样（两个模块互不 import，测试里要能单独加载）
    stem = os.path.splitext(os.path.basename(name))[0]
    return stem.endswith('_chapter') or stem.endswith('_chapter_ann')


def _is_page_image(name):
    """实时辅助的「界面记录」图（RealtimeAssistTask._page_log）：单独限量，见 prune_page_images。"""
    return os.path.splitext(os.path.basename(name))[0].endswith('_page')


def prune_page_images(root, keep):
    """界面记录的图只留最近 keep 张（.json 跟着图删），不碰别的现场图。"""
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if _is_page_image(f))
    images = [f for f in files if not f.endswith('.json')]
    for old in images[:-keep]:
        os.remove(old)
        note = os.path.splitext(old)[0] + '.json'
        if os.path.exists(note):
            os.remove(note)


def prune_debug_dir(root, keep):
    """debug_tiles 只留最近 keep 张图（.json 记录不算数，跟着自己的图一起删；图没了的记录也删）。

    「没有可推的」存的章节地图不算在 keep 里、也不在这里删（stage_select 单独限量）——
    9/29 18:33 起它一会儿存了上百张，按总数删的话把部署失败这些现场图全挤掉了。
    界面记录（_page）同理，单独限量（prune_page_images）。"""
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs)
    images = [f for f in files if not f.endswith('.json')]
    chapter = [f for f in images if _is_chapter_image(f) or _is_page_image(f)]
    images = [f for f in images if not (_is_chapter_image(f) or _is_page_image(f))]
    for old in images[:-keep]:
        os.remove(old)
    alive = set(os.path.splitext(f)[0] for f in images[-keep:] + chapter)
    for f in files:
        if f.endswith('.json') and os.path.splitext(f)[0] not in alive:
            os.remove(f)

# ---- 用户给的两条额外判据（2026-09-25）----
#
# 判据一：**可部署的地格一定贴着已经上场的人形**（用户原话：田字格、日字格）。
# 已部署人形头顶有个数字牌 1/2/3…，OCR 地图区域里的单个数字就能拿到锚点。
# 认不全没关系 —— 实测 3 个牌子常常只认出 1 个，有一个锚点就够把远处景物甩掉了。
# 这条是为了治「雾蒙蒙的蓝灰色地图」：图13 里景物混进了蓝色掩码，蒙出 3 个假峰
# （r=84/78/69）全都排在真空格（r=45）前面，靠编号牌一锚就只剩正确的那个。
UNIT_BADGE_RE = re.compile(r'^[1-9]$')
UNIT_BADGE_MIN_CONF = 0.6
TILE_NEAR_UNIT_RATIO = 0.11        # 候选离某个编号牌不能超过这么远（屏宽占比）
#
# 判据二：**选中助战之后，能放人的空地格会闪烁**，站了人的不会。
# 这是唯一不受光照/地图影响的判据 —— 静态画面里两者的蓝色几乎一模一样。
# 所以放在「选完人、第一次部署没成」之后用，那时候人已经选好，正好在闪。
BLINK_FRAMES = 8
BLINK_INTERVAL = 0.15              # 抓帧间隔（秒），总共约 1.2s
BLINK_MIN_SWING = 6.0              # 亮度摆幅绝对下限（0~255）
BLINK_RATIO = 1.8                  # 还要比第二名的摆幅大这么多倍才算数
# 第一名没有明显大过第二名、但前几名一起明显比其余的闪（前 k 名的最后一个也比第 k+1 名大 BLINK_RATIO 倍）：
# 按闪烁幅度把这几个排在最前面一起试。9/29 19:26 ZA-1-5：37.1 / 30.4 / 其余 ≤6.5，以前判成「判不出来」，
# 先试了整图闪烁那块、再按原顺序点，第 6 下才点中空格，中间 3 下点到了自己人站的格子
BLINK_GROUP_MAX = 3
#
# 判据二的加强版：**不靠候选、在整张地图上找「在闪」的区域**（2026-09-29 章节最后的首领关 SA-1-8）。
# 上面那条只在已有候选点上比亮度，候选一个都没有就无从比起，而那一关恰好两种情况都切不出候选：
# - 14:51 地格铺在红地毯上，半透明的蓝透出来是**粉紫色**（H≈149，不是蓝），颜色判据全部落空；
# - 14:49 地格贴着屏幕上下边，被顶栏和「作战开始」按钮挡得只剩一角（最后靠点中按钮边上露出来的
#   那一角才放上人）。
# 闪烁不看颜色、露一角也在闪。选人之前先录一段「本来就在动的」（首领待机、火光、人形待机），
# 选人之后只看多出来的那部分，挑成块、均匀地闪的区域。
BLINK_MAP_SCALE = 4                # 缩到 1/4 再算，一帧几毫秒
# 逐像素摆幅的门槛（固定值）。选中助战后空格实测 28.7（9/25）、51.4 / 91.9（9/29）。
# 不按「最亮处的几成」定：首领动作、火光的摆幅常常上百，按比例的话真闪烁反而够不上门槛
BLINK_MAP_MIN_SWING = 12.0
# 选人后的摆幅要超过选人前的 2 倍才算多出来的：火光这种随机闪动两段各不相同，1 倍减不干净；
# 底图还往四周扩一格，吃掉动作的小幅位移。静止的地格选人前摆幅只有 2~4（压缩噪声），扣 2 倍无妨
BLINK_BASE_WEIGHT = 2.0
BLINK_MAP_FULL_AREA = 150          # 1/4 图上一整格的大概面积：比它小的（火苗、零星亮点）排序时打折
BLINK_MAP_MIN_AREA = 12            # 1/4 图上的最小面积（像素）：按钮边上露出来的一角也要留住
BLINK_MAP_MAX_AREA_RATIO = 0.05    # 比这还大的（占地图区域）是首领 / 大片特效，不是一格
BLINK_MAP_MIN_FILL = 0.3           # 面积 / 外接框：人形待机、火苗这种细碎的边缘闪动填不满
# 一致性：整块区域的平均亮度曲线的方差 / 各像素自己曲线方差的平均。地格闪烁是整块一起亮一起暗 ≈ 1；
# 首领、人形动作的边缘各像素变化方向各不相同，平均下来互相抵消 ≈ 0.1~0.3
BLINK_MAP_MIN_COHERENCE = 0.5
BLINK_MAP_TOP = 4
# 「作战开始」和它左边的刷新按钮会呼吸发光：只在按钮那一条里按颜色抠掉（亮黄，H≈18~21、S>160、
# V>210，9/29 三张现场图实测），再外扩一点。整屏按颜色抠会把黄色地面标线、左下角详情卡也抠掉
BLINK_BUTTON_BAND = (0.30, 0.84, 0.66, 1.0)     # x0, y0, x1, y1（屏幕占比）
BLINK_BUTTON_HSV = ((12, 120, 150), (40, 255, 255))

# 从「作战效能」数字中心往上多少（屏幕高度占比）能落到卡片立绘中心。
CARD_PORTRAIT_DY = 0.065

# 好友助战卡片面板的上边界（与 support_panel_box 同源）：点击点绝不能越过它，
# 否则会点到上面的元素筛选条 / 页签上去。
SUPPORT_PANEL_TOP = 0.12

# 又一个特殊返回值：人选好了，但怎么都没能放到地格上（见 _on_deploy_failed）。
DEPLOY_FAILED = 'deploy_failed'
# 右侧人形面板是收起的，点了几次「展开人形列表」都没展开 —— 和「这个界面没有助战页签」不同，
# 后者可以照常开打，前者硬开打就是少一个人。
PANEL_STUCK = 'panel_stuck'
# 点地格的时候部署数自己涨了（游戏像是直接放上了自己的人形），阵容已经被打乱。
LINEUP_CHANGED = 'lineup_changed'

# pick_friend_support 的特殊返回值：当前界面根本没有「好友助战」页签。
# 这跟「列表里没有她」是两回事 —— 退出重进也变不出一个页签来，不该触发重进。
NO_TAB = 'no_tab'
# 另一个特殊返回值：地格没选中。**没选中地格就别去选助战** —— 人放不上去，
# 选了也是白选，还白花十几秒翻助战列表。和「列表里没有她」也是两回事：
# 退出重进换一批好友并不能让地格变得选得中，所以这种情况不该触发重进、也不找替补。
NO_TILE = 'no_tile'
# 又一个：点了「好友助战」没切过去（亮着的还是「我的人形」），或者借到一半发现停在「我的人形」页上（I-46）。
# 「我的人形」页上，元素筛选条那个高度是「使用整备室配置」下拉框 —— 10/12 00:17 就这样点进去、换掉了阵容预设。
# 一下都不再点，按「放不上」停手交给你
TAB_STUCK = 'tab_stuck'
# 点了「好友助战」之后最多等这么久看它亮起来；还亮着「我的人形」就再点一次，最多点这么多次
SUPPORT_TAB_SWITCH_WAIT = 2.0
SUPPORT_TAB_CLICK_MAX = 2

# 「好友助战」页签下方的元素筛选条：6 个图标的中心（屏幕相对坐标，实测自 3175x1800 客户区）
ELEMENT_TAB_Y = 0.101
ELEMENT_TAB_X = (0.790, 0.828, 0.866, 0.907, 0.945, 0.981)

SUPPORT_TAB = '好友助战'
# 「友」偶尔被 OCR 认成别的字：10/1 13:40 读成「改造好发助战」（并了左边的字），当成没有助战页签、没借人就开打了。
# 第二个字放宽成任意一个字；只读到「助战」两个字的不算
SUPPORT_TAB_RE = re.compile('好.助战')
# 右侧人形面板收起时，右上角只剩这个按钮、没有「我的人形 / 好友助战」页签
EXPAND_PANEL_RE = re.compile('展开人形列表')
# 面板**展开了**的正面证据：「我的人形」页签。点图标能展开之后才用得上 —— 展开了却没有「好友助战」
# （这一关本来就不给借助战）要算「没有助战页签」照常开打，不能算 PANEL_STUCK 停手等人。
# 9/25~9/26 日志里读到「我的人形」的 283 帧，同一帧 283/283 也读到了「好友助战」（反过来也是），
# 所以「有我的人形、没好友助战」基本不会是 OCR 漏读。
MY_UNITS_TAB_RE = re.compile('我的人形')
# 「展开人形列表」这几个字只是标签，**真正的按钮是它右边那个深色方块图标**（方块里画着 ← 箭头）。
# 以前点的是 OCR 文字框中心，3/3 次都没展开（9/27 00:06、00:43；9/29 11:12，每次连点 3 下之后
# PANEL_STUCK 停手，9/29 那次用户手动接管了 12 分钟）。
# 在三张现场图上量的（3200x1800，debug_tiles/2026092{7,9}/*_panel_stuck.jpg）：
#     OCR 文字框（实跑 onnxocr）         右边缘 x=3033~3035，高 h=40~41px，中心 y≈87
#     图标方块（底色 BGR≈(28,28,16)）     x=3056~3156，y=40~140，中心 (3106, 90)
# 折成文字框高度 h 的倍数：方块横跨「右边缘 + 0.51h ~ 2.98h」，中心在 +1.74h ~ +1.82h；
# 竖直方向方块中心只比文字中心低 2~4px，直接用文字中心。
# **按字高算，不按屏宽算**：同一张图缩到 2560 / 1920 宽重跑 OCR，算出来的点照样在方块中心附近。
# 第 1 次点方块中心；第 2 次往左挪一点（离方块左边缘还有 ~0.9h）；第 3 次退回点文字本身兜底。
EXPAND_ICON_DX = (1.75, 1.4)
EXPAND_ICON_MARGIN = 10         # 点击点离屏幕右边缘至少留这么多像素
# 最多点几下：每个图标偏移一次，再加一次点文字兜底
EXPAND_MAX_CLICKS = len(EXPAND_ICON_DX) + 1
# 「展开人形列表」6 个字本身的宽度 / 字高：三张现场图实跑 onnxocr 是 207/41、201/40、208/41（5.03~5.07），
# 缩到 2560 / 1920 宽重跑是 5.00~5.17。OCR 把图标里的图形并进框时，用它从框左边推回文字的右边缘。
EXPAND_TEXT_W = 5.05

# 当前页签底下那条橙色下划线（HSV，OpenCV 口径）。实测只出现在选中的那个页签下方，
# y 落在页签文字框底边往下约 0.005~0.013 屏高处。
# 用它判断「好友助战」是不是已经是当前页签 —— 已经是却又点一次，会把整个助战面板收回去。
TAB_UNDERLINE_HSV_LOWER = (5, 150, 180)
TAB_UNDERLINE_HSV_UPPER = (25, 255, 255)
TAB_UNDERLINE_BAND = 0.020      # 从页签文字框底边往下取多高的带来找下划线
TAB_UNDERLINE_MIN_RATIO = 0.05  # 带内橙色像素占比超过它就算选中

# 部署界面左上角的 ☰，点开是「关卡信息」页（内含「退出作战 / 继续作战」）
STAGE_MENU_BTN = (0.025, 0.041)

# 选好地格和人之后，右下角这个按钮才真正把人放上场
DEPLOY_BTN = '部署人形'
# 还没选中地格时，右下角显示的提示。ok 的 find_boxes 传字符串是**精确相等**，
# 要子串匹配必须传编译好的正则。
TILE_HINT_RE = re.compile('点击地格|拖拽以部署')

# 挪视角时鼠标落在哪：地图左中部，避开右侧助战面板(x>0.74)和底部那排地格/按钮
VIEW_PROBE_POINT = (0.35, 0.55)

# ---- 镜头怎么平移 ----
# 独立版用的是 ok 的 Genshin 交互（后台 PostMessage）。逐个方法读过源码、对过日志之后：
#   - 滚轮：genshin.do_scroll 先 `import mouse`，ok-gf2 的 python 里没装这个包，
#     异常被 operate() 吞掉 —— **一次都没发出去过**（助战列表翻页也一直是坏的）。
#     所以这里自己发 WM_MOUSEWHEEL（_wheel），发完还要验证画面真的变了。
#   - 拖拽：mouse_down / move / mouse_up 是三次独立的 operate()，每次之后都 WA_INACTIVE
#     并把光标放回原处，move 也不带按键标志 —— 在后台 9 次全部失败。
#     所以这里把一整个拖拽放进**一次** operate() 里，每步都带 MK_RBUTTON 并挪真光标。
#     **左键拖拽不试**：右下角写着「拖拽以部署单位」，左键拖拽会把人拖上场。
#   - WASD：唯一在后台有实证能动的（'d' 13:44 / 15:40，'s' 09:06）。但 genshin 的
#     send_key_up 只 deactivate、**从不发 WM_KEYUP**，按住多久全凭运气。
#     所以这里在一次 operate() 里自己发 KEYDOWN / 等待 / KEYUP（_hold_key）。
#   - 方向键（缺扩展键标志，会被当成小键盘）、Q/E、鼠标推到屏幕边缘：都没动过，去掉。
# 以前「WASD 只能上下、不能左右」是探测的毛病：每个轴只试正方向，
# 刚好顶在地图边界上就判成「这个轴不能动」，还把这个否定结论记一整次运行。
# 现在两个方向都试，**只记住成功的方式**，失败不缓存。
PAN_METHOD_NAMES = {'keys_wasd': 'WASD 键', 'right_drag': '右键拖拽'}
# 每种按键方案在两个轴上分别用哪两个键：(负方向, 正方向)
PAN_KEY_SETS = {
    'keys_wasd': {'x': ('a', 'd'), 'y': ('w', 's')},
}
PAN_METHODS_X = ('keys_wasd', 'right_drag')
PAN_METHODS_Y = ('keys_wasd', 'right_drag')
# 拖拽的**中点**候选。屏幕正中优先 —— 实测从偏上的位置拖不动。
PAN_CENTERS = ((0.50, 0.50), (0.40, 0.42), (0.55, 0.60))
# 拖拽以中点对称展开，两端都离右侧助战面板(x>0.74)远远的，不会变成滚动助战列表。
PAN_BOUNDS_X = (0.06, 0.72)
PAN_BOUNDS_Y = (0.08, 0.76)
PAN_STEP = 0.30                 # 每步挪屏幕宽的 30%，留足重叠
PAN_STEPS_PER_DIR = 3           # 一个方向最多连挪 3 步（累计 0.9 屏宽）
PAN_DRAG_STEPS = 16             # 一次拖拽分几小步移动鼠标（一步跳到底游戏收不到）
PAN_DRAG_STEP_SLEEP = 0.02
PAN_KEY_HOLD = 0.45             # 方向键按住多久算「一步」

# 位移验证：拿地图区域做**相位相关**，直接得到「整幅图移了多少像素、往哪边」。
# 只看两张截图有没有差别是不够的 —— 人物待机动画、火光特效都会让画面变，
# 但它们不会让整幅图一致地平移，相位相关的置信度就上不去。
PAN_MIN_SHIFT_RATIO = 0.02      # 位移要达到缩略图宽的这个比例才算真动了
PAN_MIN_RESPONSE = 0.02         # 相位相关置信度下限

# 挪视角找地格一整套下来要几十秒。连着这么多次都没成，就别再每场战斗都白费这个时间了，
# 后面直接跳过地格、只管选人。
TILE_SEARCH_GIVE_UP = 2

# 动画播放时先点一下屏幕，「跳过」按钮才会冒出来。点中央：那里在过场里通常没有可点的东西。
SKIP_REVEAL_POINT = (0.5, 0.5)
# 父类 skip_dialogs 分段跑，每段之间补一次上面那下点击
SKIP_CHUNK_SECONDS = 8

# 退出重进之后等部署界面（_wait_deploy_screen）：只点认得出的，不盲点右上角
START_BATTLE_TEXT = '作战开始'
SKIP_TEXT = '跳过'
SKIP_TOP_RATIO = 0.12           # 「跳过」永远在最上面 y≤80px（9/25~9/29 日志 354 次点击）
NO_ALERT_TEXT = '今日不再提示'
REVEAL_MAX_BOXES = 8            # 字比这多的画面是菜单，不盲点
REVEAL_TAP_INTERVAL = 2.0
REVEAL_TAP_LIMIT = 12


def _makelong(low, high):
    """WPARAM/LPARAM 拼装。win32api.MAKELONG 对负数（向下滚、副屏坐标）不友好，自己拼。"""
    return ((int(high) & 0xFFFF) << 16) | (int(low) & 0xFFFF)


# ---------------------------------------------------------------- 找空地格：纯图像函数
# 放在类外面，离线回放截图时可以直接调用（不需要 OCR / 窗口）。

# 左下角详情卡（选中了人形 / 助战才有）。以前不管有没有卡都把左下角（x<0.30W、y>0.78H）和整条底边（y>0.93H）抹掉：
# 10/11 23:55 VA-1-5 唯一的空格在屏幕底边左侧（x 0.18~0.29W、y 0.92H 往下），四级候选一个都没出，镜头又挪不动，
# 放不上助战、停手等你（I-40）。现在按这一帧有没有卡来定：卡底那条蓝血条在 0.944H、0.03~0.21W 整行都是这个蓝 ——
# 10/12 量了 13 张有卡的图都是 1.000，34 张没卡的最高 0.108（就是 VA-1-5 底边那个蓝格）
CARD_BAR_ROWS = (0.92, 0.97)
CARD_BAR_X = (0.03, 0.21)
CARD_BAR_HSV = ((95, 110, 170), (112, 255, 255))
CARD_BAR_MIN = 0.6
BOTTOM_FREE_X = 0.33            # 没卡时底部按钮条只抹这条线右边（旋转按钮 0.35W 起、「作战开始」0.40W 起）


# 左下角详情卡被点成「展开」的（I-58，10/14 09:12 BF-1-2：找空格时点到了卡上）：攻击 / 防御 / 技能 / 武器铺开，
# 「支援 OTs-14 Lv.60」那一行挪到半屏高度，平时读名字的那一条（0.68~0.94H）里只剩数值 -> 每张卡都读不出名字，
# 以为名单里没有威玛西娜，退去借最强的。展开时名字那一行在这里（091457 实测 0.44~0.50H）
DETAIL_EXPANDED_RE = re.compile('暴击|行动力|稳态减伤|防御')
DETAIL_EXPANDED_NAME = (0.0, 0.40, 0.30, 0.52)
DETAIL_EXPANDED_AREA = (0.25, 0.38)  # 展开的卡占 x < 0.25W、y > 0.38H
CARD_AREA = (0.235, 0.78)           # 收起的卡占 x < 0.235W、y > 0.78H（实测右边框在 0.232W；_roi_mask 屏蔽得更宽，那是给生成候选用的）
# 展开的卡也有一条蓝血条，在半屏高度（091457 / 095802 两张展开的都是 1.000，别的现场图最高 0.076，复查实测）
EXPANDED_BAR_ROWS = (0.53, 0.61)
EXPANDED_BAR_X = (0.03, 0.17)
# 退而借最强的那张放不上去（人数不涨）：多半是你队里有同一个角色（10/14 OTs-14），换下一张再试，最多试这么多张
FALLBACK_CARD_TRIES = 3
DETAIL_NAME_SKIP_RE = re.compile(r'^(?:支援|Lv\.?\d*|\[?作战效能.*|[\d\-/.:%：\[\]]+|.)$')


_GLUED_LV_RE = re.compile(r'[Ll][Vv]\s*[.。]?\s*\d*')


def detail_name(text):
    """详情卡文字（read_selected_unit_name）里的角色名：「支援 OTs-14 Lv.60 1865-443 [作战效能：25995」-> 'OTs-14'；读不出 None。
    取「Lv」前面那个（长名字常和等级粘在一起：「纳美西丝·衍光Lv.60」-> 「纳美西丝·衍光」）；读进来的地图上的字（等级角标「3」）跳过。"""
    toks = (text or '').split()
    for i, t in enumerate(toks):
        m = _GLUED_LV_RE.search(t)
        if not m:
            continue
        head = t[:m.start()].strip()
        if head and not DETAIL_NAME_SKIP_RE.match(head):
            return head
        for p in reversed(toks[:i]):
            if not DETAIL_NAME_SKIP_RE.match(p):
                return p
        break
    for t in toks:
        if not DETAIL_NAME_SKIP_RE.match(t):
            return t
    return None


def same_unit(a, b):
    """两个详情卡上的名字是不是同一个角色（纳美西丝 / 纳美西丝·衍光、读错一个字都算）。"""
    try:
        from unit_name import same_character
    except ImportError:                 # 打包漏了它：按字面比
        return bool(a) and a == b
    return same_character(a, b)


def detail_card_expanded(frame):
    """左下角详情卡是不是被点成了「展开」的（看半屏高度那条蓝血条）。读不了按「没展开」算（不多跳过候选）。"""
    try:
        h, w = frame.shape[:2]
        y0, y1 = int(EXPANDED_BAR_ROWS[0] * h), int(EXPANDED_BAR_ROWS[1] * h)
        x0, x1 = int(EXPANDED_BAR_X[0] * w), int(EXPANDED_BAR_X[1] * w)
        if y1 <= y0 or x1 <= x0 or frame.ndim != 3:
            return False
        hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1, :3]), cv2.COLOR_BGR2HSV)
        lo, hi = CARD_BAR_HSV
        row = (cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0).mean(axis=1)
        return float(row.max()) >= CARD_BAR_MIN
    except (cv2.error, ValueError, AttributeError):
        return False


def detail_card_visible(frame):
    """左下角详情卡在不在（看卡底那条蓝血条）。读不了按「在」算 —— 那样屏蔽和以前一样，不会多出候选。"""
    try:
        h, w = frame.shape[:2]
        y0, y1 = int(CARD_BAR_ROWS[0] * h), int(CARD_BAR_ROWS[1] * h)
        x0, x1 = int(CARD_BAR_X[0] * w), int(CARD_BAR_X[1] * w)
        if y1 <= y0 or x1 <= x0 or frame.ndim != 3:
            return True
        hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1, :3]), cv2.COLOR_BGR2HSV)
        lo, hi = CARD_BAR_HSV
        row = (cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0).mean(axis=1)
        return float(row.max()) >= CARD_BAR_MIN
    except (cv2.error, ValueError, AttributeError):
        return True


def _roi_mask(h, w, card=True):
    """只在地图区域里找：去掉顶部状态栏、右侧面板、底部按钮条，有详情卡时再去掉左下角那块（见 CARD_BAR_ROWS 上面）。"""
    m = np.full((h, w), 255, np.uint8)
    m[:int(0.10 * h), :] = 0
    m[:, int(0.74 * w):] = 0
    if card:
        m[int(0.93 * h):, :] = 0
        m[int(0.78 * h):, :int(0.30 * w)] = 0
    else:
        m[int(0.93 * h):, int(BOTTOM_FREE_X * w):] = 0
    return m


def _gray_stack(frames, scale=BLINK_MAP_SCALE):
    """一串帧缩小 scale 倍的灰度图，叠成 (帧数, 高, 宽) 的 float32。"""
    small = []
    for f in frames:
        h, w = f.shape[:2]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        small.append(cv2.resize(g, (w // scale, h // scale), interpolation=cv2.INTER_AREA).astype(np.float32))
    return np.stack(small)


def swing_map(frames, scale=BLINK_MAP_SCALE):
    """一串帧逐像素的亮度摆幅（最大 - 最小），在缩小 scale 倍的灰度图上算。"""
    st = _gray_stack(frames, scale)
    return st.max(0) - st.min(0)


def coherence(series):
    """(帧数, 像素数) 的亮度曲线：整块平均曲线的方差 / 各像素曲线方差的平均。一起亮一起暗 ≈ 1。"""
    x = series - series.mean(axis=0, keepdims=True)
    own = float((x * x).mean())
    if own <= 1e-6:
        return 0.0
    s = x.mean(axis=1)
    return float((s * s).mean()) / own


def blink_mask(frame, scale=BLINK_MAP_SCALE):
    """找闪烁的范围（缩小后的尺寸）：去掉右侧人形面板、顶部「部署人形」条、左下角详情卡和底部按钮。

    和 _roi_mask 不同，不整条切掉顶部和底部 —— 静态的界面本来就不闪，用不着切；
    整条切掉的话，贴着边、只露一角的地格也跟着没了。
    """
    h, w = frame.shape[:2]
    hs, ws = h // scale, w // scale
    m = np.full((hs, ws), 255, np.uint8)
    m[:, int(0.74 * ws):] = 0                                  # 右侧人形面板（选中的卡框会动）
    m[:int(0.068 * hs), int(0.36 * ws):int(0.64 * ws)] = 0     # 顶部「部署人形 / 作战效能」条
    m[int(0.76 * hs):, :int(0.30 * ws)] = 0                    # 左下角详情卡（选中人之后才出现）
    x0, y0, x1, y1 = BLINK_BUTTON_BAND
    bx0, by0, bx1, by1 = int(x0 * ws), int(y0 * hs), int(x1 * ws), hs
    small = cv2.resize(frame[int(y0 * h):, int(x0 * w):int(x1 * w)], (bx1 - bx0, by1 - by0),
                       interpolation=cv2.INTER_AREA)
    lo, hi = BLINK_BUTTON_HSV
    btn = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), np.array(lo, np.uint8), np.array(hi, np.uint8))
    btn = cv2.dilate(btn, np.ones((3, 3), np.uint8), iterations=2)
    band = m[by0:by1, bx0:bx1]
    band[btn > 0] = 0
    return m


def blink_regions(frames, base=None, scale=BLINK_MAP_SCALE):
    """在整张地图上找「成块、均匀地闪」的区域。

    :param base: 选人之前录的 swing_map（本来就在动的东西），有就先减掉
    :return: [(平均摆幅, 中心x, 中心y, 等效半径), ...]（全分辨率坐标），按摆幅 × 饱满度从大到小
    """
    if not frames:
        return []
    stack = _gray_stack(frames, scale)
    sw = stack.max(0) - stack.min(0)
    if base is not None and getattr(base, 'shape', None) == sw.shape:
        base = cv2.dilate(base.astype(np.float32), np.ones((3, 3), np.uint8))
        sw = np.clip(sw - BLINK_BASE_WEIGHT * base, 0, None)
    mask = blink_mask(frames[-1], scale)
    sw[mask == 0] = 0
    if not (mask > 0).any():
        return []
    b = (sw >= BLINK_MAP_MIN_SWING).astype(np.uint8) * 255
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    max_area = BLINK_MAP_MAX_AREA_RATIO * float((mask > 0).sum())
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < BLINK_MAP_MIN_AREA or area > max_area:
            continue
        fill = area / float(bw * bh)
        if fill < BLINK_MAP_MIN_FILL:
            continue
        comp = (lab == i)
        mean = float(sw[comp].mean())
        if mean < BLINK_MAP_MIN_SWING:
            continue
        if coherence(stack[:, comp]) < BLINK_MAP_MIN_COHERENCE:
            continue                    # 动作边缘：各像素各变各的，不是整块一起闪
        # 落点取离区域边界最远的那一点（凹形 / 细长的一角也点在里面，不会点到外面去）。
        # 相邻两格（田字格、日字格）常连成一片：挖掉第一个落点周围，还有一个差不多深的点就再给一个
        pad = np.pad(comp.astype(np.uint8), 1)
        dist = cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
        score = mean * min(1.0, fill / 0.5) * min(1.0, area / float(BLINK_MAP_FULL_AREA)) ** 0.75
        r = max(float(np.sqrt(area / np.pi)), 2.0) * scale
        first = None
        for k in range(2):
            cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            peak = float(dist[cy, cx])
            if peak <= 0 or (first is not None and peak < 0.6 * first):
                break
            first = peak if first is None else first
            out.append((score * (1.0 if k == 0 else 0.99), mean,
                        int((cx + 0.5) * scale), int((cy + 0.5) * scale), r / (1 if k == 0 else 2)))
            cv2.circle(dist, (int(cx), int(cy)), int(max(2, 1.6 * peak)), 0, -1)
    out.sort(key=lambda t: -t[0])
    return [(mean, cx, cy, r) for _, mean, cx, cy, r in out]


def blueness_tophat(frame):
    """T = (B-R) - 大窗口中值(B-R)：只留「比周围更蓝」的部分，把均匀偏蓝的阴影和雾减掉。"""
    f16 = frame.astype(np.int16)
    br = np.clip(f16[:, :, 0] - f16[:, :, 2], 0, 255).astype(np.uint8)
    h, w = br.shape
    small = cv2.resize(br, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    k = min(max(3, int(TOPHAT_K * w / 4)) | 1, 255)
    bg = cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.subtract(br, bg)


def _occupancy(frame, mask):
    """区域里有多少像素明显偏离中位色 —— 站了人的格子里有人形，这个比例高。"""
    px = frame[mask].astype(np.int16)
    if len(px) <= 50:
        return 1.0
    return float((np.abs(px - np.median(px, 0)).sum(1) > 90).mean())


def _solidity_rect_vertices(pad):
    """(实心度, 矩形度, 多边形顶点数)。"""
    cnts = cv2.findContours(pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return 0.0, 0.0, 0
    c = max(cnts, key=cv2.contourArea)
    ca = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    return ((ca / hull) if hull else 0.0), ((ca / (rw * rh)) if rw * rh else 0.0), \
        len(cv2.approxPolyDP(c, TILE_QUAD_EPS * cv2.arcLength(c, True), True))


def _piece_center(blob, pad, x, y, cent):
    """落点用质心；质心不在块内（凹形）时退回距离变换峰值。"""
    cx, cy = int(cent[0]), int(cent[1])
    if not blob[min(max(cy - y, 0), blob.shape[0] - 1), min(max(cx - x, 0), blob.shape[1] - 1)]:
        _, _, _, (px, py) = cv2.minMaxLoc(cv2.distanceTransform(pad, cv2.DIST_L2, 5))
        cx, cy = x + px - 2, y + py - 2
    return cx, cy


def _hue_miss(misses, kind, lim, hue, blob, pad, x, y, cent, ox=0, oy=0):
    """形状都过了、只差色相这一道的块记进 misses（离边界 HUE_MISS_MARGIN 以内；ox / oy 是局部坐标的偏移）。"""
    if misses is None or not (lim[0] - HUE_MISS_MARGIN <= hue <= lim[1] + HUE_MISS_MARGIN):
        return
    cx, cy = _piece_center(blob, pad, x, y, cent)
    side = '偏紫' if hue > lim[1] else '偏青'     # 超了上界（红 / 暖色地面）还是低于下界（青绿地面 / 青色 V 格）
    misses.append((kind, int(ox + cx), int(oy + cy), round(float(hue), 1), side))


def hole_candidates(frame, T, hsv, p=HOLE_PARAMS, misses=None):
    """① 被格子亮边框围成的闭合「洞」。空格内部干净、边框完整；站了人的格子被人形打断。

    :param misses: 给一个 list 就把「只差色相没过」的块记进去（取证用）
    :return: [(x, y, 内接半径, 说明)]，干净的在前
    """
    h, w = frame.shape[:2]
    E = (T >= p['t_edge']).astype(np.uint8) * 255
    E = cv2.morphologyEx(E, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['close_px'],) * 2))
    roi = _roi_mask(h, w, detail_card_visible(frame))
    free = cv2.bitwise_and(cv2.bitwise_not(E), roi)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(free, 4)
    edge = cv2.bitwise_and(cv2.dilate(cv2.bitwise_not(roi), np.ones((3, 3), np.uint8)), free)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = 255
    leak = set(np.unique(lab[(edge > 0) & (free > 0)]).tolist())   # 连到画面外的不是洞
    out = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if i in leak or area < np.pi * (p['min_r'] * w) ** 2 or area > 12 * (p['max_r'] * w) ** 2:
            continue
        blob = (lab[y:y + ch, x:x + cw] == i).astype(np.uint8) * 255
        pad = cv2.copyMakeBorder(blob, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
        r_in = float(cv2.distanceTransform(pad, cv2.DIST_L2, 5).max())
        if not (p['min_r'] * w <= r_in <= p['max_r'] * w):
            continue
        sol, rect, _ = _solidity_rect_vertices(pad)
        if sol < p['min_solidity'] or rect < p['min_rect']:
            continue
        full = np.zeros((h, w), np.uint8)
        full[y:y + ch, x:x + cw] = blob
        ring = (cv2.dilate(full, np.ones((9, 9), np.uint8)) > 0) & (E > 0)
        if ring.sum() < 20:
            continue
        ring_hue = float(np.median(hsv[:, :, 0][ring]))
        if np.median(hsv[:, :, 2][ring]) < p['ring_min_v']:
            continue
        if not (p['ring_h'][0] <= ring_hue <= p['ring_h'][1]):
            _hue_miss(misses, 'hole', p['ring_h'], ring_hue, blob, pad, x, y, cents[i])
            continue
        occ = _occupancy(frame, cv2.erode(full, np.ones((9, 9), np.uint8)) > 0)
        cx, cy = _piece_center(blob, pad, x, y, cents[i])
        out.append((occ > p['occ_max'], occ, -rect, cx, cy, r_in,
                    'hole r%.0f q%.2f occ%.2f H%.0f' % (r_in, rect, occ, ring_hue)))
    out.sort()
    return [(cx, cy, r, s) for *_, cx, cy, r, s in out]


def blob_candidates(frame, T, hsv, p=BLOB_PARAMS, misses=None):
    """③ 在 top-hat 上找偏蓝的块，再按亮边框切成一格一格（原切格法的 top-hat 版）。"""
    h, w = frame.shape[:2]
    V, H = hsv[:, :, 2], hsv[:, :, 0]
    L = (T >= p['t_low']).astype(np.uint8) * 255
    L = cv2.morphologyEx(L, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['close_px'],) * 2))
    L = cv2.bitwise_and(L, _roi_mask(h, w, detail_card_visible(frame)))
    Fm = np.zeros_like(L)
    cv2.drawContours(Fm, cv2.findContours(L, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, 255, -1)
    nb, blab, bst, _ = cv2.connectedComponentsWithStats(Fm, 8)
    out = []
    for bi in range(1, nb):
        if bst[bi][4] < 0.0001 * w * h:
            continue
        x0, y0, bw, bh, _ = bst[bi]
        sel = blab[y0:y0 + bh, x0:x0 + bw] == bi
        Tb, Vb, Hb = T[y0:y0 + bh, x0:x0 + bw], V[y0:y0 + bh, x0:x0 + bw], H[y0:y0 + bh, x0:x0 + bw]
        tthr = max(p['ridge_min_T'], float(np.percentile(Tb[sel], p['ridge_pct_T'])))
        vthr = float(np.percentile(Vb[sel], p['ridge_pct_V']))
        ridge = cv2.dilate((sel & ((Tb >= tthr) | (Vb >= vthr))).astype(np.uint8) * 255,
                           np.ones((5, 5), np.uint8))
        inner = cv2.morphologyEx(cv2.bitwise_and(sel.astype(np.uint8) * 255, cv2.bitwise_not(ridge)),
                                 cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(inner, 4)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < np.pi * (p['min_r'] * w) ** 2:
                continue
            blob = (lab[y:y + ch, x:x + cw] == i).astype(np.uint8) * 255
            pad = cv2.copyMakeBorder(blob, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
            r_in = float(cv2.distanceTransform(pad, cv2.DIST_L2, 5).max())
            if not (p['min_r'] * w <= r_in <= p['max_r'] * w):
                continue
            sol, rect, nv = _solidity_rect_vertices(pad)
            if rect < p['min_rect'] or sol < p['min_sol']:
                continue
            full = np.zeros((bh, bw), np.uint8)
            full[y:y + ch, x:x + cw] = blob
            rim = (cv2.dilate(full, np.ones((11, 11), np.uint8)) > 0) & (full == 0) & (Tb >= p['ridge_min_T'])
            rim_h = float(np.median(Hb[rim])) if rim.sum() > 20 else float(np.median(Hb[full > 0]))
            if not (p['rim_h'][0] <= rim_h <= p['rim_h'][1]):
                if misses is not None:
                    _hue_miss(misses, 'blob', p['rim_h'], rim_h, blob, pad, x, y, cents[i], x0, y0)
                continue
            occ = _occupancy(frame[y0:y0 + bh, x0:x0 + bw], cv2.erode(full, np.ones((5, 5), np.uint8)) > 0)
            cx, cy = _piece_center(blob, pad, x, y, cents[i])
            out.append((occ > p['occ_max'], nv != 4, -rect, x0 + cx, y0 + cy, r_in,
                        'blob r%.0f q%.2f occ%.2f H%.0f' % (r_in, rect, occ, rim_h)))
    out.sort()
    return [(cx, cy, r, s) for *_, cx, cy, r, s in out[:p['limit']]]


def grid_mask(frame, T, hsv, p=GRID_PARAMS):
    """④ 撒点用的掩码：饱和的蓝，或者**发白但偏蓝**的格子（晴天），且比周围更蓝（去掉阴影）。"""
    h, w = frame.shape[:2]
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((H >= p['h'][0]) & (H <= p['h'][1])
         & (((S >= p['s_sat']) & (V >= p['v_sat'])) | ((S >= p['s_pale']) & (V >= p['v_pale'])))
         & (T >= p['t_fill']))
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.bitwise_and(m, _roi_mask(h, w, detail_card_visible(frame)))


def scatter_points(mask, w, limit=GRID_FALLBACK_POINTS, spacing_ratio=GRID_SPACING_RATIO):
    """在掩码上用「距离变换 + 非极大抑制」撒点，离边界越远越靠前。

    **先补一圈 0 再算距离变换**：OpenCV 把图像边框当成前景，贴着左边缘的一片蓝，
    峰值会落在 x=0 上 —— 以前真点过窗口最左边（日志 (0,836)、(0,852)）。
    """
    pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    dist = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[1:-1, 1:-1].copy()
    spacing = max(int(spacing_ratio * w), 10)
    pts = []
    for _ in range(max(1, limit)):
        _, mx, _, (px, py) = cv2.minMaxLoc(dist)
        if mx < 8:
            break
        pts.append((int(px), int(py), float(mx)))
        cv2.circle(dist, (int(px), int(py)), spacing, 0, -1)
    return pts


def parse_names(names):
    """「可露凯, 琼玖」「可露凯、琼玖」或者列表 -> ['可露凯', '琼玖']（去空白、去重、保持顺序）。"""
    if not names:
        return []
    if isinstance(names, str):
        names = re.split(r'[,，、;；/|\s]+', names)
    out = []
    for n in names:
        n = (n or '').strip()
        if n and n not in out:
            out.append(n)
    return out


def expand_icon_point(text_box, screen_w, k=EXPAND_ICON_DX[0]):
    """「展开人形列表」的 OCR 文字框 -> 它右边那个图标按钮的点击点 (x, y)，整数像素。

    x = 文字框右边缘 + k × 文字框高度，y = 文字框中心；x 夹在屏幕内（≤ 屏宽 - EXPAND_ICON_MARGIN）。
    k 的依据见 EXPAND_ICON_DX。

    OCR 把图标里的 ←/…*（9/26、9/29 日志里单独读成过「←」「…*」）并进文字框时（「展开人形列表←」），
    整框右边缘已经伸进图标里了，再加 k×h 会点出方块外（按现场图的图形位置实算 x=3163~3190，方块只到 3156）。
    所以要先把右边缘推回到「展开人形列表」最后一个字：文字左边缘 + EXPAND_TEXT_W×字高（不受并进来的图形
    有多宽影响）。文字左边缘一般就是框左边；前面也并了别的字时按字数比例估（和好友助战页签的裁法一样）。
    """
    h = float(text_box.height)
    right = text_box.x + text_box.width
    name = text_box.name or ''
    m = EXPAND_PANEL_RE.search(name)
    if m and m.end() < len(name):
        left = text_box.x + text_box.width * m.start() / float(len(name))
        right = min(right, left + EXPAND_TEXT_W * h)
    x = right + k * h
    y = text_box.y + h / 2.0
    if screen_w:
        x = min(x, screen_w - EXPAND_ICON_MARGIN)
    return int(round(max(0.0, x))), int(round(max(0.0, y)))


class SupportStateError(Exception):
    """退出/重进把界面留在了未知状态 —— 继续跑只会更乱，直接中止任务。"""


class TileNotSelectedError(Exception):
    """怎么都没能选中空地格 —— **停手，交回给人**。

    不往下走的理由：助战放不上去，硬按「作战开始」就是少一个人开打，
    白送一局。这种情况自己解决不了（挪视角、撒点、重进都试过了），
    只能提醒用户手动接管。
    """


class UserTakeoverError(Exception):
    """借助战途中你接手了：动了键鼠（实时辅助的输入关口发现的），或者画面已经不是部署界面了。

    放弃这次借人、停手交给你。**不能**当成「列表里没有她」去退出重进 / 找替补
    （2026-09-29 17:27：你手动选好人开打了，它还以为没找到人，点开战斗里的菜单按了「退出作战」），
    也不能被 except Exception 吞掉之后接着点 —— 所以本模块里的兜底都用 PROPAGATE 放行它。
    """


# 本模块里「出错不影响流程」的兜底 except 也必须放行的异常
class OverlayBlockedError(Exception):
    """部署界面上盖着教学 / 提示（10/12 00:11 VA-1-10「剧情预设人形」，I-43）：点地格、读右下角都是白费，
    停下借人 / 补人，交给关卡流程去关它（有叉点叉、没叉点中间翻页），关掉之后重新借。"""


class SupportTabLostError(Exception):
    """要点「好友助战」页上按固定位置的东西（元素筛选、卡片），亮着的却是「我的人形」（I-46）。
    borrow_support 接住它，按 TAB_STUCK 停手；中间各处兜底都放行它（在 PROPAGATE 里）。"""


PROPAGATE = STOP_EXCEPTIONS + (UserTakeoverError, OverlayBlockedError, SupportTabLostError)

# 教学 / 提示的说明文字：屏幕中下部一长句。10/12 量过：「剧情预设人形」教学那句 32 个汉字在 (0.49W, 0.87H)；
# 23 张正常的部署界面中下部一句 8 个字以上的都没有（「本关至少需要出击4人」9 个字，所以门槛 12）
OVERLAY_CAPTION_MIN = 12
OVERLAY_CAPTION_BAND = (0.15, 0.50, 0.75, 0.97)
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')


def caption_text(name):
    """这块字像不像一句教学 / 提示的说明：≥12 个汉字，而且不是右下角那几句固定的提示 / 按钮。"""
    name = name or ''
    if len(_CJK_RE.findall(name)) < OVERLAY_CAPTION_MIN:
        return False
    return not (TILE_HINT_RE.search(name) or OCCUPIED_RE.search(name) or DEPLOY_BTN in name)


def overlay_caption(boxes, W, H):
    """屏幕中下部那句教学 / 提示的说明（框）；没有返回 None。"""
    x0, y0, x1, y1 = OVERLAY_CAPTION_BAND
    for b in boxes or []:
        if not caption_text(getattr(b, 'name', None)):
            continue
        cx, cy = b.x + b.width / 2.0, b.y + b.height / 2.0
        if x0 * W <= cx <= x1 * W and y0 * H <= cy <= y1 * H:
            return b
    return None

# 画面上有这些 = 已经开打了 / 正在确认开打，不是还能借人的部署界面
NOT_DEPLOY_RE = re.compile(r'行动结束|行动完成|是否开始作战|还有可部署')


class SupportPickMixin:
    """挂在 BaseGfTask 子类上，提供「在部署界面借指定好友助战」的能力。"""

    # ---------------------------------------------------------------- 节奏

    @property
    def ui_delay(self):
        """每次点击后额外等多久（秒），给界面动画留时间。

        点太快的后果不是「少等一下」而是直接点错：助战面板还在滑入时点页签，
        会被当成一次收起。所有点击后的 sleep 都加上它。
        """
        return float(getattr(self, '_ui_delay', 1.0))

    # ---------------------------------------------------------------- 现场记录

    def _trail(self, kind, text):
        """最近的动作 / 日志（墙上时间），存现场图时一起写进 .json。"""
        tr = getattr(self, '_case_trail', None)
        if tr is None:
            tr = self._case_trail = collections.deque(maxlen=CASE_TRAIL)
        tr.append((time.time(), kind, text))

    def log_info(self, message, *args, **kwargs):
        try:
            self._trail('info', str(message))
        except Exception:
            pass
        return super().log_info(message, *args, **kwargs)

    def log_error(self, message, *args, **kwargs):
        try:
            self._trail('error', '%s %s' % (message, ' '.join(repr(a) for a in args[:1])))
        except Exception:
            pass
        return super().log_error(message, *args, **kwargs)

    def _save_case_note(self, image_path, tag, extra=None):
        """在现场图旁边写一份同名 .json：当时在哪一关、第几阶段、窗口多大、相关配置、最近 30 条动作 /
        日志、最近一次整屏 OCR 读到的字。给以后查问题用，**写失败绝不影响流程**（连日志都只打一行）。"""
        try:
            if not image_path:
                return None
            run = getattr(self, '_sr_current', None)
            ocr = getattr(self, '_last_full_ocr', None)
            opts = {}
            try:
                opts = dict(self.support_options() or {})
            except PROPAGATE:
                raise
            except Exception:
                pass
            note = {
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'tag': tag,
                'image': os.path.basename(image_path),
                'task': type(self).__name__,
                'window': [getattr(self, 'width', None), getattr(self, 'height', None)],
                'stage': getattr(self, '_active_code', None),
                'run': None if run is None else {
                    'kind': run.kind, 'phase': run.phase, 'battle_started': run.battle_started,
                    'hud_seen': run.hud_seen, 'victory': run.victory, 'failed': run.failed,
                    'label': run.label, 'taps': run.tap_total, 'seconds': round(time.time() - run.wall0, 1),
                    'intro_taps': getattr(run, 'intro_taps', None), 'tutorial_taps': getattr(run, 'tutorial_taps', None),
                    'x_clicks': getattr(run, 'x_clicks_run', None), 'tap_fail': getattr(run, 'tap_fail', None),
                    'tap_given_up': getattr(run, 'tap_given_up', None)},
                'switches': {k: getattr(self, k, None) for k in (
                    '_yield_to_user', '_tap_to_reveal_skip', '_move_view_for_tile', '_await_manual',
                    '_close_tutorial_x')},
                'support_options': {k: v for k, v in opts.items() if isinstance(v, (str, int, float, bool))},
                'ocr_age_seconds': None if not ocr else round(time.time() - ocr[0], 1),
                'ocr': [] if not ocr else [
                    '%s @%d,%d %dx%d %.2f' % (b.name, b.x, b.y, b.width, b.height,
                                              float(getattr(b, 'confidence', 0) or 0))
                    for b in (ocr[1] or [])[:80]],
                'trail': ['%s.%03d %-5s %s' % (time.strftime('%H:%M:%S', time.localtime(t)), int(t * 1000) % 1000,
                                               k, s[:300])
                          for t, k, s in getattr(self, '_case_trail', ())],
                'extra': extra or {},
            }
            path = os.path.splitext(image_path)[0] + '.json'
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(note, f, ensure_ascii=False, indent=1, default=str)
            return path
        except PROPAGATE:
            raise
        except Exception as e:
            try:
                logger.info('写现场记录失败（不影响流程）：%r' % (e,))
            except Exception:
                pass
            return None

    # ---------------------------------------------------------------- 区域定义

    def support_panel_box(self) -> Box:
        """好友助战卡片列表所在的右侧面板。"""
        return self.box_of_screen(0.74, SUPPORT_PANEL_TOP, 1.0, 0.93, name='support_panel')

    def support_detail_box(self) -> Box:
        """左下角「支援 + 角色名 + Lv.xx」详情卡。"""
        return self.box_of_screen(0.02, 0.68, 0.34, 0.94, name='support_detail')

    def map_box(self) -> Box:
        """地图区域：避开顶部状态栏、右侧助战面板和底部按钮条。"""
        return self.box_of_screen(0.0, 0.10, 0.74, 0.93, name='map')

    def deploy_count_box(self) -> Box:
        """顶部「部署人形 3/4」。y 从 0.015 起，避开 FPS/CPU 之类的性能悬浮条。"""
        return self.box_of_screen(0.30, 0.015, 0.62, 0.10, name='deploy_count')

    # ---------------------------------------------------------------- 读界面

    def read_deploy_count(self):
        """读顶部「部署人形 x/y」，返回 (已部署, 上限)；读不到返回 None。

        只采信「部署人形」标签**同一行、右边最近**的那个框 —— 这条横带里还有敌人头顶的
        名字（「莫普1.5·试作版」），以前取第一个能解析的框，把「1.5」读成过「1/5」。
        标签和计数被 OCR 并成一个框（「部署人形3/4」）时，从标签后面截出来解析。
        10/11 起换成 deploy_count.read_deploy_count_boxes：2560x1440 下计数常和后面的字并成「3/4作战效能」「3/41」，
        10/10 MA-1-2 整关都读不到；0/0（进部署界面刚加载那一两秒）不再当成「阵容已满」。
        """
        return read_deploy_count_boxes(self.ocr(box=self.deploy_count_box()), self.width, self.height)

    def read_selected_unit_name(self):
        """读左下角详情卡上的全部文字（拼成一串），用来判断当前选中的是谁。

        卡被点成「展开」的（这一条里只有「暴击」「行动力」这些数值、没有 Lv）：改读上面名字那一行（I-58）。"""
        boxes = self.ocr(box=self.support_detail_box())
        text = ' '.join(b.name for b in boxes if b.name)
        expanded = bool(DETAIL_EXPANDED_RE.search(text)) and 'Lv' not in text
        self._detail_expanded = expanded
        if not expanded:
            return text
        x0, y0, x1, y1 = DETAIL_EXPANDED_NAME
        boxes = self.ocr(box=self.box_of_screen(x0, y0, x1, y1, name='support_detail_expanded'))
        named = ' '.join(b.name for b in boxes if b.name)
        if not getattr(self, '_detail_expanded_said', False):
            self._detail_expanded_said = True
            self.log_info('左下角详情卡是展开的，名字从上面那一行读：%s' % (named or '（也没读到）'))
        return named or text

    @staticmethod
    def name_matches(text, target):
        """详情卡文字里是否出现了 target。

        先按子串精确匹配；不中时对等长片段容忍 1 个字的 OCR 误差
        （例：威玛西娜 -> 威玛西纳）。名字本身够长够特别，容错 1 个字不会误伤。
        """
        if not text or not target:
            return False
        if target in text:
            return True
        n = len(target)
        if n < 3:
            return False
        for i in range(len(text) - n + 1):
            chunk = text[i:i + n]
            if sum(1 for a, b in zip(chunk, target) if a != b) <= 1:
                return True
        return False

    def list_support_cards(self):
        """列出当前可见的助战卡片锚点（即每张卡下方的「作战效能」数字框）。"""
        anchors = self.ocr(box=self.support_panel_box(), match=POWER_RE)
        return sorted(anchors, key=lambda b: (b.y, b.x))

    def _guard_support_page(self, what):
        """按「好友助战」页的固定位置点东西之前：亮着的是「我的人形」就一下都不点（I-46）。

        只认正面证据（own_panel.own_tab_active 看两个页签下面的橙色下划线）：看不出来时照旧点。
        :raises SupportTabLostError: 亮着的是「我的人形」
        """
        frame = self.frame
        if frame is None:
            return
        if own_tab_active(frame, self.width, self.height) is True:
            raise SupportTabLostError('要点%s，可亮着的是「我的人形」页签，不是「好友助战」—— 一下都不点'
                                      '（那一页同一个位置是「使用整备室配置」下拉框 / 你的人形）' % what)

    def click_support_card(self, anchor: Box, after_sleep=None):
        """点锚点对应的卡片立绘（锚点本身是数字条，往上挪到立绘中心更稳）。"""
        self._guard_support_page('助战卡片（作战效能 %s）' % anchor.name)
        x = (anchor.x + anchor.width / 2) / self.width
        y = (anchor.y + anchor.height / 2) / self.height - CARD_PORTRAIT_DY
        # 翻页后首行立绘可能被滚出面板、只剩数字条露在上沿；此时原样上移会点到元素筛选条，
        # 等于替用户切了一次筛选、整个候选列表被换掉。夹住下限。
        y = max(y, SUPPORT_PANEL_TOP + 0.01)
        self.click_relative(x, y, name='support_card_' + str(anchor.name),
                            after_sleep=(0.9 + self.ui_delay) if after_sleep is None else after_sleep)

    def scroll_support_list(self, amount=-4):
        """在助战列表上滚轮翻页；amount < 0 往下翻。

        走 _wheel 而不是 self.scroll：后者在独立版里因为缺 `mouse` 包一次都没滚动过，
        所以以前永远只看得到第一页（9/26 15:53 只看了 3 张卡就将就借了个 4243 的）。
        翻没翻动由调用方看「这一页有没有新卡」来判断。
        """
        self._wheel(int(0.88 * self.width), int(0.50 * self.height), amount)
        self.sleep(1.0 + self.ui_delay)
        self.next_frame()

    # ---------------------------------------------------------------- 元素筛选

    def select_element_filter(self, index):
        """点「好友助战」下方元素条的第 index 个图标（1~6），把候选缩到同元素。

        :return: True = 筛选已生效；False = 没点 / 筛空了已撤销
        """
        if not index or index < 1 or index > len(ELEMENT_TAB_X):
            return False
        x = ELEMENT_TAB_X[index - 1]
        self._guard_support_page('元素筛选第 %d 个图标' % index)
        before = len(self.list_support_cards())
        self.click_relative(x, ELEMENT_TAB_Y, after_sleep=1.2 + self.ui_delay,
                            name='element_tab_' + str(index))
        self.next_frame()
        after = len(self.list_support_cards())
        self.log_info('元素筛选第 %d 个图标：候选 %d -> %d 张' % (index, before, after))
        if after == 0 and before > 0:
            # 一张都不剩，多半是元素选错了。再点同一个图标取消筛选，改走全量查找。
            self.log_info('筛选后一张卡都不剩，取消筛选改为全量查找')
            self.clear_element_filter(index)
            return False
        return True

    def clear_element_filter(self, index):
        """再点一次同一个元素图标，把筛选取消掉。"""
        if not index or index < 1 or index > len(ELEMENT_TAB_X):
            return
        self._guard_support_page('元素筛选第 %d 个图标（取消筛选）' % index)
        self.click_relative(ELEMENT_TAB_X[index - 1], ELEMENT_TAB_Y,
                            after_sleep=1.2 + self.ui_delay,
                            name='element_tab_off_' + str(index))
        self.next_frame()

    # ---------------------------------------------------------------- 找地格

    def _blue_mask(self, frame):
        """蓝色地格掩码，并把肯定不是地格的区域抹掉。"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array(TILE_HSV_LOWER, dtype=np.uint8),
                           np.array(TILE_HSV_UPPER, dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # 顶部状态栏、右侧助战面板、底部按钮条；左下角详情卡那条蓝血条只在有卡时抹（I-40）
        return cv2.bitwise_and(mask, _roi_mask(h, w, detail_card_visible(frame)))

    def _split_tiles(self, frame):
        """把蓝色区域按「格子亮边框」切成一格一格。

        **阈值按每个连通块各自算**，理由见 TILE_RIDGE_PERCENTILE 上面那段注释。

        :return: [(内接半径, 落点x, 落点y)]，按半径从大到小
        """
        h, w = frame.shape[:2]
        mask = self._blue_mask(frame)
        v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        min_area = TILE_MIN_AREA_RATIO * w * h
        nb, blob_lab, blob_stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for bi in range(1, nb):
            if blob_stats[bi][4] < min_area:
                continue
            sel = (blob_lab == bi)
            # 亮边框 = 比**这一块**自身大多数像素都亮的那些
            thr = float(np.percentile(v[sel], TILE_RIDGE_PERCENTILE))
            ridge = cv2.dilate((sel & (v >= thr)).astype(np.uint8) * 255,
                               np.ones((5, 5), np.uint8))
            inner = cv2.morphologyEx(
                cv2.bitwise_and(sel.astype(np.uint8) * 255, cv2.bitwise_not(ridge)),
                cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            n, lab, stats, cents = cv2.connectedComponentsWithStats(inner, 8)
            for i in range(1, n):
                x, y, cw, ch, area = stats[i]
                if area < min_area:
                    continue
                blob = (lab[y:y + ch, x:x + cw] == i).astype(np.uint8) * 255
                pad = cv2.copyMakeBorder(blob, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
                _, mx, _, _pk = cv2.minMaxLoc(cv2.distanceTransform(pad, cv2.DIST_L2, 5))
                # **落点用质心，不用距离变换的峰值**。峰值按定义是「离边界最远的点」，
                # 站了人的格子会被人形把峰值挤到格子边上去 —— 实测图21 峰值离真正的
                # 格心 84px，而格子才 170px 高，已经贴着隔壁格的边界，一点就点到邻格
                # （这就是「它会点到 2/3 号的位置」的原因）。质心稳得多。
                out.append((float(mx), int(cents[i][0]), int(cents[i][1]))
                           + self._shape_of(pad))
        # 四边形优先，再比矩形度（见上面那段注释）。半径只当闸门，不参与排序。
        out.sort(key=lambda t: (t[3] != 4, -t[4]))
        return out

    @staticmethod
    def _shape_of(blob):
        """这块的形状有多像一个地格：返回 (多边形顶点数, 矩形度)。

        矩形度 = 轮廓面积 / 最小外接旋转矩形面积。等轴菱形转正之后就是个矩形，
        所以真地格接近 1；雾、走廊这些不规则区域明显更低。
        """
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0, 0.0
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        rect_area = rw * rh
        rectness = (cv2.contourArea(c) / rect_area) if rect_area > 0 else 0.0
        nv = len(cv2.approxPolyDP(c, TILE_QUAD_EPS * cv2.arcLength(c, True), True))
        return nv, float(rectness)

    def find_deploy_tiles(self, empty_only=True):
        """找能站人的地格，按「格子内部有多开阔」从大到小排，框心即落点。

        做法见 TILE_RIDGE_PERCENTILE 上面那段注释。**排序不可靠**：田字格里空格和
        站了人的格子内接半径几乎一样，真正判「这格能不能放人」的是点完之后顶部
        「部署人形 x/y」涨没涨 —— 见 _deploy_with_tile_retry。

        :param empty_only: True 只返回够大、值得一点的；False 全给，调试用
        """
        frame = self.frame
        if frame is None:
            return []
        w = frame.shape[1]
        min_r = MIN_TILE_RADIUS_RATIO * w
        max_r = MAX_TILE_RADIUS_RATIO * w
        found = self._split_tiles(frame)
        tiles = []
        for mx, px, py, nv, rectness in found:
            if empty_only and not (min_r <= mx <= max_r):
                continue
            r = max(int(mx), 1)
            # 不在这里截断成 MAX_TILE_CANDIDATES 个：先截断再排除「已试过的」，
            # 第 5 名以后的格子就永远轮不到了。截断放到 tile_candidates 里、排除之后做。
            tiles.append(Box(max(px - r, 0), max(py - r, 0), 2 * r, 2 * r, 1.0,
                             'tile_r%d_v%d_q%.2f' % (r, nv, rectness)))
        if empty_only and not tiles:
            best = max((f[0] for f in found), default=0.0)
            top = sorted(found, key=lambda f: -f[0])[:4]
            self.log_info('切出 %d 块，最大的一块内接半径 %.0fpx（要求 %.0f~%.0fpx），'
                          '判定当前视野里没有能放人的地格%s' % (
                              len(found), best, min_r, max_r,
                              '：' + '; '.join('(%d,%d)r%.0f' % (f[1], f[2], f[0]) for f in top) if top else ''))
            if (best >= TILE_NEAR_RATIO * min_r and not getattr(self, '_tile_near_saved', False)
                    and not getattr(self, '_tile_saving', False)):
                # 只差一点点够格：存一张图，看看那几块到底是不是空格（取证，不改判定）
                self._tile_near_saved = True
                self._tile_near = [(int(f[1]), int(f[2]), round(float(f[0]), 1)) for f in top]
                self._save_tile_debug('tile_small')
        return tiles

    # ---------------------------------------------------------------- 人形编号牌

    def unit_badge_points(self):
        """地图区域里的单个数字 = 已部署人形头顶那个编号牌，返回中心点列表。

        读不全无所谓（实测 3 个牌子常常只认出 1 个），有一个锚点就够用了。
        同一帧只 OCR 一次（以前一次重试里会跑两遍）。
        """
        frame = self.frame
        cache = getattr(self, '_badge_cache', None)
        if cache is not None and cache[0] is frame:
            return cache[1]
        try:
            boxes = self.ocr(box=self.map_box())
        except PROPAGATE:
            raise
        except Exception as e:
            self.log_info('找人形编号牌时 OCR 失败，锚点这条跳过：%r' % (e,))
            return []
        pts = []
        for b in boxes:
            name = (b.name or '').strip()
            if UNIT_BADGE_RE.match(name) and getattr(b, 'confidence', 1.0) >= UNIT_BADGE_MIN_CONF:
                pts.append((b.x + b.width // 2, b.y + b.height // 2))
        self._badge_cache = (frame, pts)
        return pts

    def _prefer_near_units(self, tiles):
        """把贴着已部署人形的地格候选**排到前面**（见 TILE_NEAR_UNIT_RATIO 那段注释）。

        以前这里是**过滤**：离编号牌远的直接扔掉。两个问题 ——
        1. 站了人的格子就在牌子正下方，永远过得了滤；过滤只会把空格和它挤在一起。
        2. 「空格一定挨着已上场的人形」在广场那张图上不成立：空格离最近的牌子 565px，
           门槛才 338px，真空格被整个扔掉。
        所以只改顺序、不删候选（稳定排序，原来的先后在各组内部保留）。
        """
        if not tiles:
            return tiles
        pts = self.unit_badge_points()
        if not pts:
            return tiles
        lim = TILE_NEAR_UNIT_RATIO * self.width

        def near(t):
            cx, cy = t.x + t.width // 2, t.y + t.height // 2
            return any((cx - px) ** 2 + (cy - py) ** 2 <= lim * lim for px, py in pts)

        return sorted(tiles, key=lambda t: not near(t))

    # ---------------------------------------------------------------- 闪烁

    def _grab_frames(self, n=BLINK_FRAMES):
        """连抓 n 帧，间隔 BLINK_INTERVAL（共约 1.2 秒）。抓不到返回空列表。"""
        frames = []
        for i in range(n):
            if i:
                self.sleep(BLINK_INTERVAL)
                self.next_frame()
            frame = self.frame
            if frame is None:
                return []
            frames.append(frame)
        return frames

    def _blink_baseline(self):
        """选人**之前**录一段「本来就在动的东西」，选人之后找闪烁时减掉（见 BLINK_MAP_SCALE 那段）。

        右下角不是「请点击地格」（选中了自己的人形）时空格可能已经在闪了，录进去就会把真闪烁减没，不录。
        """
        try:
            if self.tile_state(rereads=0) != TILE_HINT:
                return None
            frames = self._grab_frames()
            return swing_map(frames) if frames else None
        except PROPAGATE:
            raise
        except Exception as e:
            self.log_info('录闪烁底图失败（不影响流程）：%r' % (e,))
            return None

    def blink_map_candidates(self, frames, base=None):
        """整张地图上「成块、均匀地闪」的区域，当地格候选（不看颜色、露一角也算）。"""
        try:
            regions = blink_regions(frames, base=base)[:BLINK_MAP_TOP]
        except PROPAGATE:
            raise
        except Exception as e:
            self.log_info('整图找闪烁出错（不影响流程）：%r' % (e,))
            return []
        self.log_info('整张地图上找闪烁%s：%s' % (
            '（减掉了选人前就在动的）' if base is not None else '',
            '; '.join('(%d,%d)=%.0f' % (cx, cy, m) for m, cx, cy, _ in regions) or '没有'))
        out = []
        for m, cx, cy, r in regions:
            r = int(max(r, 8))
            out.append(Box(cx - r, cy - r, 2 * r, 2 * r, confidence=1.0, name='blinkmap s%.0f' % m))
        return out

    def rank_blinking_tiles(self, tiles, frames=None):
        """连抓几帧比亮度摆幅，挑出「一直在闪」的地格：一个明显最闪 -> [它]；前几名一起明显比其余的闪 ->
        这几个按幅度排好；分不出来 -> []，交回给原来的顺序。

        依据见 BLINK_FRAMES 那段注释。这条只用来加分，绝不该因为它判不出来就不部署了。

        :param frames: 已经抓好的帧（和整图找闪烁共用同一段，省 1.2 秒）
        """
        if len(tiles) < 2:
            return []
        if frames is None:
            frames = self._grab_frames()
        if not frames:
            return []
        series = [[] for _ in tiles]
        for frame in frames:
            h, w = frame.shape[:2]
            for k, t in enumerate(tiles):
                r = max(t.width // 4, 3)
                cx, cy = t.x + t.width // 2, t.y + t.height // 2
                patch = frame[max(cy - r, 0):min(cy + r, h), max(cx - r, 0):min(cx + r, w)]
                series[k].append(float(patch.mean()) if patch.size else 0.0)
        swings = [max(s) - min(s) for s in series]
        order = sorted(range(len(tiles)), key=lambda k: -swings[k])
        self.log_info('地格闪烁幅度：%s' % ', '.join(
            '(%d,%d)=%.1f' % (tiles[k].x + tiles[k].width // 2,
                              tiles[k].y + tiles[k].height // 2, swings[k]) for k in order))
        # 前 k 名（k = 1..BLINK_GROUP_MAX）都够亮、第 k 名又明显比第 k+1 名闪：这 k 个按幅度排在最前面
        for k in range(1, min(BLINK_GROUP_MAX, len(tiles) - 1) + 1):
            sk, nxt = swings[order[k - 1]], swings[order[k]]
            if sk < BLINK_MIN_SWING:
                break
            if sk >= BLINK_RATIO * max(nxt, 0.1):
                if k > 1:
                    self.log_info('前 %d 个候选一起明显比其余的闪（%.1f ~ %.1f，其余 ≤ %.1f），按闪烁幅度先试它们'
                                  % (k, swings[order[0]], sk, nxt))
                return [tiles[i] for i in order[:k]]
        self.log_info('几个候选闪得差不多，闪烁这条判不出来')
        return []

    def find_blinking_tile(self, tiles, frames=None):
        """rank_blinking_tiles 里只有一个明显最闪的那种：返回它；否则 None。"""
        ranked = self.rank_blinking_tiles(tiles, frames=frames)
        return ranked[0] if len(ranked) == 1 else None

    def grid_points(self, limit=GRID_FALLBACK_POINTS):
        """在「像地格」的区域上撒一批落点，按「离边界越远越靠前」排。

        切不出格子时的兜底，理由见 GRID_FALLBACK_POINTS 上面那段注释。
        掩码换成了 grid_mask（晴天发白的格子也算、阴影不算），并且补边、不会落到 x=0。
        """
        frame = self.frame
        if frame is None:
            return []
        T = blueness_tophat(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return scatter_points(grid_mask(frame, T, hsv), frame.shape[1], limit)

    def tile_candidates(self, skip_tried=True):
        """当前画面里所有值得点一下的落点，按先后顺序排好。

        四级，前一级的点优先，0.03·屏宽以内的重复点只留第一个：
          ① hole  被亮边框围住的干净区域（晴天发白的空格靠它；21 张图上从没给出过站人的格子）
          ② tile  原来的切格法（夜图 / 雾图 / 饱和蓝的格子上一直很好用），编号牌附近的排前面
          ③ blob  top-hat 上切出来的块（排除青色「V」格）
          ④ grid  新掩码上撒点
        「已经点过的」**先排除、再截断**，第 5 名以后的格子才轮得到。

        :return: [Box]，Box.name 标明来源和几何量，方便看日志
        """
        frame = self.frame
        if frame is None:
            return []
        w = frame.shape[1]
        T = blueness_tophat(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        default_r = int(0.012 * w)

        def box(x, y, r, name):
            r = max(int(r), 4)
            return Box(max(int(x) - r, 0), max(int(y) - r, 0), 2 * r, 2 * r, 1.0, name)

        def fresh(bs):
            return [b for b in bs if not (skip_tried and self._tile_tried(b))]

        misses = []
        holes = fresh([box(x, y, r, s) for x, y, r, s in hole_candidates(frame, T, hsv, misses=misses)])
        tiles = fresh(self._prefer_near_units(self.find_deploy_tiles()))[:MAX_TILE_CANDIDATES]
        for t in tiles:
            t.name = 'tile ' + t.name
        blobs = fresh([box(x, y, r, s) for x, y, r, s in blob_candidates(frame, T, hsv, misses=misses)])
        grid = fresh([box(x, y, default_r, 'grid d%.0f' % d)
                      for x, y, d in scatter_points(grid_mask(frame, T, hsv), w)])

        tol = 0.03 * w
        out = []
        for b in holes + tiles + blobs + grid:
            cx, cy = b.x + b.width // 2, b.y + b.height // 2
            if any(abs(cx - (o.x + o.width // 2)) < tol and abs(cy - (o.y + o.height // 2)) < tol
                   for o in out):
                continue
            out.append(b)
        # 只差颜色的那些：同一格已经是候选了（hole 过了、同一格的 blob 没过）就标上是第几个，别让人以为它被漏了（I-61 复查）
        marked = []
        for kind, x, y, hue, side in misses:
            k = next((i for i, o in enumerate(out) if abs(x - (o.x + o.width // 2)) < tol
                      and abs(y - (o.y + o.height // 2)) < tol), None)
            marked.append((kind, x, y, hue, side, '' if k is None else '=候选#%d %s' % (k + 1, out[k].name.split(' ')[0])))
        self._tile_hue_misses = marked
        return out

    def tile_state(self, rereads=2):
        """点完地格之后右下角是什么状态（四态，含义见 OCCUPIED_RE 上面那段注释）。

        只有「读不出」才重读 —— 另外三态都是游戏给的确定答复，不用等。
        最后一次读到的原文存在 self._last_br，打日志用。
        """
        for i in range(rereads + 1):
            if i:
                self.sleep(0.5)
                self.next_frame()
            boxes = self.ocr(box=self.box.bottom_right)
            self._last_br = [b.name for b in boxes if b.name]
            if self.find_boxes(boxes, match=DEPLOY_BTN):
                return TILE_EMPTY
            if self.find_boxes(boxes, match=OCCUPIED_RE):
                return TILE_OCCUPIED
            if self.find_boxes(boxes, match=TILE_HINT_RE):
                return TILE_HINT
        return TILE_UNKNOWN

    def _click_tile(self, tile, why=''):
        """点一个候选并读右下角。每一下都打一行 TILE_CLICK，失败时照着日志就能复盘。"""
        cx, cy = tile.x + tile.width // 2, tile.y + tile.height // 2
        self.click(cx, cy, after_sleep=1.0 + self.ui_delay, name='tile')
        self._tried_tiles = list(getattr(self, '_tried_tiles', ())) + [(cx, cy)]
        self._tile_click_budget = getattr(self, '_tile_click_budget', 999) - 1
        self.next_frame()
        st = self.tile_state()
        cnt = self.read_deploy_count()
        self.log_info('TILE_CLICK %s(%d,%d) [%s] -> %s  右下角=%s  部署=%s'
                      % (why, cx, cy, tile.name, TILE_STATE_NAMES[st],
                         getattr(self, '_last_br', []), '%d/%d' % cnt if cnt else '?'))
        ref = getattr(self, '_count_ref', None)
        if cnt and (not ref or cnt[1] != ref[1]):
            # 上限都不一样，说明入口那次读错了（或者压根没读到）—— 改信这次的
            self._count_ref = cnt
            if not ref and cnt[0] >= cnt[1] and getattr(self, '_count_unknown_at_entry', False):
                # 入口没读到、现在第一次读到就是满的：阵容本来就满，不是点地格把人放上去的（I-26）。别再找地格了
                self.log_info('进部署界面时没读到部署数，现在读到 %d/%d：阵容本来就是满的，不找地格了' % cnt)
                self._lineup_full = True
                self._lineup_changed = True      # 借这个闩让找地格 / 挪镜头的各层循环马上停
        elif st != TILE_EMPTY and cnt and ref and cnt[0] > ref[0]:
            # 点地格本身不该让人数变多 —— 变多了说明游戏直接把某个人形放上去了
            # （9/26 15:38 疑似出现过一次：一下撒点之后队伍作战效能正好多了一个自己人形的战力）。
            # 再点下去只会继续打乱阵容，停手。
            self.log_info('点地格之后部署数从 %d 涨到了 %d，游戏像是直接放上了一个人形，不再继续点'
                          % (ref[0], cnt[0]), notify=True)
            self._lineup_changed = True
        if st == TILE_UNKNOWN:
            cap = next((n for n in getattr(self, '_last_br', None) or () if caption_text(n)), None)
            if cap:
                # 右下角读到的是一句说明，不是「部署人形 / 撤回人形 / 请点击地格」：教学 / 提示盖住了（I-43）
                raise OverlayBlockedError('部署界面上盖着教学 / 提示（「%s」）' % cap[:20])
        return st

    def _try_click_candidates(self, limit=MAX_TILE_CLICKS_PER_VIEW, first=None, extra=()):
        """挨个点候选，直到右下角出现「部署人形」为止。

        右下角的四态是**干净的判据**：只有「部署人形」算选中；「撤回人形 / 编入队伍」是点到了
        站着人的格子，「请点击地格」是没点中，读不出来也**不算**选中 ——
        以前把读不出当成选中放行，正是「选完人却放不上去」的头号原因。

        点到站人的格子后不用先取消：直接点下一个空格就会换成选中那一格。

        :param first: 优先试的一个候选（闪烁认出来的那格 / 上次确认过是空格的那格），
                      **不受「已经点过」的限制**
        :param extra: 紧跟在 first 后面试的候选（整图找闪烁找到的其余区域），也不受「已经点过」的限制
        :param limit: 这个视野里最多点几下；另外还受 self._tile_click_budget 的总数限制
        """
        if getattr(self, '_tile_click_budget', 999) <= 0:
            self.log_info('这一轮找地格已经点了太多下，不再点了')
            return False
        tol = 0.03 * self.width
        cands = []
        for c in [first] + list(extra) + list(self.tile_candidates()):
            if c is None:
                continue
            cx, cy = c.x + c.width // 2, c.y + c.height // 2
            if any(abs((d.x + d.width // 2) - cx) < tol and abs((d.y + d.height // 2) - cy) < tol
                   for d in cands):
                continue
            cands.append(c)
        if not cands:
            return False
        n = min(len(cands), max(1, limit))
        self.log_info('地格候选 %d 个，这个视野点前 %d 个：%s' % (len(cands), n, '; '.join(
            '(%d,%d)%s' % (c.x + c.width // 2, c.y + c.height // 2, c.name.split(' ')[0])
            for c in cands[:n])))
        for i, tile in enumerate(cands[:n]):
            if getattr(self, '_tile_click_budget', 999) <= 0:
                self.log_info('这一轮找地格已经点了太多下，不再点了')
                return False
            if self._on_detail_card(tile.x + tile.width // 2, tile.y + tile.height // 2):
                # 候选是详情卡冒出来之前算的：点下去会把卡点成「展开」的，名字就读不出来了（I-58）
                self.log_info('候选 (%d,%d) 落在左下角详情卡上，跳过' % (tile.x + tile.width // 2, tile.y + tile.height // 2))
                continue
            st = self._click_tile(tile, '%d/%d ' % (i + 1, n))
            if st == TILE_EMPTY:
                self._empty_tile = tile
                return True
            if getattr(self, '_lineup_changed', False):
                return False
        return False

    def _on_detail_card(self, cx, cy):
        """(cx, cy) 落在左下角详情卡上吗。每次看**现在这一帧**的卡（不用读名字时记下的状态：换了部署界面就过时了，
        I-58 复查）；展开的卡大得多。"""
        W, H = float(getattr(self, 'width', 0) or 1), float(getattr(self, 'height', 0) or 1)
        frame = getattr(self, 'frame', None)       # 拿不到画面（测试桩）：不跳过，和以前一样
        if frame is None:
            return False
        if detail_card_expanded(frame):
            return cx < DETAIL_EXPANDED_AREA[0] * W and cy > DETAIL_EXPANDED_AREA[1] * H
        if not detail_card_visible(frame):
            return False
        return cx < CARD_AREA[0] * W and cy > CARD_AREA[1] * H

    # ---------------------------------------------------------------- 试过哪些格

    def _reset_tile_tries(self):
        """清空「已经点过的格子」。镜头动过之后必须清：旧的屏幕坐标对应的已经是别处了。"""
        self._tried_tiles = []
        self._empty_tile = None

    def _tile_tried(self, tile):
        """这一格刚才点过没有（同一格换个帧检测出来的落点会差几十像素）。"""
        return self._point_tried(tile.x + tile.width // 2, tile.y + tile.height // 2)

    def _point_tried(self, cx, cy):
        tol = 0.03 * self.width
        return any(abs(cx - x) < tol and abs(cy - y) < tol
                   for x, y in getattr(self, '_tried_tiles', ()))

    # ---------------------------------------------------------------- 退出重进

    def _wait_deploy_screen(self, time_out=120):
        """等回到部署界面（底部出现「作战开始」）。返回「作战开始」的框，没等到返回 None。

        以前走 ``BaseGfTask.skip_dialogs(has_dialog=True)``：认不出画面就每 2.5 秒盲点一次
        右上角 (0.95, 0.04)，最长 2 分钟（2026-09-29 用户反馈「一直点右上角」的来源之一）。
        现在只点认得出的东西（顶部「跳过」、跳过确认框）；认不出、字又少（过场 / 对话）时
        点一下屏幕中央唤出「跳过」，0.35 秒后只看顶部一条、有就点掉（按钮点出来 1~2 秒就淡出）。
        字多的画面（菜单）和加载画面不点。也不经过任何 skip_dialogs 重写，不会在借人途中再借一次。
        """
        deadline = time.time() + time_out
        last_tap, taps = 0.0, 0
        while time.time() < deadline:
            self.next_frame()
            try:
                boxes = self.ocr() or []
            except AttributeError:
                # WGC 偶尔抓回空帧（上游 skip_dialogs 同样处理）
                self.sleep(1)
                continue
            start = self.find_boxes(boxes, match=START_BATTLE_TEXT, boundary=self.box.bottom)
            if start:
                return start
            top = [b for b in boxes if b.name == SKIP_TEXT and b.y < SKIP_TOP_RATIO * self.height]
            if top:
                self.click_box(top[0], after_sleep=0.8)
                continue
            alert = self.find_boxes(boxes, match=NO_ALERT_TEXT)
            if alert:
                self.click_box(alert[0], after_sleep=0.3)
                self.next_frame()
                ok_btn = self.find_boxes(self.ocr() or [], match=['确认', '确定'])
                if ok_btn:
                    self.click_box(ok_btn[0], after_sleep=1.0)
                continue
            loading = any(re.search(r'资源加载中|\d{1,3}%', b.name or '') for b in boxes)
            if (not loading and len(boxes) <= REVEAL_MAX_BOXES and taps < REVEAL_TAP_LIMIT
                    and time.time() - last_tap >= REVEAL_TAP_INTERVAL):
                taps += 1
                last_tap = time.time()
                self.click_relative(*SKIP_REVEAL_POINT, after_sleep=0.35, name='tap_to_reveal_skip')
                self.next_frame()
                band = self.ocr(box=self.box_of_screen(0.5, 0, 1, SKIP_TOP_RATIO)) or []
                top = [b for b in band if b.name == SKIP_TEXT]
                if top:
                    self.click_box(top[0], after_sleep=0.8)
                continue
            self.sleep(0.5)
        return None

    def _still_on_deploy_screen(self):
        """重新截一帧看：还在能借人的部署界面吗 —— 底部有「作战开始」，没有战斗 HUD、没有开打确认框。
        读不出来当作不在（宁可不做，也别在不认识的画面上按退关）。"""
        self.next_frame()
        try:
            boxes = self.ocr() or []
        except PROPAGATE:
            raise
        except Exception:
            return False
        if not self.find_boxes(boxes, match=START_BATTLE_TEXT, boundary=self.box.bottom):
            return False
        return not any(NOT_DEPLOY_RE.search(b.name or '') for b in boxes)

    def reenter_stage(self, time_out=120):
        """从部署界面退出本关再重新进来（用来刷新好友助战列表）。

        路径：左上角 ☰ -> 关卡信息 -> 左下角「退出作战」-> 关卡详情面板「挑战」-> 部署界面

        三态返回，因为「没退成」和「退了但回不来」的严重程度完全不同：

        :return: True  = 已经重新回到部署界面
                 None  = 压根没退出本关，且已确认界面仍停在部署界面（安全，照常开打即可）
                 False = 界面停在未知状态（调用方应当中止，别再瞎点）
        :raises UserTakeoverError: 要退的时候画面已经不是部署界面了 —— 一个键都不按
        """
        # 退关收不回来：先确认还在部署界面。借人要几十秒，这期间你可能已经手动选好人开打了，
        # 战斗里左上角也是菜单、里面也有「退出作战」（2026-09-29 17:27 就这样把开打的关退了）
        if not self._still_on_deploy_screen():
            raise UserTakeoverError('准备退出重进时画面已经不是部署界面了（多半是你手动选好人开打了），'
                                    '不退出关卡')
        self.click_relative(*STAGE_MENU_BTN, after_sleep=2 + self.ui_delay, name='stage_menu')
        self.next_frame()

        if not self.wait_click_ocr(match='退出作战', box=self.box.bottom_left, time_out=8,
                                   after_sleep=3, raise_if_not_found=False, log=True):
            # 没进到「关卡信息」页，本关根本没退出。原路退回并确认还在部署界面。
            # 这时 ☰ 点出了什么没人知道（9/29 20:14 4-6，之前五次都点得出来）：存一张图
            self.log_info('左上角点开后没看到「退出作战」，放弃重进')
            self._save_tile_debug('reenter_nomenu')
            self.wait_click_ocr(match='继续作战', box=self.box.bottom_right, time_out=3,
                                after_sleep=2, raise_if_not_found=False)
            # 这里刻意用 wait_ocr 而不是 _wait_deploy_screen：后者认不出画面时会点屏幕中央
            # 唤出跳过，而这条「不确定在哪」的路径上什么都不该盲点。
            if self.wait_ocr(match='作战开始', box=self.box.bottom, time_out=8,
                             raise_if_not_found=False):
                return None
            return False

        # 退出可能还有一次确认框
        self.wait_click_ocr(match=['确认', '确定'], box=self.box.bottom, time_out=4,
                            after_sleep=2, raise_if_not_found=False, log=True)

        # 退回关卡详情面板，重新点「挑战」
        if not self.wait_click_ocr(match='挑战', box=self.box.right, time_out=25,
                                   after_sleep=3, raise_if_not_found=False, log=True):
            self.log_info('退出后没找到「挑战」，无法重进本关')
            return False

        if not self._wait_deploy_screen(time_out=time_out):
            self.log_info('重进后没等到「作战开始」')
            return False
        return True

    # ---------------------------------------------------------------- 主流程

    def tab_is_active(self, tab_box):
        """页签文字框正下方有没有那条橙色下划线 —— 有就说明它是当前页签。"""
        frame = self.frame
        if frame is None or tab_box is None:
            return False
        h, w = frame.shape[:2]
        y0 = max(0, tab_box.y + tab_box.height)
        y1 = min(h, y0 + int(TAB_UNDERLINE_BAND * h))
        x0 = max(0, tab_box.x)
        x1 = min(w, tab_box.x + tab_box.width)
        if y1 <= y0 or x1 <= x0:
            return False
        band = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(band,
                           np.array(TAB_UNDERLINE_HSV_LOWER, dtype=np.uint8),
                           np.array(TAB_UNDERLINE_HSV_UPPER, dtype=np.uint8))
        return float((mask > 0).mean()) >= TAB_UNDERLINE_MIN_RATIO

    def support_tab_box(self, time_out=6):
        """右上角「好友助战」页签的位置框；当前界面没有这个页签就返回 None。

        两个以前会漏的情况：
        - **右侧人形面板是收起的**：右上角只有「展开人形列表」、没有页签（9/26 12:54、13:20
          两次都因此当成「没有助战页签」，少一个人开打）。看到它就点一下展开再找 ——
          点的是文字**右边的图标**，文字本身点了不展开（见 EXPAND_ICON_DX / _click_expand_panel）。
          返回 None 时 ``self._panel_stuck`` 区分两种情况：True = 面板一直没展开（PANEL_STUCK）；
          False = 本来就展开着 / 点开了但这一关没有助战页签（NO_TAB）。
        - OCR 把旁边的字和页签并成一个框（「病毒好友助战」）：ok 的 find_boxes 传字符串是
          精确相等，所以用正则匹配，再把框裁到最后 4 个字上（下划线判定要用页签自己的宽度）。
        """
        deadline = time.time() + max(1.0, time_out)
        expand_clicks = 0
        self._panel_stuck = False
        while True:
            self.next_frame()
            boxes = self.ocr(box=self.box.top_right)
            tabs = self.find_boxes(boxes, match=SUPPORT_TAB_RE)
            if tabs:
                self._panel_stuck = False
                b = tabs[0]
                name = b.name or ''
                m = SUPPORT_TAB_RE.search(name)
                n = float(len(name) or 1)
                if m is not None and (len(name) > len(SUPPORT_TAB) or name != SUPPORT_TAB):
                    # 并了别的字的、认错一个字的：按字数比例裁到页签那 4 个字上，名字统一成「好友助战」
                    b = Box(b.x + b.width * m.start() / n, b.y, b.width * (m.end() - m.start()) / n, b.height,
                            b.confidence, SUPPORT_TAB)
                return b
            expand = self.find_boxes(boxes, match=EXPAND_PANEL_RE)
            if expand:
                # 面板收起着。点一下可能被动画吞掉，所以最多点 EXPAND_MAX_CLICKS 次，每次都把截止时间往后推
                self._panel_stuck = True
                if expand_clicks < EXPAND_MAX_CLICKS:
                    expand_clicks += 1
                    self._click_expand_panel(expand[0], expand_clicks)
                    deadline = max(deadline, time.time() + 2.0 + self.ui_delay)
                    continue
            elif self._panel_stuck and self.find_boxes(boxes, match=MY_UNITS_TAB_RE):
                # 点开了（「展开人形列表」没了、「我的人形」页签出来了），只是没有「好友助战」——
                # 这一关本来就不给借，按「没有助战页签」照常开打，别报 PANEL_STUCK 停手等人。
                # 只认这个正面证据：一帧什么都没读到（过场/漏读）不算展开，仍按卡住处理。
                self._panel_stuck = False
                self.log_info('点「展开人形列表」之后面板展开了（读到「我的人形」），但没有「%s」页签'
                              % SUPPORT_TAB)
            if time.time() >= deadline:
                self.log_info('右上角没找到「%s」页签，读到的是 %s'
                              % (SUPPORT_TAB, [x.name for x in boxes if x.name]))
                return None
            self.sleep(0.5)

    def _click_expand_panel(self, text_box, attempt):
        """点「展开人形列表」：前几次点文字右边的图标（EXPAND_ICON_DX 里每个偏移一次），之后点文字本身。

        文字本身 3/3 次没点开过（见 EXPAND_ICON_DX），留着当最后一下兜底，万一哪个版本文字也能点。
        """
        if attempt <= len(EXPAND_ICON_DX):
            k = EXPAND_ICON_DX[attempt - 1]
            x, y = expand_icon_point(text_box, self.width, k)
            self.log_info('右侧人形面板是收起的，点「展开人形列表」右边的图标（第 %d 次）：(%d, %d)，'
                          '文字框右边缘 + %.2f×字高' % (attempt, x, y, k))
            self.click(x, y, after_sleep=1.0 + self.ui_delay, name='展开人形列表_图标')
        else:
            self.log_info('右侧人形面板是收起的，图标点了 %d 次都没展开，改点「展开人形列表」文字本身兜底（第 %d 次）'
                          % (len(EXPAND_ICON_DX), attempt))
            self.click_box(text_box, after_sleep=1.0 + self.ui_delay)

    def has_support_tab(self, time_out=6):
        """当前界面有没有「好友助战」页签 —— 只看不点。

        「这个界面压根没有助战页签」要在花几十秒挪视角找地格**之前**就知道，
        但这时候还不能真去点它（点完地格面板会跳回「我的人形」，白点一次）。
        """
        return self.support_tab_box(time_out) is not None

    def switch_to_support_tab(self, time_out=6):
        """确保停在「好友助战」页签。

        **已经是当前页签时绝不能再点** —— 再点一次会把整个助战面板收回去，
        这正是「进关卡后选人界面自己收起来」的原因。靠橙色下划线判断当前页签。

        点完要看它真亮起来了（I-46）：10/12 00:17:45 点了一下没切过去，接着照固定位置点元素筛选，点进了「我的人形」页的
        「使用整备室配置」下拉框，阵容预设被换掉。亮着的还是「我的人形」就再点一次（这时它确实不是当前页签，再点不会收面板），
        还不行就抛 SupportTabLostError（borrow_support 按 TAB_STUCK 停手）。看不出亮的是哪个时照旧当切过去了 ——
        后面按固定位置点之前还有一道闸（_guard_support_page）。
        """
        tab = self.support_tab_box(time_out)
        if tab is None:
            return False
        if self.tab_is_active(tab):
            self.log_info('「好友助战」已经是当前页签，不再点（再点会把面板收回去）')
            return True
        for attempt in range(SUPPORT_TAB_CLICK_MAX):
            self.click_box(tab, after_sleep=1.0 + self.ui_delay)
            if self._wait_support_tab() is not True:
                return True
            if attempt + 1 < SUPPORT_TAB_CLICK_MAX:
                tab = self.support_tab_box(2) or tab
                # 第一下可能只是晚到：再点之前用刚截的这一帧再看一眼，已经亮了就别点（再点会把面板收回去，I-46 复查）
                if self.tab_is_active(tab) or own_tab_active(self.frame, self.width, self.height) is False:
                    return True
                self.log_info('点了「好友助战」，亮着的还是「我的人形」，再点一次')
        raise SupportTabLostError('点了 %d 次「好友助战」都没切过去，亮着的还是「我的人形」' % SUPPORT_TAB_CLICK_MAX)

    def _wait_support_tab(self):
        """点完「好友助战」之后等它亮起来。:return: own_tab_active 最后一次的读数（False = 切过去了）"""
        deadline = time.time() + SUPPORT_TAB_SWITCH_WAIT
        while True:
            self.next_frame()
            st = own_tab_active(self.frame, self.width, self.height) if self.frame is not None else None
            if st is not True or time.time() >= deadline:
                return st
            self.sleep(0.3)

    def scan_support_cards(self, max_pages=3, log_prefix='助战候选'):
        """逐张点开助战卡片并读出角色名，返回 {作战效能: 详情卡文字}。调试用。"""
        found = {}
        for page in range(max(1, max_pages)):
            self.next_frame()
            anchors = self.list_support_cards()
            fresh = [a for a in anchors if a.name not in found]
            if not fresh and page > 0:
                break
            for a in fresh:
                self.click_support_card(a)
                self.next_frame()
                found[a.name] = self.read_selected_unit_name()
                self.log_info(log_prefix + ' 作战效能=' + str(a.name) + ' -> ' + found[a.name])
            if page < max_pages - 1:
                self.scroll_support_list()
        return found

    def borrow_support(self, target_name, *args, **kwargs):
        """借助战（参数和返回值见 _borrow_support）。

        借到一半发现停在「我的人形」页上（点「好友助战」没切过去，I-46）：一下都不再点，按「放不上」处理 ——
        开关「选不中地格时停手」开着就抛 TileNotSelectedError 停手交给你。
        """
        try:
            return self._borrow_support(target_name, *args, **kwargs)
        except SupportTabLostError as e:
            self.log_info(str(e))
            try:
                stop = inspect.signature(SupportPickMixin._borrow_support).bind(
                    self, target_name, *args, **kwargs).arguments.get('stop_on_no_tile', True)
            except TypeError:
                stop = True
            return self._on_deploy_failed(stop, TAB_STUCK)

    def _borrow_support(self, target_name, max_pages=3, deploy=True, element_index=0,
                       reenter_times=0, min_power=0, ui_delay=None,
                       fallback_strongest=True, fallback_min_power=0,
                       stop_on_no_tile=True, backup_names=(), backup_element_index=0):
        """借助战；找不到（或战力不达标）就退出重进本关换一批好友再找。

        顺序：主选（够强，在 element_index 那一列里找）
              -> 备选（够强，在 backup_element_index 那一列里找；**同一份好友列表**，不为主选退出重进）
              -> 两个都没有才退出重进换一批好友 -> 最后一次放宽门槛（带个弱一点的也比不带强）
              -> 还是没有就借战力最高的其他助战（见 fallback_strongest）。

        :param backup_names: 备选角色名，按优先级（列表或逗号分隔的字符串）
        :param backup_element_index: 备选在元素条的第几列（1~6）；0 = 和主选在同一列里一起找

        :param fallback_strongest: 重进机会用完仍找不到她时，改借战力最高的其他助战
        :param fallback_min_power: 替补的优先门槛，够这个数的里面挑最高的
        :param stop_on_no_tile: 地格怎么都选不中时的态度。
                                True  = 抛 TileNotSelectedError，交给调用方停手等人接管
                                False = 只记一笔就返回，让这一关少一个人照常开打
        :return: True = 借到了
        :raises SupportStateError: 重进失败、界面停在未知状态
        """
        if ui_delay is not None:
            self._ui_delay = float(ui_delay)
        target_name = (target_name or '').strip()
        if not target_name:
            self.log_info('未填写「助战角色名」，跳过借助战')
            return False

        self._max_pages = max(1, max_pages)
        backups = [n for n in parse_names(backup_names) if n != target_name]
        names = '」「'.join([target_name] + backups)
        limit = max(0, reenter_times)
        reentered = 0
        for attempt in range(limit + 1):
            picked = self.pick_friend_support(target_name, max_pages=max_pages, deploy=deploy,
                                              element_index=element_index, min_power=min_power,
                                              allow_below=(attempt >= limit),
                                              backup_names=backups,
                                              backup_element_index=backup_element_index)
            if picked is True:
                return True
            if picked in (DEPLOY_FAILED, PANEL_STUCK, LINEUP_CHANGED):
                return self._on_deploy_failed(stop_on_no_tile, picked)
            if picked == NO_TAB:
                # 这个界面没有助战页签（多半是多阶段关卡的第二阶段部署），
                # 退出重进也变不出一个页签来，直接按原流程开打
                return False
            if picked == NO_TILE:
                # 地格没选中。退出重进换一批好友、改借替补，都不能让地格变得选得中 ——
                # 这两条路都没意义，所以无论开关怎么设都不走。
                if stop_on_no_tile:
                    # 硬开打就是少一个人，白送一局 —— 停手交回给人
                    raise TileNotSelectedError(
                        '没能选中空地格（右下角一直是「请点击地格或拖拽以部署单位」），'
                        '助战放不上去')
                self.log_info('地格没选中，按设置不停手 —— 这一关少一个人照常开打',
                              notify=True)
                return False
            if attempt >= limit:
                break
            self.log_info('第 %d 次没找到够强的「%s」（门槛 %d），退出重进本关换一批好友'
                          % (attempt + 1, names, min_power), notify=True)
            state = self.reenter_stage()
            if state is None:
                # 没退成、还在部署界面：不再试着重进。在这份列表里走完「最后一次放宽门槛」那一趟（和重进成功时
                # 一样，复查 I12），还没有再走下面「借最强的」那条后路。
                # 以前直接 return False —— 9/29 20:14 4-6 就这样没借人、3/4 开打，输了（配置里借最强的是开着的）
                self.log_info('没能退出本关，但界面仍停在部署界面：不再重进，在这份列表里放宽门槛再找一遍')
                self.next_frame()
                if self.list_support_cards():
                    # 上一趟把列表翻到了最后一页：滚回顶部再找（列表没开着就不滚，免得滚轮落在战场上）
                    self._scroll_support_top(max_pages)
                limit = attempt + 1         # 下一趟就是最后一趟（allow_below），之后 break
                continue
            if not state:
                raise SupportStateError(
                    '退出重进关卡失败，界面可能停在「关卡信息」或地图上，请手动确认后再跑')
            reentered += 1

        self.log_info('%s没找到「%s」'
                      % (('重进 %d 次仍' % reentered) if reentered else '', names), notify=True)
        if fallback_strongest:
            got = self.pick_strongest_support(max_pages=max_pages, min_power=fallback_min_power,
                                              deploy=deploy)
            if got == DEPLOY_FAILED:
                return self._on_deploy_failed(stop_on_no_tile, got)
            if got:
                return True
        self.log_info('替补也没借到，按原流程继续', notify=True)
        return False

    def _on_deploy_failed(self, stop_on_no_tile, why=DEPLOY_FAILED):
        """助战没能放上场（人选好了却放不上 / 面板展不开 / 点地格时阵容自己变了）。

        以前「人选好了却放不上」被 _after_picked 吞掉、照样返回「借到了」，于是少一个人按
        「作战开始」（9/26 下午 8 次「阵容没上满」都是这么来的）。现在和「选不中地格」同样处理：
        开关打开就停手交给人。
        **不能**返回普通的 False —— 那会被当成「列表里没有她」，触发退出重进 / 找替补。
        """
        self._save_tile_debug(why)
        msg = {
            DEPLOY_FAILED: '助战已经选好，但没能放上地格（右下角没出现「部署人形」，或按了之后部署数没涨）',
            PANEL_STUCK: '右侧人形面板是收起的，点了「展开人形列表」也没展开，找不到「好友助战」页签',
            LINEUP_CHANGED: '点地格时部署数自己涨了（游戏像是直接放上了你的人形），阵容已被打乱 —— '
                            '请检查阵容：需要的话撤回一个再放助战，或者直接点「作战开始」',
            TAB_STUCK: '点了「好友助战」没切过去（亮着的还是「我的人形」），为了不点错（那一页同一个位置是「使用整备室配置」'
                       '下拉框），借助战停下 —— 请你切到「好友助战」选好助战、放上地格',
        }.get(why, why)
        if stop_on_no_tile:
            raise TileNotSelectedError(msg)
        self.log_info('%s —— 按设置不停手，这一关照常开打' % msg, notify=True)
        return False

    def pick_strongest_support(self, max_pages=3, min_power=0, deploy=True):
        """替补方案：翻完助战列表，挑战力最高的那个借。

        ``min_power`` 是**优先**门槛而不是硬门槛：够门槛的里面挑最高的；一张都不够
        就退而挑全场最高的 —— 带个弱助战也比一个不带强（以前这里是干脆不借）。

        不去动元素筛选：走到这一步时 pick_friend_support 通常已经把筛选取消过了，
        再去点那个图标只会又筛回去（那对图标是开关式的）。
        """
        if not self.switch_to_support_tab():
            self.log_info('要找替补助战，但没有「好友助战」页签')
            return False
        # 前面找人时列表可能已经翻到最后一页了（翻页修好之后才会这样），先回到顶部再清点
        self._scroll_support_top(max_pages)
        seen = {}                       # 作战效能 -> 它在第几页
        for page in range(max(1, max_pages)):
            self.next_frame()
            for a in self.list_support_cards():
                seen.setdefault(a.name, page)
            if page < max_pages - 1:
                self.scroll_support_list()
        if not seen:
            self.log_info('替补也没戏：助战列表里一张卡都没识别到')
            return False

        qualified = {k: v for k, v in seen.items() if int(k) >= min_power}
        best = max(qualified or seen, key=lambda k: int(k))
        if min_power and not qualified:
            self.log_info('没有作战效能 >= %d 的助战，退而选全场最高的 %s'
                          % (min_power, best))
        self.log_info('改借战力最高的助战（作战效能 %s，共看了 %d 张卡）'
                      % (best, len(seen)), notify=True)
        # 够门槛的从高到低，然后是不够门槛的从高到低：放不上去就换下一张（I-58）
        order = sorted(qualified, key=int, reverse=True) + sorted(set(seen) - set(qualified), key=int, reverse=True)
        refused = []                    # 放不上去的角色名（同一个角色的另一张也跳过）
        tries = misses = 0
        result = False
        for power in order:
            if tries >= FALLBACK_CARD_TRIES or misses >= 2:
                break
            # 点过地格之后右边可能变成了「我的人形」：每张都先切回「好友助战」（已经是就不点，I-58 复查）
            if tries and not self.switch_to_support_tab():
                break
            # 翻回顶部，再一页一页往下找那张卡（滚 4 格不一定正好是一页，所以不按页码硬翻）
            if not self._goto_card(power, max_pages):
                misses += 1
                self.log_info('翻回去点 %s 那张卡时没找到它' % power)
                continue
            name = detail_name(self.read_selected_unit_name())
            if name and any(same_unit(name, r) for r in refused):
                continue
            tries += 1
            # 这个战力是「替补」的，不能拿去当下一关找目标角色的提示
            self._support_power_hint = None
            self._picked_power, self._picked_target = power, None
            self.log_info('已借替补助战，作战效能 %s%s' % (power, ('（%s）' % name) if name else ''))
            result = self._after_picked(None, deploy, tile_pending=getattr(self, '_tile_pending', False))
            if result is not DEPLOY_FAILED:
                return result
            if self.tile_state(rereads=0) != TILE_EMPTY:
                return result           # 是找不到空格放不上（不是这张卡被拒）：换卡也没用
            if name:
                refused.append(name)
            self.log_info('替补助战（作战效能 %s%s）放不上去 —— 多半是你队里有同一个角色，换下一张'
                          % (power, ('，%s' % name) if name else ''))
        return result if tries else False

    def _click_card_by_power(self, power):
        """按「作战效能」数字在当前可见列表里重新点中那张卡。"""
        self.next_frame()
        for a in self.list_support_cards():
            if a.name == power:
                self.click_support_card(a)
                self.next_frame()
                return True
        return False

    def _visible_powers(self):
        self.next_frame()
        return tuple(sorted(a.name for a in self.list_support_cards()))

    def _scroll_support_top(self, max_pages=3):
        """把助战列表滚回顶部：往上滚到可见的卡不再变化为止（最多 max_pages+1 次）。"""
        before = self._visible_powers()
        for _ in range(max(1, max_pages) + 1):
            self.scroll_support_list(4)
            now = self._visible_powers()
            if now == before:
                return
            before = now

    def _goto_card(self, power, max_pages=None):
        """在整个助战列表里找到作战效能为 power 的卡并点中。

        以前翻页从来没生效过（滚轮发不出去），卡永远在第一页，所以只看当前页就够了。
        翻页修好之后，找人时列表会停在最后一页 —— 回头点前面那页的卡就得先滚回顶部再往下找。
        """
        if not power:
            return False
        if self._click_card_by_power(power):
            return True
        pages = max(1, max_pages or getattr(self, '_max_pages', 3))
        self._scroll_support_top(pages)
        for page in range(pages + 1):
            if self._click_card_by_power(power):
                return True
            if page < pages:
                before = self._visible_powers()
                self.scroll_support_list()
                if self._visible_powers() == before:
                    break            # 到底了 / 滚不动
        return False

    def _search_support_cards(self, targets, max_pages, min_power=0):
        """在列表当前状态下逐张点开找人，一遍扫描同时找主选和备选。

        :param targets: [主选, 备选1, 备选2, ...]，按优先级排
        :return: (found, below)
                 found[名字] = 战力达标的那张卡的「作战效能」字符串（第一张），没有就是 None；
                 below[名字] = (战力数值, 作战效能字符串)，战力不达标里最强的一张，没有就是 None。
                 **主选一旦找到达标的就立刻停**，此刻它已经是选中状态；
                 其余情况（备选、不达标的）都要用 _goto_card 重新点回去。
        """
        if isinstance(targets, str):
            targets = [targets]
        primary = targets[0]
        hint = getattr(self, '_support_power_hint', None)
        tried = set()
        found = dict.fromkeys(targets)
        below = dict.fromkeys(targets)
        for page in range(max(1, max_pages)):
            self.next_frame()
            anchors = self.list_support_cards()
            fresh = [a for a in anchors if a.name not in tried]
            if not fresh:
                if page == 0:
                    self.log_info('好友助战列表里没识别到任何卡片')
                break
            # 上一关记住的作战效能优先试，命中就是一次点击的事
            fresh.sort(key=lambda a: (a.name != hint, a.y, a.x))
            for a in fresh:
                tried.add(a.name)
                self.click_support_card(a)
                self.next_frame()
                name = self.read_selected_unit_name()
                self.log_debug('助战卡片 作战效能=' + str(a.name) + ' -> ' + name)
                # 点了卡、左下角却读不到名字：先确认还在部署界面。你可能已经手动开打了，
                # 这一页剩下的「卡片」位置在战斗画面上就是地图，接着点会点到战场上
                # （9/29 17:27 开打之后它又照着旧的卡片位置点了 3 下）
                if not name and not self._still_on_deploy_screen():
                    raise UserTakeoverError('找人途中画面已经不是部署界面了（多半是你手动选好人开打了），不再点卡')
                who = next((t for t in targets if self.name_matches(name, t)), None)
                if who is None:
                    continue
                power = int(a.name)          # POWER_RE 保证是纯数字
                if power >= min_power:
                    if who == primary:
                        found[who] = a.name
                        return found, below
                    if found[who] is None:
                        found[who] = a.name
                        self.log_info('找到备选「%s」（作战效能 %d），先记下，继续找「%s」'
                                      % (who, power, primary))
                    continue
                self.log_info('找到「%s」但作战效能 %d 低于门槛 %d，继续找下一张'
                              % (who, power, min_power))
                if below[who] is None or power > below[who][0]:
                    below[who] = (power, a.name)
            if page < max_pages - 1:
                self.scroll_support_list()

        self.log_info('看了 %d 张卡：%s' % (len(tried), '；'.join(
            '「%s」%s' % (t, ('有（作战效能 %s）' % found[t]) if found[t] else
                         (('最高只有 %d（门槛 %d）' % (below[t][0], min_power)) if below[t] else '没有'))
            for t in targets)))
        return found, below

    @staticmethod
    def _choose_support(targets, found, below, allow_below):
        """按优先级挑一个：主选达标 > 备选达标（按填的先后）> [最后一次机会才考虑] 主选不达标 > 备选不达标。

        :return: (名字, 作战效能字符串, 达标吗)；一个都没有返回 (None, None, False)
        """
        for t in targets:
            if found[t]:
                return t, found[t], True
        if allow_below:
            for t in targets:
                if below[t]:
                    return t, below[t][1], False
        return None, None, False

    def pick_friend_support(self, target_name, max_pages=3, deploy=True, element_index=0,
                            min_power=0, allow_below=False, backup_names=(), backup_element_index=0):
        """在当前部署界面借好友助战里的 ``target_name``（不含退出重进）。

        :param min_power: 作战效能门槛，低于它的不要；0 = 不检查
        :param allow_below: 全列表都达不到门槛时，是否退而求其次选最强的那张
        :param backup_names: 备选角色名（按优先级）。这份列表里没有够强的主选时，
                             **就在同一份列表里**改借备选，不为了主选去退出重进
        :param backup_element_index: 备选在元素条的第几列；主选那一列里没有够强的主选时，
                                     换到这一列找备选。0 = 只在主选那一列（和全量兜底）里找
        :return: True          = 已选中，``deploy=True`` 时也已经放上场（或阵容本来就满）
                 NO_TAB        = 当前界面根本没有「好友助战」页签
                 NO_TILE       = 没有能放人的空地格，而且列表里也没有她（不值得重进 / 找替补）
                 DEPLOY_FAILED = 人选好了，但怎么都没放上地格
                 False         = 有页签，但列表里没有（够强的）她
        """
        target_name = (target_name or '').strip()
        if not target_name:
            self.log_info('未填写「助战角色名」，跳过借助战')
            return False

        # 刚进关卡时部署界面还在动画里，这时候碰右侧面板会被当成误触。先等它落定。
        self.sleep(1.0 + self.ui_delay)
        self.next_frame()

        before = self.read_deploy_count()
        # 读不到就多看几眼：10/10 18:56 进来那一下没读到，其实阵容已经 4/4 满了，当成要借人白找了 1 分钟地格（I-26）
        for _ in range(DEPLOY_COUNT_REREADS):
            if before is not None:
                break
            self.sleep(0.5)
            self.next_frame()
            before = self.read_deploy_count()
        self._reset_tile_tries()
        self._press_notes = []          # 这个部署界面里每次按「部署人形」没放上的取证（I-62）
        self._count_ref = before
        self._count_unknown_at_entry = before is None
        self._lineup_full = False
        self._lineup_changed = False
        self._tile_pending = False
        self._blink_base = None
        self.log_info('部署界面：部署人形 %s' % ('%d/%d' % before if before else '读不到'))

        # 阵容已经满了（强制出战的关卡常见）：再点地格只会点到站着人的格子，也没有位置放助战
        if deploy and before and before[0] >= before[1]:
            self.log_info('阵容已满 %d/%d，不借助战' % before)
            return True

        # 先只**看**一眼有没有助战页签，别点。没有的话就不必白花几十秒去找地格了。
        # （右侧面板收起时会顺手点「展开人形列表」，见 support_tab_box）
        if not self.has_support_tab():
            if getattr(self, '_panel_stuck', False):
                return PANEL_STUCK
            self.log_info('当前界面没有「好友助战」页签，跳过借助战')
            return NO_TAB

        # 顺序：**先点空地格，再选人，最后点「部署人形」**（主路线，验证最充分）。
        # 选不中空地格时退一步「先选人、再找地格」：选中助战之后空地格会闪，更好认；
        # 这个顺序游戏也认（旧日志里 7/7 次点中空格后右下角都出现了「部署人形」、部署数也涨了）。
        if deploy:
            picked_tile = self.pick_empty_tile(
                move_view=(getattr(self, '_move_view_for_tile', True)
                           and getattr(self, '_tile_search_failures', 0) < TILE_SEARCH_GIVE_UP))
            # 再硬确认一次：右下角必须是「部署人形」。站着人的格子、读不出来都不算。
            if picked_tile and self.tile_state() != TILE_EMPTY:
                self.log_info('右下角不是「部署人形」（%s），地格其实没选中'
                              % getattr(self, '_last_br', []))
                picked_tile = False
            if picked_tile:
                self._tile_search_failures = 0
            else:
                self._tile_search_failures = getattr(self, '_tile_search_failures', 0) + 1
                if self._tile_search_failures == TILE_SEARCH_GIVE_UP:
                    self.log_info('连续 %d 次都没能先选中空地格，后面几关先不挪视角了'
                                  % TILE_SEARCH_GIVE_UP, notify=True)
                now = self.read_deploy_count()
                full_from_start = getattr(self, '_lineup_full', False) or (
                    getattr(self, '_count_unknown_at_entry', False) and now and now[0] >= now[1]
                    and not (getattr(self, '_lineup_changed', False) and not getattr(self, '_lineup_full', False)))
                if full_from_start:
                    # 进来时没读到部署数、现在读到满员，点地格时也没见过人数往上涨：阵容本来就是满的（I-26）。
                    # 以前当成「阵容在点地格时变了」停手，10/10 18:56 干等了 19 分钟
                    self.log_info('阵容已满 %s（进部署界面时没读到部署数），不借助战'
                                  % ('%d/%d' % now if now else '满'))
                    self._lineup_changed = False
                    return True
                if getattr(self, '_lineup_changed', False) or (now and now[0] >= now[1]):
                    self.log_info('阵容在点地格时变了（部署 %s），助战放不上去了'
                                  % ('%d/%d' % now if now else '?'), notify=True)
                    return LINEUP_CHANGED
                self._save_tile_debug('no_tile')
                self.log_info('没先选中空地格，改成先选人：选中助战后空地格会闪，再找一次', notify=True)
                self._tile_pending = True
                # 趁还没选人（空格还不闪）录一段「本来就在动的」，选人之后找闪烁时减掉
                self._blink_base = self._blink_baseline()

        # **点完地格必须重新切一次「好友助战」**：点地格会让右侧面板自动跳回
        # 「我的人形」（2026-09-25 用户截图证实），不切回来后面翻的是自己的队伍。
        # switch_to_support_tab 自带「已经是当前页签就不点」的保护，白调一次也不会收面板。
        if not self.switch_to_support_tab():
            if getattr(self, '_panel_stuck', False):
                return PANEL_STUCK
            self.log_info('没能切到「好友助战」页签，跳过借助战')
            return NO_TAB

        # 她可能本来就是默认选中的助战，先省一轮点击 —— 但战力不达标就不能省
        if self.name_matches(self.read_selected_unit_name(), target_name):
            hint = getattr(self, '_support_power_hint', None)
            if not min_power or (hint and hint.isdigit() and int(hint) >= min_power):
                self.log_info('「' + target_name + '」已是当前助战，无需重新选择')
                self._picked_power, self._picked_target = hint, target_name
                return self._after_picked(before, deploy, tile_pending=self._tile_pending)
            self.log_info('当前助战已是「%s」，但战力未知/不达标，仍逐张核对' % target_name)

        targets = [target_name] + [n for n in parse_names(backup_names) if n != target_name]
        backups = targets[1:]
        names = '」「'.join(targets)

        def merge(more, more_below):
            for t in more:
                found[t] = found[t] or more[t]
                cands = [x for x in (below[t], more_below[t]) if x]
                below[t] = max(cands) if cands else None

        # 按元素把候选缩一缩，少点很多张卡。主选在「助战元素筛选」那一列里找（备选要是同一列也顺便找）
        filt = element_index if (element_index and self.select_element_filter(element_index)) else 0
        narrowed = bool(filt)          # 找过的都是筛过的列表：都没有就要全量兜底
        found, below = self._search_support_cards(targets, max_pages, min_power)
        # 扫描碰到达标的第一目标会当场停下，那张卡此刻就是选中状态（再点一次可能反而取消选中）
        selected_now = target_name if found[target_name] else None

        # 备选在她自己那一列里找（2026-09-29 用户截图：可露凯在第 6 列，威玛西娜在第 2 列 ——
        # 以前在第 2 列里连她一起找，当然找不到，只能最后取消筛选、整个列表一张张点过去碰运气）
        if (filt and backups and backup_element_index and backup_element_index != filt
                and not found[target_name] and not any(found[t] for t in backups)):
            self.log_info('第 %d 列里没有够强的「%s」，换到第 %d 列找备选「%s」'
                          % (filt, target_name, backup_element_index, '」「'.join(backups)))
            self.clear_element_filter(filt)
            filt = 0
            if self.select_element_filter(backup_element_index):
                filt = backup_element_index
                more, more_below = self._search_support_cards(backups, max_pages, min_power)
                merge(more, more_below)
                if more[backups[0]]:
                    selected_now = backups[0]

        if narrowed and not any(found.values()):
            # 两列里都没有够强的：元素序号可能填错了 —— 取消筛选再全量找一遍，
            # 让填错只是变慢而不是彻底找不到。备选那一列筛空时 select_element_filter
            # 已经自己撤销了筛选（filt 为 0），这时列表已是全量，直接找
            self.log_info('筛选后的候选里没有够强的「%s」，取消筛选改为全量查找' % names)
            if filt:
                self.clear_element_filter(filt)
            more, more_below = self._search_support_cards(targets, max_pages, min_power)
            merge(more, more_below)
            selected_now = target_name if more[target_name] else None

        who, power, qualified = self._choose_support(targets, found, below, allow_below)
        if who is None:
            if not any(found.values()) and not any(below.values()):
                self.log_info('这次进入的好友助战列表里没有「%s」' % names)
            # 地格也没选中：退出重进 / 找替补都不能让地格变得选得中，按「选不中地格」处理。
            # 否则交给 borrow_support 去决定要不要退出重进换一批好友
            return NO_TILE if self._tile_pending else False

        if who != target_name:
            self.log_info('列表里没有够强的「%s」，改借备选「%s」（作战效能 %s）'
                          % (target_name, who, power), notify=True)
        elif not qualified:
            self.log_info('已无重进机会，退而求其次选作战效能 %s 的「%s」' % (power, who))
        if not (qualified and who == selected_now):
            # 扫描当场停下的那张此刻就是选中状态；其余情况都要翻回去重新点
            if not self._goto_card(power, max_pages):
                self.log_info('想回头选 %s 那张，但翻遍列表也没找到它' % power)
                return NO_TILE if self._tile_pending else False
            if not self.name_matches(self.read_selected_unit_name(), who):
                self.log_info('点回作战效能 %s 那张，左下角却不是「%s」，不借了' % (power, who))
                return NO_TILE if self._tile_pending else False

        if who == target_name:
            self._support_power_hint = power if qualified else None
        self._picked_power, self._picked_target = power, who
        self.log_info('已选中助战「%s」%s(作战效能 %s)'
                      % (who, '' if who == target_name else '（备选）', power))
        return self._after_picked(before, deploy, tile_pending=self._tile_pending)

    def _after_picked(self, before, deploy, tile_pending=False):
        """选完人之后把她放上场。

        :return: True = 放上去了（或不需要部署）；DEPLOY_FAILED = 没放上去。
                 以前这里吞掉失败、永远返回 True，于是少一个人照样按「作战开始」。
        """
        if not deploy:
            return True
        try:
            ok = self._deploy_with_tile_retry(before, tile_pending=tile_pending)
        except PROPAGATE:
            raise
        except Exception as e:
            self.log_error('部署助战出错', e)
            ok = False
        return True if ok else DEPLOY_FAILED

    def _count_rose(self, before):
        """部署数比 before 多了吗 —— 放弃之前再看一眼，按完按钮读得太早的话人其实已经上场了。"""
        now = self.read_deploy_count()
        ref = before or getattr(self, '_count_ref', None)
        if now and ref and now[1] == ref[1] and now[0] > ref[0]:
            self.log_info('部署数已经从 %d 涨到 %d，助战其实放上去了' % (ref[0], now[0]))
            return True
        return False

    def _deploy_with_tile_retry(self, before, tile_pending=False):
        """点右下角「部署人形」把人放上去；没放上就换一格再试。

        田字格里几格长得几乎一样，靠画面分不出哪格是空的（见 find_deploy_tiles），
        所以让游戏当裁判：顶部「部署人形 x/y」涨了才算数。

        :param tile_pending: 地格还没选中（「先选人再找地格」那条路），先找格子再按部署
        """
        self._tile_click_budget = RESELECT_CLICK_BUDGET
        if tile_pending:
            # 第一轮在同一个视野里已经把候选全点过一遍了。那时助战还没选，空格不闪；
            # 现在选好了，同样这些格子要**重新**给机会，不然这条兜底一个候选都拿不到。
            self._reset_tile_tries()
        need_tile = tile_pending
        blink_used = False
        went_back = False
        for attempt in range(max(1, DEPLOY_RETRY_TILES)):
            if need_tile:
                if not self._reselect_tile_for_support(use_blink=not blink_used):
                    if self._count_rose(before):
                        return True
                    self.log_info('没有能换的空地格了，放弃部署')
                    return False
                blink_used = True
            if self.deploy_selected_unit(before):
                return True
            if self._count_rose(before):
                return True
            # 刚按过「部署人形」却没放上：如果右下角还是那个按钮、选中的也还是助战，
            # 多半只是那一下没被游戏收到 —— 同一格再按一次，别急着换格（换格会点到站人的格子）
            st = self.tile_state(rereads=0)
            need_tile = not (st == TILE_EMPTY and self._support_still_selected())
            if st == TILE_HINT and not went_back and getattr(self, '_empty_tile', None) is not None:
                # 格子被取消了（右下角变回「请点击地格」），刚才那格是确认过的空格：先回那一格再按一次。
                # 重试那条路里闪烁 / 整图闪烁的杂点会排在它前面，点过的格子又被剔掉，唯一的空格就轮不到了（I-62）
                went_back = True
                if self._back_to_empty_tile(before):
                    return True
                if self._count_rose(before):
                    return True
                need_tile = not (self.tile_state(rereads=0) == TILE_EMPTY and self._support_still_selected())
            if attempt + 1 < DEPLOY_RETRY_TILES:
                self.log_info('这一格没放上，%s（第 %d 次）'
                              % ('换一格重新部署' if need_tile else '同一格再按一次', attempt + 2))
        return False

    def _back_to_empty_tile(self, before):
        """按完「部署人形」格子被取消时：回到刚才确认过是空格的那一格，助战没选着就重新选回来，再按一次。

        :return: 放上了返回 True；那格不再是空格 / 助战选不回来 / 再按也没放上返回 False（之后照旧走换格重试）
        """
        tile = self._empty_tile
        cx, cy = tile.x + tile.width // 2, tile.y + tile.height // 2
        if self._on_detail_card(cx, cy):
            # 选了助战之后左下角的详情卡盖住了那一格：点下去会把卡点成「展开」的（I-58），照旧换格（复查）
            self.log_info('格子被取消了，刚才那一格 (%d,%d) 被左下角详情卡挡住，照旧换格' % (cx, cy))
            return False
        self.log_info('格子被取消了：回到刚才确认是空格的那一格 (%d,%d) 再试一次' % (cx, cy))
        if self._click_tile(tile, '回原格 ') != TILE_EMPTY:
            self.log_info('那一格现在不是空格了，照旧换格')
            return False
        if not self._support_still_selected():
            power = getattr(self, '_picked_power', None)
            self.log_info('助战没选着了，把作战效能 %s 那张卡重新点回来' % power)
            if not power or not self.switch_to_support_tab() or not self._goto_card(power):
                self.log_info('没能把助战重新选回来')
                return False
            if self.tile_state() != TILE_EMPTY or not self._support_still_selected():
                return False
        return bool(self.deploy_selected_unit(before))

    def _press_context(self):
        """按「部署人形」之前记下：游戏在不在前台、你的光标在画面上哪（取证用，读不到就是 None）。"""
        note = {'t': time.strftime('%H:%M:%S')}
        f = getattr(self, 'frame', None)
        if f is not None and hasattr(f, 'copy'):
            note['_frame0'] = f.copy()          # 按之前那一帧：没放上时拿它当底图，只报按下之后新冒出来的字
        for key, name in (('foreground', '_game_foreground'), ('cursor', '_cursor_in_frame')):
            fn = getattr(self, name, None)
            try:
                note[key] = fn() if callable(fn) else None
            except PROPAGATE:
                raise
            except Exception:
                note[key] = None
        return note

    def _press_wait(self, total):
        """按完「部署人形」之后等 total 秒，中途按 PRESS_FRAME_TIMES 各抓一帧（拷贝，不做 OCR）。"""
        frames, prev, longest = [], 0.0, 0.0
        for dt in PRESS_FRAME_TIMES:
            if dt > total:
                break
            self.sleep(dt - prev)
            longest = max(longest, dt - prev)
            prev = dt
            f = self.next_frame()
            if f is None:
                f = getattr(self, 'frame', None)
            if f is not None and hasattr(f, 'copy'):
                frames.append((dt, f.copy()))
        if total > prev:
            self.sleep(total - prev)
            longest = max(longest, total - prev)
        guard = getattr(self, '_sr_guard_input', None)
        if longest < 0.5 and callable(guard):
            guard()        # 每一段都太短、等待里一次都没看你动没动手（界面等待设得很小时）：补看一次（复查）
        return frames

    def _report_press_frames(self, frames, note):
        """没放上：OCR 按下后那几帧的画面中间找提示字、打一行日志、存图（存不下不影响流程）。"""
        try:
            note = dict(note or {})
            frame0 = note.pop('_frame0', None)
            box = self.box_of_screen(*PRESS_TOAST_BOX)

            def words_of(f):
                return [b.name for b in (self.ocr(box=box, frame=f) or []) if getattr(b, 'name', None)]
            base = set(words_of(frame0)) if frame0 is not None else set()
            texts = [(dt, words_of(f)) for dt, f in frames]
            # 地图上本来就有的字（敌人名牌、人形编号……）按之前就在：只报按下之后新冒出来的（复查）
            new = [(dt, [x for x in w if x not in base]) for dt, w in texts]
            seen = ['+%.2fs %s' % (dt, ' / '.join(w)) for dt, w in new if w]
            note.update(toast=[[dt, w] for dt, w in texts], toast_new=[[dt, w] for dt, w in new],
                        baseline=sorted(base))
            notes = getattr(self, '_press_notes', None)
            if notes is None:
                notes = self._press_notes = []
            self.log_info('按下「%s」之后画面中间新出来的字：%s；按的时候游戏%s前台，你的光标 %s（这个部署界面第 %d 次没放上）'
                          % (DEPLOY_BTN, '；'.join(seen) if seen else '没有',
                             {True: '在', False: '不在'}.get(note.get('foreground'), '（不知道在不在）'),
                             note.get('cursor'), len(notes) + 1))
            notes.append(note)
            if len(notes) > 1:
                return                          # 只存这个部署界面第一次没放上的那几帧（复查：别挤掉别的现场图）
            day = time.strftime('%Y%m%d')
            folder = os.path.join(DEBUG_TILE_DIR, day)
            os.makedirs(folder, exist_ok=True)
            stem = os.path.join(folder, '%s_deploy_press' % time.strftime('%H%M%S'))
            files = []
            for dt, f in frames:
                h, w = f.shape[:2]
                path = '%s_%03d.jpg' % (stem, int(dt * 100))
                cv2.imencode('.jpg', cv2.resize(f, (w // 2, h // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(path)
                files.append(os.path.basename(path))
            note['frames'] = files
        except PROPAGATE:
            raise
        except Exception as e:
            self.log_info('存按下之后的画面失败（不影响流程）：%r' % (e,))

    def _reselect_tile_for_support(self, use_blink=True):
        """人已经选好的情况下，重新选一个空地格。

        - 选中助战之后，**能放人的空地格会闪**，站了人的不会 —— 先用闪烁挑一个排在最前面。
          闪烁在全部候选上比（包括之前点过的），因为之前点的时候助战还没选、空格不闪。
        - 闪烁判不出来时，先试上一次确认过是空格的那一格。
        - 其余候选走和选人之前一样的四级管线，每点一下都读右下角四态。
        - 点中空格之后核对左下角详情卡还是不是助战：中途点到站人的格子，
          选中的就变成了那个人形，这时要把助战卡重新点回来再部署。
        """
        self.next_frame()
        first, extra = None, []
        if use_blink:
            # 同一段帧两种用法：在已有候选里比谁闪得最厉害；再在整张地图上找在闪的区域
            # （候选一个都切不出来时 —— 粉紫色的格子、被按钮挡得只剩一角的格子 —— 只能靠后者）
            frames = self._grab_frames()
            ranked = self.rank_blinking_tiles(self.tile_candidates(skip_tried=False)[:8], frames=frames)
            extra = self.blink_map_candidates(frames, base=getattr(self, '_blink_base', None))
            if ranked:
                first, extra = ranked[0], ranked[1:] + extra
            elif extra:
                first = extra.pop(0)
            if first is not None:
                self.log_info('按闪烁先试 (%d,%d)' % (first.x + first.width // 2, first.y + first.height // 2))
        if first is None:
            first = getattr(self, '_empty_tile', None)
        if not self._try_click_candidates(first=first, extra=extra):
            return False
        if self._support_still_selected():
            return True
        power = getattr(self, '_picked_power', None)
        self.log_info('选中的已经不是助战了，把作战效能 %s 那张卡重新点回来' % power)
        if not power or not self.switch_to_support_tab() or not self._goto_card(power):
            self.log_info('没能把助战重新选回来')
            return False
        return self.tile_state() == TILE_EMPTY and self._support_still_selected()

    def _support_still_selected(self):
        """左下角详情卡上显示的还是我们选的助战吗（名字对得上，或带「支援」标签）。"""
        text = self.read_selected_unit_name()
        target = getattr(self, '_picked_target', None)
        return bool(text) and (('支援' in text) or (bool(target) and self.name_matches(text, target)))

    def tile_selected(self):
        """当前是否选中了一个**能放人**的空地格。只有右下角是「部署人形」才算。

        四态细节见 tile_state；站着人的格子、读不出来都算「没选中」。
        """
        return self.tile_state() == TILE_EMPTY

    def _save_tile_debug(self, event):
        """把当前画面和候选落点存下来（原图 + 标注图），下次失败就有现场可对照。

        只保留最近 DEBUG_TILE_KEEP 张。存图失败绝不影响主流程。
        """
        if getattr(self, '_tile_saving', False):
            return                      # 标注时要重新找候选，别在这里面再触发一次存图
        self._tile_saving = True
        try:
            frame = self.frame
            if frame is None:
                return
            day = time.strftime('%Y%m%d')
            folder = os.path.join(DEBUG_TILE_DIR, day)
            os.makedirs(folder, exist_ok=True)
            stem = os.path.join(folder, '%s_%s' % (time.strftime('%H%M%S'), event))
            cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(stem + '.jpg')
            vis = frame.copy()
            try:
                vis[grid_mask(frame, blueness_tophat(frame), cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)) > 0] //= 2
                for i, c in enumerate(self.tile_candidates(skip_tried=False)[:12]):
                    cx, cy = c.x + c.width // 2, c.y + c.height // 2
                    cv2.circle(vis, (cx, cy), max(c.width // 2, 12), (0, 255, 255), 3)
                    cv2.putText(vis, '%d %s' % (i + 1, c.name.split(' ')[0]), (cx + 14, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                for x, y in getattr(self, '_tried_tiles', ()):
                    cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 4)
                for kind, x, y, hue, _side, _dup in getattr(self, '_tile_hue_misses', None) or ():
                    # 只差色相没过的 hole / blob：橙色方框 + 色相（I-61 取证：实时帧上的色相比 JPEG 上可靠）
                    cv2.rectangle(vis, (x - 30, y - 30), (x + 30, y + 30), (0, 128, 255), 3)
                    cv2.putText(vis, '%s H%.0f' % (kind, hue), (x + 34, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (0, 128, 255), 3)
                if event == 'tile_small':
                    # 因为太小被丢掉的那几块：品红圈 + 半径
                    for x, y, r in getattr(self, '_tile_near', ()):
                        cv2.circle(vis, (x, y), max(int(r), 8), (255, 0, 255), 3)
                        cv2.putText(vis, 'r%.0f' % r, (x + 12, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (255, 0, 255), 3)
            except PROPAGATE:
                raise
            except Exception:
                pass
            h, w = vis.shape[:2]
            cv2.imencode('.jpg', cv2.resize(vis, (w // 2, h // 2)),
                         [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(stem + '_ann.jpg')
            self.log_info('已存地格现场图：%s.jpg' % stem)
            hue_misses = list(getattr(self, '_tile_hue_misses', None) or ())
            if hue_misses:
                self.log_info('只差颜色没算进候选的格子：%s' % '; '.join(
                    '(%d,%d)%s H%.1f %s%s' % (x, y, kind, hue, side, '[%s]' % dup if dup else '')
                    for kind, x, y, hue, side, dup in hue_misses[:6]))
            self._save_case_note(stem + '.jpg', event, extra={
                'hue_misses': hue_misses,
                'tried_tiles': [list(map(int, p)) for p in getattr(self, '_tried_tiles', ())],
                'deploy_count_ref': getattr(self, '_count_ref', None),
                'bottom_right': getattr(self, '_last_br', None),
                'tile_pending': getattr(self, '_tile_pending', None),
                'picked': [getattr(self, '_picked_target', None), getattr(self, '_picked_power', None)],
                'deploy_press': getattr(self, '_press_notes', None) or None,
                'near_tiles': getattr(self, '_tile_near', None) if event == 'tile_small' else None})
            prune_debug_dir(DEBUG_TILE_DIR, DEBUG_TILE_KEEP)
        except PROPAGATE:
            raise
        except Exception as e:
            self.log_info('存地格现场图失败（不影响流程）：%r' % (e,))
        finally:
            self._tile_saving = False

    # ---------------------------------------------------------------- 挪视角

    def _map_roi(self):
        """地图区域的灰度缩略图（float32），用来比对镜头动没动。

        裁掉右侧助战面板、顶部状态栏和底部按钮条 —— 那些是固定 UI，
        混进来会把「没动」判成「动了」。
        """
        frame = self.frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        sub = frame[int(0.15 * h):int(0.85 * h), int(0.05 * w):int(0.70 * w)]
        if sub.size == 0:
            return None
        return np.float32(cv2.cvtColor(cv2.resize(sub, (320, 180)), cv2.COLOR_BGR2GRAY))

    def _fresh_roi(self):
        """先抓新帧再取缩略图 —— self.frame 可能是好几秒前缓存的那一帧。"""
        self.next_frame()
        return self._map_roi()

    def _map_shift(self, before, after):
        """两帧之间地图整体平移了多少 —— 相位相关，直接给出方向和幅度。

        比「两张图有没有差别」可靠得多：人物待机动画、火光特效都会让画面变，
        但它们不会让**整幅图**一致地平移，相位相关的响应值就上不去。

        :return: (dx, dy, 置信度)，单位是缩略图像素；判不了返回 (0, 0, 0)
        """
        if before is None or after is None or before.shape != after.shape:
            return 0.0, 0.0, 0.0
        han = getattr(self, '_pan_hann', None)
        if han is None or han.shape != before.shape:
            han = cv2.createHanningWindow((before.shape[1], before.shape[0]), cv2.CV_32F)
            self._pan_hann = han
        (dx, dy), resp = cv2.phaseCorrelate(before, after, han)
        return float(dx), float(dy), float(resp)

    def _moved(self, before, after):
        """镜头到底动没动。返回 (动了吗, dx, dy, 置信度)。"""
        dx, dy, resp = self._map_shift(before, after)
        dist = (dx * dx + dy * dy) ** 0.5
        need = PAN_MIN_SHIFT_RATIO * (before.shape[1] if before is not None else 320)
        return (dist >= need and resp >= PAN_MIN_RESPONSE), dx, dy, resp

    # ---- 直接对游戏窗口发消息（原因见 PAN_METHOD_NAMES 上面那段注释）----

    def _own_input(self, fn):
        """挪视角自己拼的输入（滚轮 / 按住键 / 拖拽）也要过宿主的输入关口：借助战途中你在操作就不做，
        做了记成「自己的动作」，别被当成你在动（见 stage_runner._sr_input）。整个手势算一次输入 ——
        按下和松开之间不会被打断。插件版没有这个关口，直接做。"""
        gate = getattr(self, '_sr_input', None)
        return gate(fn) if gate else fn()

    def _genshin(self):
        """ok 的 Genshin 交互对象；不是这种交互（比如插件版配了别的）就返回 None。"""
        it = getattr(getattr(self, 'executor', None), 'interaction', None)
        if it is None or not all(hasattr(it, a) for a in ('operate', 'post', 'capture')):
            return None
        if not hasattr(it.capture, 'get_abs_cords'):
            return None
        return it

    def _wheel(self, x, y, count):
        """在 (x, y)（窗口内像素）滚 count 格；count<0 往下 / 缩小。

        不走 self.scroll：Genshin 交互的 do_scroll 要 `import mouse`，ok-gf2 没装这个包，
        一次都没发出去过。这里在一次 operate()（激活窗口 -> 做完 -> 还原光标）里，
        把真光标挪到目标点，再给窗口发 WM_MOUSEWHEEL（lParam 是**屏幕**坐标）。
        """
        it = self._genshin()
        if it is None:
            self._own_input(lambda: self.scroll(x, y, count))
            return
        import win32api
        import win32con
        ax, ay = it.capture.get_abs_cords(x, y)
        ax, ay = int(ax), int(ay)
        sign = 1 if count > 0 else -1
        wp = _makelong(0, sign * win32con.WHEEL_DELTA)
        lp = _makelong(ax, ay)

        def fn():
            win32api.SetCursorPos((ax, ay))
            time.sleep(0.05)
            it.post(win32con.WM_MOUSEMOVE, 0, _makelong(x, y))
            for _ in range(abs(int(count))):
                it.post(win32con.WM_MOUSEWHEEL, wp, lp)
                time.sleep(0.06)
            time.sleep(0.1)

        self._own_input(lambda: self._operate_blocking(it, fn))
        reset = getattr(self.executor, 'reset_scene', None)
        if reset:
            reset()

    @staticmethod
    def _operate_blocking(it, fn):
        """it.operate(fn, block=True)，但无论出什么事都保证解除 BlockInput。

        ok 的 operate() 收尾那几步（还原光标）不在 try 里，那里一抛异常，
        物理键鼠就一直被锁着，只能 Ctrl+Alt+Del。BlockInput(False) 重复调用无害。
        """
        try:
            it.operate(fn, block=True)
        finally:
            unblock = getattr(it, 'unblock_input', None)
            if unblock:
                try:
                    unblock()
                except Exception:
                    pass

    def _hold_key(self, key, hold):
        self._own_input(lambda: self._hold_key_now(key, hold))

    def _hold_key_now(self, key, hold):
        """按住一个键 hold 秒再**真的松开**。

        Genshin 交互的 send_key_up 只 deactivate、不发 WM_KEYUP；send_key 的 lParam
        又把所有键的扫描码写死成 A 键的。所以自己发 KEYDOWN / 等待 / KEYUP
        （用 PostMessage 交互的 lParam，扫描码是对的），KEYUP 放在 finally 里保证一定发。
        不走 operate()：它会在前后存/还原光标位置，按住的这半秒里你挪了鼠标会被拽回去；
        按键本来就用不着光标。
        """
        it = self._genshin()
        pi = getattr(it, 'post_interaction', None) if it is not None else None
        if pi is None or not hasattr(pi, 'send_key_down') or not hasattr(pi, 'send_key_up'):
            self.send_key(key, down_time=hold)
            return
        hw = getattr(it, 'hwnd_window', None)
        bg = hw is not None and not hw.is_foreground()
        if bg:
            it.activate()
        try:
            pi.send_key_down(key, activate=False)
            time.sleep(hold)
        finally:
            pi.send_key_up(key)
            time.sleep(0.03)
            if bg:
                it.deactivate()

    def _scene_change(self, a, b):
        """两张缩略图之间「明显变了」的像素占比。"""
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float((cv2.absdiff(a, b) > 12).mean())

    def _zoom(self, count):
        """滚轮缩放，并**验证画面真的变了**。

        以前直接比较滚轮前后两帧，前一帧还是几秒前缓存的，云影、光斑一动就「变了」——
        滚轮明明没发出去，却 8 次里 8 次报「画面变了」，每关白耗十几秒。
        现在先量一段**不做任何操作**时的自然变化当基线，滚完的变化要明显超过它才算数。

        :return: True 生效 / False 没反应 / None 画面自己动得太厉害，判不了
        """
        wait = 0.5 + self.ui_delay
        a = self._fresh_roi()
        self.sleep(wait)
        b = self._fresh_roi()
        self._wheel(int(VIEW_PROBE_POINT[0] * self.width), int(VIEW_PROBE_POINT[1] * self.height), count)
        self.sleep(wait)
        c = self._fresh_roi()
        noise, change = self._scene_change(a, b), self._scene_change(b, c)
        if noise > 0.3:
            self.log_info('滚轮 %+d 格：画面自己在大幅变化（%.3f），判不了缩放有没有生效' % (count, noise))
            return None
        changed = change > max(0.08, 2.5 * noise)
        self.log_info('滚轮 %+d 格：画面变化 %.3f（不动时 %.3f）-> %s'
                      % (count, change, noise, '缩放生效' if changed else '没反应'))
        return changed

    def _zoom_search(self):
        """缩放之后再找地格。返回 True = 选中了空地格。

        先找出哪个方向能缩（**两个方向都试**：镜头可能刚好顶在一侧的极限上），
        再往这个方向多缩一档、然后反过来放大，每档都重新认一遍格子。
        连着 WHEEL_GIVE_UP 个部署界面两个方向都没反应，这次运行里就不再试缩放。
        没找到时把**真正生效了**的那几档还原回去，免得后面的平移搜索视野变小。
        """
        if getattr(self, '_wheel_fail', 0) >= WHEEL_GIVE_UP:
            return False
        applied, d = 0, None
        for sign in (-1, 1):
            self.log_info('没点中空地格，试试滚轮%s' % ('往下' if sign < 0 else '往上'))
            r = self._zoom(4 * sign)
            if r is None:
                return False
            if r:
                d, applied = sign, 4 * sign
                break
        if d is None:
            self._wheel_fail = getattr(self, '_wheel_fail', 0) + 1
            if self._wheel_fail >= WHEEL_GIVE_UP:
                self.log_info('连着 %d 个部署界面滚轮两个方向都没反应，这次运行后面不再试缩放'
                              % WHEEL_GIVE_UP, notify=True)
            return False
        self._wheel_fail = 0
        for step in (0, 4 * d, -12 * d):
            if step:
                if not self._zoom(step):
                    break
                applied += step
            self._reset_tile_tries()        # 缩放之后旧坐标对应的已经不是同一格了
            if self._try_click_candidates(limit=MOVED_VIEW_CLICKS):
                return True
            if getattr(self, '_lineup_changed', False):
                return False
        if applied:
            self._zoom(-applied)
            self._reset_tile_tries()
        return False

    def _drag(self, fx, fy, tx, ty, key='right'):
        """按住鼠标某个键从 (fx,fy) 拖到 (tx,ty)（相对坐标），**整个手势在一次 operate() 里**。

        ok 的 mouse_down / move / mouse_up 是三次独立的 operate()，每次之后都把窗口设成
        非激活、光标放回原处，move 也不带按键标志 —— 游戏眼里根本不是一次拖拽。
        这里每一步都挪真光标、并在 WM_MOUSEMOVE 里带上按住的键。
        """
        x0, y0 = int(fx * self.width), int(fy * self.height)
        x1, y1 = int(tx * self.width), int(ty * self.height)
        it = self._genshin()
        if it is None:
            def gesture():
                self.mouse_down(x0, y0, key=key, name='pan_%s' % key)
                try:
                    for i in range(1, PAN_DRAG_STEPS + 1):
                        self.move(int(x0 + (x1 - x0) * i / PAN_DRAG_STEPS),
                                  int(y0 + (y1 - y0) * i / PAN_DRAG_STEPS))
                        self.sleep(PAN_DRAG_STEP_SLEEP)
                finally:
                    self.mouse_up(key=key)

            self._own_input(gesture)
        else:
            import win32api
            import win32con
            if key == 'right':
                down, up, mk = win32con.WM_RBUTTONDOWN, win32con.WM_RBUTTONUP, win32con.MK_RBUTTON
            else:
                down, up, mk = win32con.WM_LBUTTONDOWN, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON

            def at(x, y, msg, wparam):
                # 挪光标失败也要把消息发出去 —— 尤其是 finally 里那个松开按键
                try:
                    ax, ay = it.capture.get_abs_cords(x, y)
                    win32api.SetCursorPos((int(ax), int(ay)))
                except Exception:
                    pass
                it.post(msg, wparam, _makelong(x, y))

            def fn():
                at(x0, y0, win32con.WM_MOUSEMOVE, 0)
                time.sleep(0.03)
                at(x0, y0, down, mk)
                time.sleep(0.06)
                try:
                    for i in range(1, PAN_DRAG_STEPS + 1):
                        at(int(x0 + (x1 - x0) * i / PAN_DRAG_STEPS),
                           int(y0 + (y1 - y0) * i / PAN_DRAG_STEPS), win32con.WM_MOUSEMOVE, mk)
                        time.sleep(PAN_DRAG_STEP_SLEEP)
                    time.sleep(0.15)
                finally:
                    at(x1, y1, up, 0)
                    time.sleep(0.03)

            self._own_input(lambda: self._operate_blocking(it, fn))
        self.sleep(0.4 + self.ui_delay)
        self.next_frame()

    def _pan_once(self, method, dx, dy, center=None):
        """按 ``method`` 往 (dx, dy) 方向挪一下镜头（相对屏幕的比例）。"""
        cx, cy = center or PAN_CENTERS[0]

        def clamp(v, lo, hi):
            return min(max(v, lo), hi)

        if method == 'right_drag':
            fx = clamp(cx - dx / 2.0, *PAN_BOUNDS_X)
            tx = clamp(cx + dx / 2.0, *PAN_BOUNDS_X)
            fy = clamp(cy - dy / 2.0, *PAN_BOUNDS_Y)
            ty = clamp(cy + dy / 2.0, *PAN_BOUNDS_Y)
            self._drag(fx, fy, tx, ty, key='right')
        elif method in PAN_KEY_SETS:
            # 按住对应的键。dx>0 = 想让镜头往右边看
            for key, hold in self._pan_keys_for(method, dx, dy):
                self._hold_key(key, hold)
            self.sleep(0.4 + self.ui_delay)
            self.next_frame()

    @staticmethod
    def _pan_keys_for(method, dx, dy):
        """方向 -> [(键, 按住多久)]。按住时长和想挪的距离成正比。"""
        keys = PAN_KEY_SETS.get(method, {})
        out = []
        for axis, delta in (('x', dx), ('y', dy)):
            pair = keys.get(axis)
            if not delta or not pair:
                continue
            out.append((pair[1] if delta > 0 else pair[0],
                        PAN_KEY_HOLD * abs(delta) / PAN_STEP))
        return out

    def _try_pan(self, method, dx, dy, center=None):
        """挪一下并**验证真的挪动了**。每一次的实测位移都打到日志里（以前是 debug，看不到）。

        :return: (动了吗, dx像素, dy像素)
        """
        before = self._fresh_roi()
        self._pan_once(method, dx, dy, center)
        moved, mdx, mdy, resp = self._moved(before, self._fresh_roi())
        self.log_info('平移[%s] 目标(%+.2f,%+.2f) -> 实测位移(%+.1f,%+.1f) 置信度%.3f %s'
                      % (PAN_METHOD_NAMES.get(method, method), dx, dy, mdx, mdy, resp,
                         '动了' if moved else '没动'))
        if moved:
            # 镜头动了，之前点过的屏幕坐标对应的已经是别的地方了
            self._reset_tile_tries()
        if method == 'right_drag':
            # 右键按下的地方可能正好站着人形。游戏里右键拖拽会不会动到人形没实测过，
            # 保险起见挪完看一眼部署数：变了（多了或少了）就立刻停手，别再挪了
            now, ref = self.read_deploy_count(), getattr(self, '_count_ref', None)
            if now and ref and now[1] == ref[1] and now[0] != ref[0]:
                self.log_info('右键拖拽之后部署数从 %d 变成了 %d，停止挪镜头' % (ref[0], now[0]),
                              notify=True)
                self._lineup_changed = True
        return moved, mdx, mdy

    def detect_pan_method(self, axis='x'):
        """挨个试出**这个轴上**哪种方式能真的挪动镜头，记住它。

        **两个方向都试**：只试正方向的话，镜头刚好顶在那一侧的地图边界上，
        就会把「这个轴能动」误判成「不能动」—— 以前「WASD 只能上下、不能左右」就是这么来的。
        **只记住成功的**：失败不缓存，下一个部署界面（换了地图、换了镜头位置）还会再试。

        :param axis: 'x' 横向 / 'y' 纵向
        :return: 方式名；这个轴上怎么都挪不动就返回 None
        """
        cache = getattr(self, '_pan_method', None)
        if not isinstance(cache, dict):
            cache = {}
            self._pan_method = cache
        if cache.get(axis):
            return cache[axis]

        methods = PAN_METHODS_X if axis == 'x' else PAN_METHODS_Y
        label = '横向' if axis == 'x' else '纵向'
        for method in methods:
            for sign in (1, -1):
                step = (sign * PAN_STEP, 0) if axis == 'x' else (0, sign * PAN_STEP)
                moved, _, _ = self._try_pan(method, *step)
                if getattr(self, '_lineup_changed', False):
                    return None
                if moved:
                    # 原路挪回去，别给后面的搜索留一个偏移
                    self._try_pan(method, -step[0], -step[1])
                    cache[axis] = method
                    self.log_info('%s可以用「%s」挪动，后面就用它'
                                  % (label, PAN_METHOD_NAMES[method]), notify=True)
                    return method
            self.log_info('「%s」%s两个方向都挪不动，换下一种' % (PAN_METHOD_NAMES[method], label))
        self.log_info('%s这次怎么都挪不动镜头' % label, notify=True)
        return None

    def _search_by_panning(self):
        """一个方向一个方向地挪过去找空地格，每挪一步都验证真的动了。

        每一步都**重新截图重新识别**，绝不沿用挪动前的坐标。
        某个方向挪不动了（多半是到地图边界了）就不在这个方向上死磕，直接换下一个。
        横纵两个轴用各自探测出来的方式 —— 可能一个能用一个不能用。
        """
        mx = self.detect_pan_method('x')
        my = self.detect_pan_method('y') if not getattr(self, '_lineup_changed', False) else None
        if getattr(self, '_lineup_changed', False):
            return False
        if mx is None and my is None:
            self.log_info('横纵都挪不动镜头，放弃平移搜索'
                          '（可以自己先把视角拖到能看见空地格的位置）', notify=True)
            return False

        dirs = []
        if mx:
            dirs += [('右', mx, PAN_STEP, 0), ('左', mx, -PAN_STEP, 0)]
        if my:
            dirs += [('下', my, 0, PAN_STEP), ('上', my, 0, -PAN_STEP)]

        for label, method, dx, dy in dirs:
            steps_done = 0
            give_up = False
            for step in range(PAN_STEPS_PER_DIR):
                self.log_info('镜头往%s挪第 %d/%d 步（%s）'
                              % (label, step + 1, PAN_STEPS_PER_DIR,
                                 PAN_METHOD_NAMES[method]))
                moved, _, _ = self._try_pan(method, dx, dy)
                if getattr(self, '_lineup_changed', False):
                    return False
                if not moved:
                    self.log_info('往%s挪不动了，多半到地图边界，换个方向' % label)
                    break
                steps_done += 1
                if self._try_click_candidates(limit=MOVED_VIEW_CLICKS):
                    return True
                if getattr(self, '_lineup_changed', False):
                    return False
                if getattr(self, '_tile_click_budget', 999) <= 0:
                    give_up = True
                    break
            # 原路挪回来：下一个方向要从原位出发；放弃搜索时也要挪回来 ——
            # 后面「先选人再找地格」要在原来的视野里找
            for _ in range(steps_done):
                self._try_pan(method, -dx, -dy)
            if give_up:
                return False
        return False

    def _try_click_empty_tiles(self, limit=MAX_TILE_CLICKS_PER_VIEW):
        """旧名字，保留给外部调用；逻辑见 _try_click_candidates。"""
        return self._try_click_candidates(limit=limit)

    def pick_empty_tile(self, move_view=True):
        """第 1 步：在地图上选中一个空地格。

        顺序：当前画面的四级候选 -> 缩放（滚轮真能用时）-> 四个方向平移。
        每点一下都读右下角四态，只有「部署人形」才算选中。
        整个过程总共最多点 TILE_CLICK_BUDGET 下。

        :param move_view: 当前画面里找不到/点不中时，是否挪动视角再找
        """
        self._tile_click_budget = TILE_CLICK_BUDGET
        self._tile_near_saved = False           # 每次找地格最多存一张「只差一点够格」的图
        if self.tile_state(rereads=0) == TILE_EMPTY:
            self.log_info('已经有选中的空地格，不用重选')
            return True
        if self._try_click_candidates():
            return True
        if getattr(self, '_lineup_changed', False):
            return False
        # 挪镜头之前这个视野点过的位置：镜头挪回原位时，最后那次复查不再点它们
        first_view = list(getattr(self, '_tried_tiles', ()))
        if not move_view:
            if self._final_view_retry(first_view):
                return True
            self.log_info('当前画面没有能点中的空地格')
            return False

        try:
            # 先缩放：滚轮不可能被理解成部署手势，也最可能一次把所有地格露出来
            if self._zoom_search():
                return True
            if getattr(self, '_lineup_changed', False):
                return False
            # 再四个方向大幅平移
            if self._search_by_panning():
                return True
        except PROPAGATE:
            raise
        except Exception as e:
            self.log_error('挪视角找地格时出错', e)
        if not getattr(self, '_lineup_changed', False) and self._final_view_retry(first_view):
            return True
        self.log_info('缩放 + 四个方向平移都试过了，仍然没找到能点的空地格', notify=True)
        return False

    def _final_view_retry(self, skip=()):
        """最后这个视野连看 FINAL_VIEW_FRAMES 帧，把这几帧里冒出来、还没点过的 hole / blob / tile 候选点一遍（I-47）。

        hole / blob 是靠空格外圈那道蓝边认的，边的色相在门槛边上时一帧有一帧没有，所以多看几帧取并集。
        :param skip: 挪镜头之前点过的屏幕坐标。镜头挪回原位时它们还是那几格，排到最后再点；**不跳过** ——
                     平移回程常常回不到原位（SF-1-2 差了约 (-230,-140)px），那时这些坐标落在别的格子上，
                     当时就有一个正好落在真空格里，跳过的话复查就白做了（I-47 复查）
        :return: True = 点中了空格
        """
        tol = 0.03 * self.width
        picked, later = [], []
        for k in range(FINAL_VIEW_FRAMES):
            if k:
                self.sleep(FINAL_VIEW_GAP)
            self.next_frame()
            for c in self.tile_candidates():
                if (c.name or '').split(' ')[0] not in FINAL_VIEW_KINDS:
                    continue
                cx, cy = c.x + c.width // 2, c.y + c.height // 2
                if any(abs(cx - (d.x + d.width // 2)) < tol and abs(cy - (d.y + d.height // 2)) < tol
                       for d in picked + later):
                    continue
                (later if any(abs(cx - x) < tol and abs(cy - y) < tol for x, y in skip) else picked).append(c)
        picked += later
        if not picked:
            return False
        self.log_info('最后这个视野连看 %d 帧，还有 %d 个没点过的 hole / blob / tile 候选，先点它们：%s' % (
            FINAL_VIEW_FRAMES, len(picked), '; '.join('(%d,%d)%s' % (c.x + c.width // 2, c.y + c.height // 2,
                                                                     c.name.split(' ')[0]) for c in picked)))
        if getattr(self, '_tile_click_budget', 999) < FINAL_VIEW_CLICKS:
            # 总数快用完了也给这几下：它们是这一轮唯一没试过的好候选
            self._tile_click_budget = FINAL_VIEW_CLICKS
        return self._try_click_candidates(limit=min(FINAL_VIEW_CLICKS, len(picked)), extra=picked)

    def deploy_selected_unit(self, before=None):
        """最后一步：点右下角「部署人形」，把刚选好的人放到刚选好的地格上。

        - 先看右下角是不是「部署人形」按钮：是「撤回人形 / 编入队伍 / 请点击地格」就立刻判
          这一格放不了（以前要白等 6 秒找按钮）。
        - 计数基准在按按钮**之前**重读一次（入口那次可能已经过时了）。
        - 按完隔 0.6s 读三次：部署数涨了算成功；计数读不到时，按钮消失也算成功。
        """
        ref = self.read_deploy_count() or before
        if ref and ref[0] >= ref[1]:
            self.log_info('阵容已满 %d/%d，不再部署' % ref)
            return True
        st = self.tile_state(rereads=1)
        if st in (TILE_HINT, TILE_OCCUPIED):
            # 游戏明确说了「没选中 / 选中的是站着人的格子」—— 不用再等按钮
            self.log_info('右下角是「%s」%s，没有「%s」按钮，这一格放不了'
                          % (TILE_STATE_NAMES[st], getattr(self, '_last_br', []), DEPLOY_BTN))
            return False
        # 读不出来时不下结论，交给下面的 wait_click_ocr 再等一会儿按钮

        # 「部署人形」在顶部也有一个（那是计数标签），所以只在右下角找按钮
        note = self._press_context()
        if not self.wait_click_ocr(match=DEPLOY_BTN, box=self.box.bottom_right, time_out=4,
                                   after_sleep=0, raise_if_not_found=False, log=True):
            self.log_info('右下角没找到「%s」按钮，跳过部署' % DEPLOY_BTN)
            return False
        frames = self._press_wait(1.0 + self.ui_delay)

        after = None
        for i in range(3):
            if i:
                self.sleep(0.6)
            self.next_frame()
            after = self.read_deploy_count()
            if ref and after and after[0] > ref[0]:
                self.log_info('助战已部署：部署人形 %d/%d -> %d/%d'
                              % (ref[0], ref[1], after[0], after[1]))
                return True
        st = self.tile_state(rereads=0)
        if not (ref and after) and st != TILE_EMPTY:
            self.log_info('读不到部署计数，但「%s」按钮已消失（右下角=%s），按已部署处理'
                          % (DEPLOY_BTN, getattr(self, '_last_br', [])))
            return True
        self.log_info('按了「%s」但没放上去：部署 %s -> %s，右下角=%s'
                      % (DEPLOY_BTN, '%d/%d' % ref if ref else '?', '%d/%d' % after if after else '?',
                         getattr(self, '_last_br', [])))
        self._report_press_frames(frames, note)
        return False


class SupportBattleMixin(SupportPickMixin):
    """在「作战开始」被按下之前插入借助战的战斗混入。

    ``BaseGfTask.auto_battle`` 靠 ``skip_dialogs`` 等到「作战开始」后立刻点它，
    所以这里重写 ``skip_dialogs``：它一旦返回「作战开始」，就说明正停在部署界面，
    此时插入借助战，再重新定位一次「作战开始」交还给父类。

    子类要做两件事：
      1. 在 ``__init__`` 里调用 ``self._init_support_state()``
      2. 实现 ``support_options()``，返回 ``borrow_support`` 的参数字典；
         ``target_name`` 为空表示这次不借
    """

    START_BATTLE = '作战开始'

    def _init_support_state(self):
        self._support_done = False
        self._battle_depth = 0
        self._support_power_hint = None
        self._ui_delay = 1.0
        self._tap_to_reveal_skip = True
        self._move_view_for_tile = True
        self._tile_search_failures = 0
        self._tried_tiles = []
        self._skip_borrow_once = False
        self._pan_method = {}        # {'x': 方式, 'y': 方式}，只记实地试成功的
        self._pan_hann = None
        self._wheel_fail = 0         # 连着几个部署界面滚轮两个方向都没反应
        self._count_ref = None       # 本轮部署界面开始时的「部署人形 x/y」
        self._lineup_changed = False  # 点地格时人数自己变了（游戏直接放上了人形）
        self._tile_pending = False   # 「先选人再找地格」那条路
        self._picked_power = None    # 选中的助战卡的作战效能（重新选回来时用）
        self._picked_target = None
        self._badge_cache = None

    def support_options(self):
        raise NotImplementedError

    def auto_battle(self, *args, **kwargs):
        # 每场战斗（含「继续前进」触发的下一阶段）都重新给一次借助战的机会。
        # 但 auto_battle 会因「继续前进」自我递归，深度 > 1 说明本关前面的阶段已经打赢了，
        # 这时再「退出重进」会把战果一起丢掉 —— 所以后续阶段只借人、不退关。
        self._support_done = False
        self._battle_depth += 1
        try:
            return super().auto_battle(*args, **kwargs)
        finally:
            self._battle_depth -= 1

    def _skip_with_reveal(self, end_match, end_box, time_out, has_dialog, raise_if_not_found):
        """把父类的 skip_dialogs 切成若干段，每段之间点一下屏幕中央。

        有些过场动画要先点一下屏幕，「跳过」按钮才会冒出来；父类只会盲点右上角，
        按钮没出来时那一下是空的。分段的好处是：正常快速推进时一段之内就结束了，
        这下额外点击根本不会触发；只有真的卡在动画里超过一段才补点。
        """
        if not getattr(self, '_tap_to_reveal_skip', True):
            return super().skip_dialogs(end_match, end_box=end_box, time_out=time_out,
                                        has_dialog=has_dialog,
                                        raise_if_not_found=raise_if_not_found)
        left = time_out
        while left > 0:
            chunk = min(SKIP_CHUNK_SECONDS, left)
            result = super().skip_dialogs(end_match, end_box=end_box, time_out=chunk,
                                          has_dialog=has_dialog, raise_if_not_found=False)
            if result:
                return result
            left -= chunk
            if left > 0 and has_dialog:
                self.log_debug('动画里卡了 %ds，点一下屏幕把「跳过」唤出来' % chunk)
                self.click_relative(*SKIP_REVEAL_POINT, after_sleep=0.4 + self.ui_delay,
                                    name='tap_to_reveal_skip')
                self.next_frame()
        if raise_if_not_found:
            raise Exception('跳过剧情超时!')
        return None

    def skip_dialogs(self, end_match, end_box=None, time_out=120, has_dialog=True,
                     raise_if_not_found=True):
        result = self._skip_with_reveal(end_match, end_box, time_out, has_dialog,
                                        raise_if_not_found)
        if not self._should_borrow(result):
            return result

        self._support_done = True
        opts = dict(self.support_options() or {})
        if self._battle_depth > 1:
            # 「继续前进」之后的阶段：只借人，绝不退关
            opts['reenter_times'] = 0
        try:
            self.borrow_support(**opts)
        except (SupportStateError, TileNotSelectedError):
            # 界面已经不在部署界面了 / 地格怎么都选不中：
            # 继续跑只会乱点或者少人开打，直接把任务停在这里，交给调用方处理
            raise
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            self.log_error('借助战出错，按原流程继续', e)

        # 借人过程中界面动过，重新定位一次「作战开始」再交还给父类。
        # 重新定位失败就退回借人之前那次的结果：auto_battle 会直接 result[0].name 取值，
        # 返回 None 会让它抛 TypeError；而「作战开始」是固定按钮，旧坐标照样点得中。
        relocated = self._skip_with_reveal(end_match, end_box, 30, has_dialog, False)
        return relocated if relocated else result

    def _should_borrow(self, result):
        # 用户手动放好人之后替他按「作战开始」的那一路：人已经在场上了，别再借一次
        if getattr(self, '_skip_borrow_once', False):
            self._skip_borrow_once = False
            self.log_info('这一场是你手动放的人，跳过借助战')
            return False
        if getattr(self, '_support_done', False):
            return False
        opts = self.support_options() or {}
        if not (opts.get('target_name') or '').strip():
            return False
        return bool(result) and any(getattr(b, 'name', '') == self.START_BATTLE for b in result)
