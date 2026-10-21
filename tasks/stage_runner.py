# -*- coding: utf-8 -*-
"""一关从进本到结算：逐帧状态机（ok-gf2 扩展）。

和 support_pick.py 一样，本文件**不定义任何 BaseTask 子类**，只提供一个混入 ——
ok 的任务扫描器会给 ok_import 下每个 .py 找第一个 BaseTask 子类注册成任务，这里没有。

为什么不再调上游的 auto_battle / skip_dialogs（2026-09-29 用户反馈 + 当天日志）：

  1. 「手动接管后一直点右上角跳过」：上游 skip_dialogs 在「什么都没认出」时只要 has_dialog=True
     就盲点右上角 (0.95, 0.04) 再睡 2 秒，循环。auto_battle 等「任务完成」那一步 time_out=900 ——
     整场战斗、以及你手动退出 / 手动操作之后，它每 ~2.5 秒盲点一次右上角，最长 15 分钟。
     再叠上 SupportBattleMixin 每 8 秒一段、段间点一下屏幕中央：9/29 10:42:44~10:51:13
     在认不出的画面上连点了 22 次中央（最后是你手动关掉任务），11:00:49~11:04:51
     一场普通战斗里点了 21 次中央。
  2. 9/29 10:45:23：auto_battle 第一次 skip_dialogs 120 秒没等到「作战开始」等，返回 None，
     上游 result[0] 直接 TypeError。
  3. 慢：盲点后睡 2 秒再 OCR，点出来 1~2 秒就淡出的「跳过」已经没了；唤出跳过的点击 8 秒才补一次；
     结算后一串固定等待（10:40:58「任务完成」到 10:41:18「本关完成」用了 20 秒）；
     剧情关 10:41:23 点「观看」到 10:41:51 才第一次点中「跳过」。上游 _is_loading_frame 还把
     「一个字都没有」当成加载，没字幕的纯过场永远不会去点。

这里的做法：每轮只截一帧、全屏 OCR 一次，按固定优先级认界面，**每轮只做一个动作**：

    今日不再提示 > 顶部「跳过」> 还有可部署 > 继续前进 > 胜负 > 作战开始(借人) > 战斗 HUD
    > 关卡里你在用的菜单 / 面板（不动）> 确认 > 画面中部的「跳过」> 弹窗 > 关卡外（返回）
    > 加载中 > 认不出（盲点策略）

盲点策略：**彻底不再盲点右上角**。唯一的盲点是点一下屏幕中央唤出「跳过」，点完 0.35 秒就只 OCR
顶部一条去找它；只在像过场 / 对话的画面上点（字块少），战斗画面和字多的界面一律不点；
连点没进展就停手只看。你动键鼠的时候（游戏在前台、2.5 秒内有真人输入）什么都不点。
"""

import ctypes
import difflib
import os
import re
import time

import cv2
import numpy as np

from src.core.BaseGfTask import map_re, pop_ups

from stage_select import is_chapter_map
from chapter_nav import story_list_info
try:
    import close_x                     # 右上角的关闭叉（教学 / 引导，I-13）
except ImportError:                    # 打包漏了它：只是认不出叉，别让整个逐帧状态机起不来
    close_x = None
from support_pick import (SKIP_REVEAL_POINT, STOP_EXCEPTIONS, OverlayBlockedError, SupportStateError,
                          TileNotSelectedError, UserTakeoverError, overlay_caption, prune_debug_dir)
from own_units import LineupShortError
try:
    import input_watch                 # 分「点了游戏画面」和「只是看看」（I-52）
except ImportError:                    # 打包漏了它：回到老办法（有输入就当你在操作）
    input_watch = None
from stamina_refill import (confirm_button, icon_point, parse_count, quantity, refill_title, selected_slot,
                            slot_stock, stamina)

# ---- 总体 ----
RUN_STAGE_TIMEOUT = 40 * 60        # 一关的总时长上限（秒），到了就交还给轮询
# 连续这么久一直认不出画面（含 0 个字块）就停手：存一张现场图、交还给轮询。
# 战斗里认不出的帧（敌方回合、技能动画）和 HUD 帧交替出现，见到 HUD 就重新计时。
STALL_SECONDS = 150
# 开打之后、结算之前放宽到和上游等「任务完成」一样的 900 秒：9/25~9/29 四天日志里轮询一次都没认出过
# 「战斗中」（行动结束 / 行动完成），9/29 11:00:40 那场自动战斗 4 分 20 秒从头到尾都是认不出的帧。
# 原来的 300 秒会把正常的长战斗判成卡住（复核模拟：10 分钟的自动战斗在 300 秒处 stalled、存图、
# 发通知，还把战斗画面交给了轮询）。战斗里盲点有上限，多等只是不点，没有副作用
BATTLE_STALL_SECONDS = 900

# ---- 跳过 ----
SKIP_TEXT = '跳过'
# 「跳过」按钮在最上面：9/25~9/29 日志里点过 354 次「跳过」，350 次 y ≤ 80px（3200x1800 / 3072x1728）；
# 另外 4 次在画面中部，单独处理（见 _sr_step 的 8b）
SKIP_TOP_RATIO = 0.12
# 点完屏幕中央只 OCR 右上这一条：按钮 x 在 0.80~0.96W；比整屏 OCR 快，缩放也小
SKIP_BAND = (0.5, 0.0, 1.0, SKIP_TOP_RATIO)
SKIP_AFTER_SLEEP = 0.5
# 同一处（同名 + 屏幕 5% 网格）3 秒内最多点 2 次；连着点了 6 次它还在就先不点它
SAME_SPOT_WINDOW = 3.0
SAME_SPOT_MAX = 2
SAME_TARGET_GIVE_UP = 6
# 连点的那个按钮要**连续**消失这么久才算「点掉了」、连点计数清零。只看一帧不行：灰掉 / 点不动的
# 「跳过」OCR 时有时无，一帧读不到就清零的话上限永远到不了（2026-09-29 复核模拟：40 分钟点了 1326 次，
# 正是「一直点右上角跳过」）。9/25~9/29 日志里同一处连着点「跳过」（中间没点别的）≥3 次只有一回：
# 9/27 13:42:38 起 6 次、每次隔 4 秒 —— 是跳过确认框挡在前面、上游一直点后面变暗的「跳过」，不是正常推进
STREAK_GONE_SECONDS = 2.0
SPOT_GRID = 0.05

# ---- 各种按钮 ----
NO_ALERT = '今日不再提示'           # 跳过剧情的确认框：勾上它再点确认
CONFIRM_NAMES = ['确认', '确定']
NO_ALERT_SLEEP = 0.3
LINEUP_NOT_FULL_RE = re.compile('还有可部署')   # 阵容没满时点「作战开始」弹的确认框
CONTINUE = '继续前进'
CONTINUE_DEDUP = 4.0                # 「继续前进」点完还会在画面上留 1~2 秒（9/29 10:17:09 / :11），别算成两个阶段
VICTORY_NAMES = ['任务完成', '对战胜利']
DEFEAT_NAMES = ['任务失败', '战斗失败', '对战失败']
START_BATTLE = '作战开始'
START_SETTLE = 0.6                  # 刚进部署界面先等它稳住再点（上游固定睡 2 秒）
START_RETRY_SECONDS = 4.0           # 点了「作战开始」这么久之内不重复点，给界面反应时间
# 点完之后画面上冒出了新的字（敌人名字之类）= 游戏已经在进战斗了，只是「作战开始」还没消失：再多等这么久。
# 9/29 19:37:20、19:41:56 ZF-1-1：点完 4 秒还读得到「作战开始」，同时已经冒出「鳄面马鹿·酸蚀I」，
# 于是在正在进战斗的画面上又点了一下
START_TRANSITION_EXTRA = 6.0
START_CLICK_LIMIT = 4               # 同一阶段点了这么多次还在部署界面 = 点不动，停手
# 点完「作战开始」游戏弹的缺人提示（10/01～10/10 七次，原文一字不差「本关至少需要出击4人！」，一闪就没，不到 10 秒）。
# 只认「需要出击…人」：同一批新冒出来的字里还有关卡目标「至少撤离4个我方单位」「前2回合击杀至少3个敌方单位」
MIN_SORTIE_RE = re.compile(r'至少需要出击\s*(\d{1,2})\s*人')
OWN_FILL_ROUNDS = 2                 # 同一阶段最多补几轮（补完点「作战开始」又弹 = 部署数读不到、补少了）
HUD_RE = re.compile('行动结束|行动完成')   # 战斗中右下角的回合按钮（上游 need_click_auto 在 bottom_right 找它）
HUD_POLL = 1.2
# 自动战斗的 HUD 上从来没有「行动结束 / 行动完成」（9/30～10/02 三天日志里 0 次），能稳定读到的是左上角的回合计数
# 「101/12回合」「01/10 回合」（x<15%W、y 在 6%～16%H）和回合横幅。10/02 你接手开打之后，轮询把整场战斗、战后的剧情
# 对话、敌人介绍弹窗都认成「关卡外」，一下不点，三次共等了你约 24 分钟（I-24）。
# 对 9/29～10/02 所有现场图的整屏 OCR 跑过：战斗帧 59/59 命中，章节地图 / 选关 / 部署 / 结算 / 教学等 66 张 0 误判；
# 关卡详情面板上的「在6回合以内获胜」在右侧、没有斜杠，整块匹配不会中
ROUND_RE = re.compile(r'^\W{0,2}\d{1,3}\s*/\s*\d{1,3}\s*回合\s*$')
ROUND_BAND = (0.0, 0.06, 0.15, 0.16)
TURN_BANNER_RE = re.compile('我方行动开始|敌方行动开始')
# 战斗里的通讯对话：左上角「通信中 / 闪电 / 一句话」，要点一下才继续打（你 10/2 说的）。没有回合计数，以前被认成「关卡外」；
# 见过 HUD 之后又只在 ≤2 个字块的画面上点，这种 3 块字的点不到。单独认出来：点屏幕中央翻下一句，每阶段最多 COMM_TAP_CAP 下
COMM_RE = re.compile('通信中')
COMM_BAND = (0.0, 0.0, 0.35, 0.25)
COMM_TAP_CAP = 12
COMM_TAP_WAIT = 1.0
COMM_TAP_INTERVAL = 2.0
# 战斗里的「通信中」对话还有一种是头像在屏幕下面、对话框右下角一个「▶| 跳过」（10/12 00:14：头像在左边时跳过在 (0.52W, 0.95H)，
# 在右边时 (0.76W, 0.95H)）。点它整段过去，比一句句点中间快；顶部 / 中部的「跳过」规则都管不到这么低（I-44）
SKIP_BOTTOM_Y = 0.85
SKIP_BOTTOM_X = (0.30, 0.85)
SKIP_BOTTOM_RE = re.compile(r'^\W{0,2}[Il|]?\s*跳过$')     # 「▶|」有时读成「I」粘在前面
COMM_LABEL_RE = re.compile('通信')                       # 头像旁边的「通信中」（有时只读出「通信」）
COMM_LABEL_Y = (0.70, 0.88)
# 头像在下方的通讯对话，「通信」标签在头像旁边：左下 (0.07W, 0.78H) 或右下 (0.86W, 0.78H)，x 在这两条线外面。
# 没有「跳过」的那种（10/11 12:58 131319_stuck）以前不算通讯对话，按战斗里的盲点点满 3 下就停，卡了 15 分钟、交还后轮询又当成
# 「关卡外」（I-48）。593 份现场记录里这个位置有「通信」的只有这 3 张真的通讯对话
COMM_LOW_X = (0.20, 0.75)
# 见过 HUD 之后点出了新台词（像对话）的那一下，不算进战斗里的 3 下（BATTLE_TAP_CAP）；退还的总数封顶 COMM_TAP_CAP。
# 下方那种对话开头几句常常读不到「通信」标签（12:58 那 3 下点之前的帧就是），只靠认标签会卡在第 3 下（I-48 复查）
# 通讯对话的 12 下是「一段对话」的上限：重新看到 HUD、离上次点通讯对话又过了这么久，就清零（一个阶段里可能有好几段）
COMM_RESET_AFTER = 10.0
# 部署界面上盖着教学 / 提示（I-43）：有叉点叉，没叉点中间翻页，最多这么多下
DEPLOY_OVERLAY_TAP_CAP = 15
DEPLOY_OVERLAY_TAP_INTERVAL = 1.5
# 关卡里、但**你在用**的菜单 / 面板：☰「关卡信息」（9/29 10:34:48 实测有「关卡描述」「特殊奖励」「退出作战」，
# 版式和地图上的关卡详情一样，顶上有关卡编号 —— is_chapter_map 会把它认成章节地图），
# 以及部署界面被你打开的面板挡住了「作战开始」（顶上也有关卡编号）。什么都不点，也不算回到关卡外
IN_STAGE_MARKERS = ['退出作战', '继续作战', '部署人形', '撤回人形', '好友助战', '我的人形']
SETTLE_CONFIRM = ['确认', '确认结算']
CONFIRM_MIN_Y = 0.4                 # 结算的「确认」在下半屏（9/29 实测 y=1639/1800）
# 下半屏的「确认」以前见到就点 —— 10/1 11:30 情报拼图不够，点「挑战」弹出「情报拼图补充 · 快速取出调度室储存情报拼图 ·
# 取出数量 0 · 取消 / 确认」，被当成结算确认连点 6 次，退回去又点挑战，循环了 8 分钟（数量是 0 才没花掉东西）。
# 现在开打前只点「刚点过跳过」之后弹的那个确认（跳过剧情的确认框）；结算之后照旧。消耗 / 购买类的框一律不点，停手提醒
SKIP_CONFIRM_WINDOW = 8.0           # 点「跳过」之后这么久之内弹出的「确认」算跳过确认框
RESOURCE_RE = re.compile(r'情报拼图补充|快速取出|取出数量|是否消耗|购买|兑换')
# 情报拼图不够：「情报拼图补充」框里选中间那格（补充道具）点「确认」补上（你 10/11 说的；认格子见 stamina_refill）。
# 右边那格是晶体货币，一下都不点；别的消耗 / 购买框照旧不点、停手
REFILL_PICK_SLEEP = 1.0             # 点中间那格之后等选中框挪过去
REFILL_CONFIRM_SLEEP = 1.2          # 点「确认」之后等框关掉
REFILL_CLOSE_TRIES = 3              # 框关得慢：最多再看这么多次（每次隔 REFILL_CLOSE_WAIT）
REFILL_CLOSE_WAIT = 0.8
REFILL_GUARD = 180.0                # 试过补之后这么久里又弹补充框（补完还不够 / 没生效 / 框没关）：不再补，照旧停手
ENTER_STAGE = ['挑战', '观看']
EXIT_CONFIRM_ROUNDS = 2             # 没看到胜负就回到关卡外：连着两帧都这样才算（面板滑出的那一下不算）
EXIT_RECHECK = 0.15                 # 那一帧之后多快再看一眼
EXIT_MIN_SECONDS = 3.0              # 一进来就看到关卡外、关卡里什么都没见过：至少等这么久（点「挑战」可能还没生效）
AFTER_VERDICT_EXIT_SECONDS = 5.0    # 看到胜负之后，字多又认不出的界面持续这么久 = 回到了关卡外（活动地图等）
# 加载画面：「…加载中」或者单独一个百分比（进度数字）。以前任何带「数字%」的字都算，
# 画面上别处带个百分比就一直当加载、一直不点
LOADING_RE = re.compile(r'加载中|^\s*\d{1,3}\s*%\s*$')
# 「加载中」连续这么久还没完：按认不出的画面处理（点加载画面无害，也许根本不是加载）。
# 9/29 12:47:30 剧情关 SL-3-5 进来就判成「加载中」，之后 27 秒一下没点，最后是你自己点过去的；
# 正常的加载 9/29 实测 6~8 秒（12:44:21、12:46:33、12:48:14、12:48:54）
LOADING_MAX = 12.0

# ---- 盲点（点屏幕中央唤出「跳过」）----
TAP_MAX_BOXES = 8                   # 字块比这多的画面是菜单 / 你打开的界面，不点
# 对话画面例外：台词、人名、选项加起来常常超过 8 个字块，按「字多 = 菜单」不点的话
# 多段剧情只跳得掉第一段（用户反馈：剧情关要跳好几次、有时要点一下画面才出「跳过」）。
# 剧情关（点了「观看」）里认不出的基本就是剧情；别的关卡里只认「字都在下半屏」的对话框布局
STORY_TAP_MAX_BOXES = 30
DIALOG_TAP_MAX_BOXES = 30
DIALOG_LOW_RATIO = 0.6              # 字块中心在这条线以下算「在对话框里」
DIALOG_LOW_SHARE = 0.7              # 除去顶部按钮条，至少这么大比例的字块在对话框里
# 一关里第一次连续这么久「认不出、字也没变」：存一张图、记下读到的字（排查用）。开打之后不存 ——
# 自动战斗的画面本来就一直认不出（9/29 四天日志里战斗 HUD 一次都没认出来过），存了也是每关一张废图
STUCK_UNKNOWN_SAVE = 20.0
TAP_TEXT_WAIT = 1.0                 # 有字的认不出画面，持续这么久才点
# 一个字都没有（无字幕过场 / 加载）持续这么久才点。点加载画面无害；再长的话，算上每轮 ~0.2s 的
# OCR 和按钮 1 秒就淡出，从过场开始到点中「跳过」会超过 3 秒
TAP_ZERO_WAIT = 1.5
TAP_INTERVAL = 1.5
TAP_SETTLE = 0.35                   # 点完多久去找「跳过」：按钮点出来 1~2 秒就淡出
TAP_GIVE_UP = 6                     # 连点这么多次都没进展（画面几乎没变、也没认出任何东西）就只看不点
# 见过战斗 HUD 之后（真在打了）：只在几乎没字、认不出好几秒的画面（战斗里的过场）点，每阶段最多 3 次
BATTLE_TAP_MAX_BOXES = 2
BATTLE_TAP_WAIT = 3.0
BATTLE_TAP_CAP = 3
# 点了「作战开始」、还没见过 HUD：开场过场 / 对话。9/29 10:40:05 点完作战开始，33 秒后
# 才点中「跳过」—— 这段要能唤出跳过，但比开打前克制一些
INTRO_TAP_WAIT = 2.0
INTRO_TAP_INTERVAL = 3.0
INTRO_TAP_CAP = 6
THUMB_SIZE = (64, 36)
THUMB_CHANGE = 4.0                  # 缩略图平均灰度差超过它算「画面变了」
# ---- 战斗里弹出的教学 / 引导（ISSUES.md I-13）----
# 9/30 4-9：点完「作战开始」弹出教学，要点屏幕中间一页页往下翻、翻完右上角出叉、点叉才开打。开场盲点每阶段
# 最多 6 下（INTRO_TAP_CAP），点完就不点了，第 1 / 2 阶段各停了 3 分钟 / 9 分钟等你。但这 6 下的上限不能放宽：
# 战斗 HUD 在所有日志里一次都没认出来过，整场自动战斗都处在「开场」模式（37 次点满 6 下里 33 次是正常战斗）。
# 所以只在「像教学」的画面上多点：游戏暂停了（不点的时候画面一动不动），而且每点一下画面都换成新的内容。
TUTORIAL_TAP_CAP = 15               # 开场 6 下用完之后，教学画面上每阶段最多再点这么多下（静止阈值还没在游戏里校准）
TUTORIAL_TAP_INTERVAL = 2.0
STATIC_WINDOW = 1.5                 # 最近这么久里的画面变化样本都很小 = 画面停着
STATIC_MIN_SAMPLES = 2
STATIC_MEAN = 0.6                   # 64x36 灰度缩略图逐像素差的均值
STATIC_FRAC = 0.004                 # 差超过 STATIC_PIX 的像素占比
STATIC_PIX = 12.0
MOTION_GAP_MAX = 2.5                # 两帧隔太久就不算一个样本
MOTION_KEEP = 30
OVERLAY_NORESP_STOP = 2             # 连着这么多下点了没换新内容（或者没停稳）：不再多点
# 游戏自己在动（不是刚点出来的变化）= 没被教学暂停，在正常打：这一阶段先不多点（复查 BS-2：自动战斗里偶尔停 1.5 秒
# 的空档会被当成教学，一下有反应就能一直点到 15 下）。点完自己这一下之后这么久之内的变化算点出来的，不算
OWN_ECHO = 2.0
# 不多点之后，画面又一动不动停了这么久（游戏像是又被暂停了，比如打到一半才弹教学）：再试一下，有反应才接着点
OVERLAY_REARM_STILL = 5.0
OVERLAY_REARM_CAP = 2               # 每阶段最多这样重新试几次
PRE_DEPLOY_REFUND_CAP = 40          # 「继续前进」之后、部署之前的对话：点出进展（跳过 / 换了一句）的那下不算进开场 6 下
# 右上角的关闭叉：只在这个带子里认、只点认出来的位置，绝不盲点右上角（9/29 你说过「一直点右上角」）
CLOSE_X_BAND = (0.80, 0.0, 1.0, 0.20)
CLOSE_X_PHASE_CAP = 3               # 每阶段最多点几次叉；整关最多 CLOSE_X_RUN_CAP 次
CLOSE_X_RUN_CAP = 6
CLOSE_X_SAME = 0.015                # 两帧里的叉离得这么近（x 屏宽）算同一个
CLOSE_X_MIN_GAP = 0.3               # 两次看到要隔这么久，才算「连着两帧都在」
CLOSE_X_PENDING = 1.5               # 等第二帧确认的这段时间不点中间（不然叉还没确认就被翻页了）
CLOSE_X_RECHECK = 3.0               # 点完这么久之内叉还在原处、画面也没变 = 点了没用
CLOSE_X_NOEFFECT_STOP = 2
USER_RECENT_GUARD = 10.0            # 你停手不到这么久时不点叉：你自己打开的面板上的叉是你的
X_TOKEN_RE = re.compile(r'^\s*[xX\u00d7\u2715\u2716\u2573]\s*$')   # OCR 偶尔把叉读成一个「X」
# 认不出的画面上「字变了」怎么算（见 text_skeleton）：和最近这么多帧读到的字比
SIG_HISTORY = 6
SIG_SIMILAR = 0.8                   # 相似度到这个数（短字块放宽到只差一个字）算同一块字，OCR 抖了一下

# ---- 让手：你在操作时不点 ----
USER_ACTIVE_SECONDS = 2.5
# ok 的 Genshin 交互点击时会 SetCursorPos（genshin.py make_mouse_position），
# 不确定会不会刷新 GetLastInputInfo —— 自己动作期间和之后 0.6 秒内的输入都不算你。
# 但「之后 0.6 秒」里的输入也可能是你刚抓起鼠标：宽限期没过完之前先不动手，过完再看
# （你还在动的话那时最近一次输入已经在宽限期外，认得出来）
OWN_INPUT_GRACE = 0.6
OWN_INPUT_KEEP = 16
# 借助战要几十秒、中间几十次点击。以前整段都记成「自己的动作」、中途也不看你，于是你在借人途中
# 动手它照点不误（2026-09-29 17:27：你选好人开打了，它还在找人，最后把开打的关退了出去）。
# 现在借人期间**每次**键鼠输入之前都看一眼，等待超过这么久之后也看一眼：你在操作就放弃这次借人、停手
GUARD_SLEEP_MIN = 0.5
# 你点开游戏看进度（切到前台、挪鼠标）不算接手，只是先停一下等你停手 / 切走（I-52）：
# 你操作了游戏（在游戏画面里点了鼠标、按了键 / 滚了滚轮，见 input_watch）才交给你。一直在动、这么久还没停，也交给你
PEEK_WAIT_MAX = 60.0
PEEK_POLL = 0.3
# 「操作了没有」看上一次看你之后的（最多往回看这么久）：两次检查之间隔着长等待（点完等 3.5 秒）时，
# 你点了一下就不动了，只看最近 2.5 秒会漏掉
USER_CHECK_SPAN = 30.0
# 排掉它自己的真键鼠（SendInput 真拖章节地图）时的宽限：它别的输入都是 PostMessage，系统里根本看不到，不用排；
# 真拖拽的按下松开都在记下的时间窗里（时刻取自 GetLastInputInfo，不受采样延迟影响）
CLICK_OWN_GRACE = 0.15
RAW_PENDING_WAIT = 0.4      # 看「你操作了没有」之前，最多等这么久让还在等原始输入的那一下分完类（input_watch.RAW_WAIT=0.3）

# ---- 节奏 ----
IDLE_SLEEP = 0.4
LOADING_SLEEP = 0.5
CLICK_BASE = 0.4                    # 点击后的等待 = 它 + min(界面等待, 0.8)
UI_DELAY_CAP = 0.8

STUCK_DEBUG_DIR = 'debug_tiles'
STUCK_DEBUG_KEEP = 120

RESULT_NAMES = {'victory': '打赢了', 'failed': '没打过', 'left': '没看到结算就回到了关卡外',
                'stalled': '认不出画面，停手交还给轮询', 'timeout': '超时'}


def battle_frame(boxes, W, H, comm=True):
    """这一帧是不是战斗画面：左上角有「N/M回合」计数，或者回合横幅；comm=True 时战斗里的通讯对话也算
    （轮询用它接管；run_stage 的 HUD 规则不算，通讯对话要点才继续）。见 ROUND_RE。"""
    W, H = float(W or 1), float(H or 1)
    for b in boxes or ():
        name = getattr(b, 'name', None) or ''
        if TURN_BANNER_RE.search(name) or (comm and comm_box(b, W, H)):
            return True
        if ROUND_RE.match(name):
            cx, cy = (b.x + b.width / 2.0) / W, (b.y + b.height / 2.0) / H
            if ROUND_BAND[0] <= cx <= ROUND_BAND[2] and ROUND_BAND[1] <= cy <= ROUND_BAND[3]:
                return True
    return False


def comm_box(b, W, H):
    """这块字是不是战斗里通讯对话的「通信中」标签：左上角那种，或者头像在下方那种（见 COMM_LOW_X）。"""
    name = getattr(b, 'name', None) or ''
    cx, cy = (b.x + b.width / 2.0) / float(W or 1), (b.y + b.height / 2.0) / float(H or 1)
    if COMM_RE.search(name) and COMM_BAND[0] <= cx <= COMM_BAND[2] and COMM_BAND[1] <= cy <= COMM_BAND[3]:
        return True
    return (bool(COMM_LABEL_RE.search(name)) and COMM_LABEL_Y[0] <= cy <= COMM_LABEL_Y[1]
            and (cx <= COMM_LOW_X[0] or cx >= COMM_LOW_X[1]))


def text_skeleton(name):
    """认不出的画面上一块字的「骨架」：去掉空白，数字换成 #。

    只用在**认不出**的画面上判断「字变了没有」：那里跳动的数字是倒计时、伤害数字、加载进度，不是进展；
    剩不到 2 个真正的字（纯数字、时间、百分比）的整块不要。部署人数、关卡编号都是在认得出的画面
    （部署界面、章节地图）上读的，不经过这里。
    """
    s = re.sub(r'\s+', '', name or '')
    sk = re.sub(r'\d+', '#', s)
    if len(re.sub(r'[#\W_]', '', sk)) < 2:
        return None
    return sk


def _seen(k, recent):
    """k 在最近一帧的字里算「见过」吗：一模一样；被拆开 / 合并（互相包含，短的那块至少有长的四成 ——
    否则新台词里恰好带着「自动」「跳过」这种按钮字就会被当成见过）；或者只差一点（OCR 抖了一下）。"""
    if k in recent:
        return True
    for x in recent:
        short, long_ = (k, x) if len(k) <= len(x) else (x, k)
        if short in long_ and len(short) >= 0.4 * len(long_):
            return True
        n = max(len(k), len(x))
        if difflib.SequenceMatcher(None, k, x).ratio() >= min(SIG_SIMILAR, (n - 1.0) / n):
            return True
    return False


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [('cbSize', ctypes.c_uint), ('dwTime', ctypes.c_uint)]


class _Run:
    """一次 run_stage 的全部状态。不挂在 self 上：每关都从零开始，不会带上一关的残留。"""

    def __init__(self, kind, in_battle, now):
        self.kind = kind
        self.t0 = now
        self.wall0 = time.time()             # 现场记录里写「这关跑了多久」用（t0 在测试里是假时钟）
        self.phase = 1
        self.victory = False
        self.failed = False
        self.battle_started = bool(in_battle)
        # 从战斗中 / 结算中接手也要真看到 HUD 才算「真在打」：轮询因为「还有可部署」判成战斗中时
        # （停手等接管期间你自己按了作战开始），确认之后还有开场对话，按「真在打」只点 ≤2 个字块
        # 的画面，3~6 个字块的对话永远跳不掉（复核模拟：卡到 stalled）
        self.hud_seen = False
        self.hud_logged = False
        self.start_clicks = 0
        self.last_start_t = None
        self.start_base = set()              # 点「作战开始」那一刻画面上的字（骨架）
        self.start_new = []                  # 点完之后回到部署界面时新冒出来的字（一闪而过的提示）
        self.start_changed = False           # 最近这次点完之后画面上冒出过新的字（正在进战斗）
        self.deploy_seen_t = None
        self.last_continue_t = None
        self.last_known_t = now
        self.unknown_since = None
        self.zero_since = None
        self.busy_since = None
        self.last_tap_t = None
        self.tap_fail = 0
        self.tap_given_up = False
        self.unknown_sig = None              # 上一帧认不出的画面读到的字（的骨架，见 text_skeleton）
        self.sig_hist = []                   # 最近几帧认不出的画面的字：新冒出来的字才算进展
        self.loading_since = None
        self.cap_logged = False
        self.unknown_saved = False
        self.unknown_path = None
        self.unknown_known_t = None          # 存那张图时的 last_known_t：同一段认不出，卡住时直接用那张
        self.last_progress_t = now           # 最近一次「有进展」：认出了界面，或者认不出但字变了
        self.tap_total = 0
        self.intro_taps = 0
        self.battle_taps = 0
        self.clicks = {}                     # 同一处的点击时间
        self.streak_key = None
        self.streak_n = 0
        self.streak_gone_t = None            # 连点的那个按钮从什么时候起一直不在
        self.refused = set()
        self.alert_clicks = 0                # 这个跳过确认框里勾了几次「今日不再提示」
        self.skip_t = None                   # 最近一次点「跳过」的时刻（之后弹的「确认」是跳过确认框）
        self.confirm_noted = False
        self.exit_rounds = 0
        self.entered = False                 # 见过关卡外以外的任何画面
        self.label = None
        self.debug = None
        self.note = None                     # 停手交还时的原因（轮询拿它提醒你，别一律说成「认不出画面」）
        self.post_continue = False
        self.motion = []
        self.prev_thumb = None
        self.prev_thumb_t = None
        self.tutorial_taps = 0
        self.overlay_noresp = 0
        self.cap_noted = False
        self.tut_noted = False
        self.tut_cap_noted = False
        self.x_cand = None
        self.x_clicks_phase = 0
        self.x_clicks_run = 0
        self.x_last = None
        self.x_noeffect = 0
        self.x_fresh = False                 # 点叉之后读到了新字
        self.x_blocked = False
        self.x_refused_saved = False
        self.user_last_t = None
        self.responses = []
        self.resp_pending = None
        self.tap_thumbs = []
        self.ov_seen = []
        self.self_moved_t = None             # 最近一次画面自己在动（不是点出来的）
        self.rearms = 0
        self.rearm_t = None
        self.refunds = 0
        self.comm = False                    # 这一帧是战斗里的通讯对话
        self.comm_taps = 0
        self.battle_refunds = 0              # 见过 HUD 之后点出了新台词、退还的盲点次数（I-48 复查）
        self.comm_cap_noted = False
        self.fill_rounds = 0                 # 「本关至少需要出击N人」之后从「我的人形」补了几轮（_sr_fill）
        self.fill_added = []                 # 这一阶段补上了谁 / 谁没上、为什么（几轮加起来，停手提醒里写）
        self.fill_skipped = []
        self.overlay_taps = 0                # 部署界面上的教学 / 提示点了几下中间（I-43）
        self.overlay_cap_noted = False
        self.refilled = False                # 这一关试过用补给补情报拼图了（只试一次，成没成都算）
        self.refill_clicked = False          # 补的时候点过框里的东西（停手提醒就不能再说「没点」）

    @property
    def verdict(self):
        return self.victory or self.failed

    def new_phase(self):
        self.phase += 1
        self.victory = self.failed = False
        self.battle_started = self.hud_seen = self.hud_logged = False
        self.start_clicks = 0
        self.last_start_t = None
        self.start_base = set()
        self.start_new = []
        self.start_changed = False
        self.deploy_seen_t = None
        self.intro_taps = self.battle_taps = self.comm_taps = self.battle_refunds = 0
        self.comm_cap_noted = False
        self.post_continue = self.phase >= 2
        self.refunds = 0
        self.fill_rounds = 0
        self.fill_added, self.fill_skipped = [], []
        self.reset_tutorial()

    def reset_tutorial(self):
        """每阶段（和「继续前进」之后第一次点「作战开始」时）重新算的教学 / 点叉状态。"""
        self.tutorial_taps = self.x_clicks_phase = self.overlay_noresp = self.x_noeffect = self.rearms = 0
        self.overlay_taps = 0
        self.cap_noted = self.tut_noted = self.tut_cap_noted = self.x_blocked = self.overlay_cap_noted = False
        self.x_cand = self.x_last = self.resp_pending = self.rearm_t = None
        self.tap_thumbs = []


class StageRunnerMixin:
    """挂在 SupportBattleMixin 的子类上：run_stage() 把一关从进本打到结算（或交还给轮询）。

    宿主要提供 SupportBattleMixin 的 _should_borrow / borrow_support / support_options，
    以及 ok 的 ocr / find_boxes / click_box / click_relative / back / box_of_screen / box。
    可选开关（实例属性）：_tap_to_reveal_skip（盲点唤出跳过）、_yield_to_user（你操作时不点）。
    """

    # ---------------------------------------------------------------- 平台（测试里替换）

    def _now(self):
        return time.time()

    def _user_idle_seconds(self):
        """距离最近一次键鼠输入过了几秒（GetLastInputInfo，系统全局）。读不到返回 None。"""
        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        tick = ctypes.windll.kernel32.GetTickCount
        tick.restype = ctypes.c_uint32
        return ((tick() - info.dwTime) & 0xFFFFFFFF) / 1000.0

    def _game_foreground(self):
        win = getattr(getattr(self.executor, 'interaction', None), 'hwnd_window', None)
        if win is None:
            win = self.hwnd
        return bool(win is not None and win.is_foreground())

    def _game_hwnd(self):
        win = getattr(getattr(self.executor, 'interaction', None), 'hwnd_window', None)
        if win is None:
            win = self.hwnd
        return getattr(win, 'hwnd', None)

    def _input_watch_allowed(self):
        """测试换掉了 _user_idle_seconds（假的键鼠）就不开真线程 —— 它会读到跑测试的人的鼠标。"""
        return input_watch is not None and type(self)._user_idle_seconds is StageRunnerMixin._user_idle_seconds

    def _input_watch(self):
        """分「操作游戏」和「只是看看」的后台线程（input_watch，I-52）。轮询一开始就开（RealtimeAssistTask.run）；
        开之前 / 停过的那段它没看着，ops_since 返回 None，调用方按老办法判。没开 / 不健康返回 None。"""
        w = getattr(self, '_iw', None)
        if w is None:
            w = False
            if self._input_watch_allowed():
                try:
                    w = input_watch.InputWatch(self._game_foreground, self._game_hwnd).start()
                except STOP_EXCEPTIONS:
                    raise
                except Exception:
                    w = False
            self._iw = w
            state = w.raw_state() if (w and hasattr(w, 'raw_state')) else None
            if state == 'off':
                self.log_info('原始输入旁听没开起来：「光标没动的那一下」照老办法算你在操作（I-64）')
        if w and not w.healthy():
            if not getattr(self, '_iw_down_said', False) and getattr(w, 'failed', False):
                self._iw_down_said = True
                self.log_info('输入监视线程停了：分不清「操作游戏」和「只是看看」，照老办法判（I-52）')
            return None
        return w or None

    # ---------------------------------------------------------------- 让手

    def _sr_input(self, fn, *args, **kwargs):
        """所有键鼠输入的关口：借助战途中先看你在不在操作（在就抛 UserTakeoverError，这一下不做），
        做完记下自己动作的时间窗，好把「自己点的」和「你动的键鼠」分开。

        嵌套（click_box -> click、一整个拖拽里的按下 / 挪动 / 松开）只在最外层过关口、记时间 ——
        按下和松开之间绝不打断。"""
        if getattr(self, '_sr_depth', 0):
            return fn(*args, **kwargs)
        self._sr_guard_input()
        self._sr_check_last = self._now()    # 刚看过你（关口没开时也算：这之后的操作下一次再看）
        trail = getattr(self, '_trail', None)
        if trail:
            trail('input', '%s %s %s' % (getattr(fn, '__name__', 'input'), ' '.join(map(str, args))[:80],
                                         kwargs.get('name') or ''))
        self._sr_depth = 1
        t0 = self._now()
        try:
            return fn(*args, **kwargs)
        finally:
            self._sr_depth = 0
            self._sr_note_own(t0, self._now())

    def click(self, *args, **kwargs):
        """所有点击都从这里过（click_box / click_relative 最后都调 click）。
        点击后的 after_sleep 拿出来单独睡：不算在动作里，睡完还要再看一眼你（见 sleep）。"""
        if getattr(self, '_sr_depth', 0):
            return super().click(*args, **kwargs)
        after = kwargs.pop('after_sleep', 0) or 0
        r = self._sr_input(super().click, *args, after_sleep=0, **kwargs)
        if after > 0:
            self.sleep(after)
        return r

    def scroll(self, *args, **kwargs):
        return self._sr_input(super().scroll, *args, **kwargs)

    def swipe(self, *args, **kwargs):
        return self._sr_input(super().swipe, *args, **kwargs)

    def send_key(self, *args, **kwargs):
        return self._sr_input(super().send_key, *args, **kwargs)

    def mouse_down(self, *args, **kwargs):
        return self._sr_input(super().mouse_down, *args, **kwargs)

    def mouse_up(self, *args, **kwargs):
        return self._sr_input(super().mouse_up, *args, **kwargs)

    def move(self, *args, **kwargs):
        return self._sr_input(super().move, *args, **kwargs)

    def back(self, *args, **kwargs):
        return self._sr_input(super().back, *args, **kwargs)

    def ocr(self, *args, **kwargs):
        """整屏 OCR 的结果留一份（现场记录里写「当时读到了什么」）。"""
        r = super().ocr(*args, **kwargs)
        if not args and kwargs.get('box') is None and kwargs.get('frame') is None:
            self._last_full_ocr = (time.time(), r)
        return r

    def sleep(self, timeout):
        """借助战途中等得久一点（≥ GUARD_SLEEP_MIN）之后也看一眼你：等的时候你动了手，
        下一步就不该再做（哪怕下一步不是键鼠输入，而是拿旧画面下结论）。"""
        r = super().sleep(timeout)
        if not getattr(self, '_sr_depth', 0) and (timeout or 0) >= GUARD_SLEEP_MIN:
            self._sr_guard_input()
        return r

    def _sr_guard_input(self):
        """借助战途中（_sr_guarding）：你在操作就抛 UserTakeoverError。

        最近一次输入落在自己上一次动作刚结束的宽限期里（_sr_unsure：分不清是点击的余波还是你刚抓起鼠标）
        就等宽限期过完再看一次 —— 你还在动的话那时最近一次输入已经在宽限期外，认得出来。"""
        if not getattr(self, '_sr_guarding', False):
            return
        what = getattr(self, '_sr_guard_what', None) or '借助战'
        for _ in range(3):
            why = self._sr_check_user(what)
            if why:
                e = UserTakeoverError('%s途中你%s' % (what, why))
                e.by_input = True              # 是你的键鼠（不是「画面不对」那种）：轮询之后可以接着借（I-52）
                raise e
            if not getattr(self, '_sr_unsure', False):
                return
            self._sr_quiet_sleep(OWN_INPUT_GRACE)

    def _sr_quiet_sleep(self, s):
        self._sr_depth = 1              # 这一下等待别再触发关口
        try:
            super().sleep(s)
        finally:
            self._sr_depth = 0

    def _user_ops_since(self, t0):
        """t0 之后你操作游戏的 [(时刻, 'click' / 'key')]（它自己的真键鼠排掉）；
        分不出来（线程没开 / 不健康 / t0 那会儿它还没看着）返回 None。"""
        w = self._input_watch()
        if w is None:
            return None
        pend = getattr(w, 'pending_since', None)
        if pend is not None:
            # 光标没动的那一下还在等原始输入分类（最多 RAW_WAIT）：等它分完再看，不然这一下在这次检查里看不到、
            # 下次检查的起点又挪过去了，就漏掉了（I-64 复查）
            end = time.time() + RAW_PENDING_WAIT
            while pend(t0) and time.time() < end:
                time.sleep(0.02)
        ops = w.ops_since(t0)
        if ops is None:
            return None
        real = getattr(self, '_sr_real_own', None) or ()
        return [(t, k) for t, k in ops if not any(a - 0.05 <= t <= b + CLICK_OWN_GRACE for a, b in real)]

    @staticmethod
    def _sr_ops_why(ops):
        return '点了游戏画面' if any(k == 'click' for _, k in ops) else '在游戏里按了键 / 滚了滚轮'

    def _sr_ops_note(self, ops):
        """这几下操作的依据（input_watch 记的：按键 0x41 / 滚轮 / 原始输入没看到……），写进交还的那句话里，
        下次误判有据可查（I-64）。没有就是空串。"""
        w = self._input_watch()
        if w is None or not ops or not hasattr(w, 'notes_since'):
            return ''
        try:
            t0 = min(t for t, _ in ops)
            seen = {t for t, _ in ops}
            notes = [why for t, k, why in w.notes_since(t0) if t in seen and k != 'click']
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return ''
        return '（%s）' % '、'.join(dict.fromkeys(notes)) if notes else ''

    def _sr_check_user(self, what):
        """借人 / 换章 / 补人 / 拖章节地图途中看你一眼（I-52）：
          - 你操作了游戏（在游戏画面里点了鼠标、按了键 / 滚了滚轮）-> 返回原因，调用方交给你；
          - 只是点开游戏看 / 挪鼠标 -> 等你停手（或者切走）再返回 None 接着做；一直不停超过 PEEK_WAIT_MAX 也交给你；
          - 分不出来（input_watch 没开 / 那段没看着）-> 老办法：user_busy 就交给你。
        「操作了没有」看上一次看你之后的（最多 USER_CHECK_SPAN 秒，至少最近 2.5 秒）：两次检查之间隔着长等待时，
        你点了一下就不动了也认得出。「你操作时暂停点击」关掉时一律不管。

        以前 user_busy 一为真就交给你：借助战途中点开游戏看一眼，这次就不借了、停着等你自己放人。"""
        if not getattr(self, '_yield_to_user', True):
            self._sr_unsure = False
            return None
        now = self._now()
        since = now - USER_ACTIVE_SECONDS
        last = getattr(self, '_sr_check_last', None)
        if last is not None and now - last <= USER_CHECK_SPAN:
            since = min(since, last)
        ops = self._user_ops_since(since)
        why = self._sr_ops_why(ops) + self._sr_ops_note(ops) if ops else None
        if why is None and self.user_busy():
            why = self._sr_wait_peek(what)
        self._sr_check_last = self._now()
        return why

    def _sr_wait_peek(self, what):
        """user_busy() 为真、没看到你操作游戏：等你停手（或者切走）返回 None；这期间你操作了 / 分不出来 / 一直不停
        超过 PEEK_WAIT_MAX -> 返回原因（交给你）。"""
        t0 = self._now()
        noted = False
        while True:
            ops = self._user_ops_since(t0 - USER_ACTIVE_SECONDS)
            if ops is None:
                return '动了键鼠'              # 分不出来：老办法
            if ops:
                return self._sr_ops_why(ops) + self._sr_ops_note(ops)
            if not self.user_busy():
                return None
            if not noted:
                noted = True
                self.log_info('你在看游戏（没点游戏画面），%s先停一下，你停手 / 切走就接着 —— 你一点游戏画面我就交给你' % what)
            if self._now() - t0 >= PEEK_WAIT_MAX:
                return '一直在动键鼠（%d 秒没停）' % PEEK_WAIT_MAX
            self._sr_quiet_sleep(PEEK_POLL)

    def _sr_note_real(self, t0, t1):
        """它自己发的真键鼠（SendInput：真拖章节地图从按下到松开）。input_watch 看得到这些，按这些时间窗排掉。"""
        real = getattr(self, '_sr_real_own', None)
        if real is None:
            real = self._sr_real_own = []
        real.append((t0, max(t0, t1)))
        del real[:-OWN_INPUT_KEEP]

    def _sr_note_own(self, t0, t1):
        own = getattr(self, '_sr_own', None)
        if own is None:
            own = self._sr_own = []
        own.append((t0, max(t0, t1)))
        del own[:-OWN_INPUT_KEEP]

    def _sr_own_end(self, t, grace=OWN_INPUT_GRACE):
        """t 落在某次自己动作的窗口里（动作期间 + 之后 grace 秒宽限，默认 0.6）：返回那次动作结束的时刻，否则 None。"""
        for a, b in reversed(getattr(self, '_sr_own', None) or ()):
            if a - 0.05 <= t <= b + grace:
                return b
        return None

    def _sr_is_own(self, t):
        return self._sr_own_end(t) is not None

    def user_busy(self):
        """你在操作吗：游戏在前台，且最近 2.5 秒内有不是它自己点出来的键鼠输入（挪鼠标也算 —— 这时它不点，免得
        抢你的光标；借人 / 换章途中要不要交给你，再看你点没点游戏画面，见 _sr_wait_peek）。

        读不到（非 Windows、拿不到窗口）一律当没人在操作。状态变了才打一行日志。
        另外设 self._sr_unsure：最近一次输入落在自己动作**之后**的宽限期里、宽限期还没过完 ——
        分不清是自己点击的余波还是你刚抓起鼠标（借助战一结束你就在动，下一轮离借人结束才 0.2 秒），
        调用方这一轮也别动手（复核模拟：原来会在你动鼠标时照点「作战开始」）。
        """
        busy = False
        self._sr_unsure = False
        if not getattr(self, '_sr_guarding', False):
            # 没在借人 / 换章：这一下就是在看你（你在动它就不点）—— 之后的关口从这里往后看你操作了没有
            self._sr_check_last = self._now()
        if getattr(self, '_yield_to_user', True):
            try:
                if self._game_foreground():
                    idle = self._user_idle_seconds()
                    if idle is not None and 0 <= idle < USER_ACTIVE_SECONDS:
                        now = self._now()
                        t_in = now - idle
                        end = self._sr_own_end(t_in)
                        busy = end is None
                        self._sr_unsure = bool(end is not None and t_in > end
                                               and now <= end + OWN_INPUT_GRACE)
            except STOP_EXCEPTIONS:
                raise
            except Exception:
                busy = self._sr_unsure = False
        if busy != getattr(self, '_sr_yielding', False):
            self._sr_yielding = busy
            if not busy:
                self.log_info('你停手了，接着自动')
            else:
                ops = self._user_ops_since(self._now() - USER_ACTIVE_SECONDS)
                self.log_info('你在看游戏（没点游戏画面），先不点' if ops == [] else '你在操作，先不点')
        return busy

    # ---------------------------------------------------------------- 小工具

    def _sr_pause(self, base=CLICK_BASE):
        try:
            ui = float(getattr(self, 'ui_delay', 1.0) or 0)
        except (TypeError, ValueError):
            ui = 1.0
        return base + min(max(ui, 0.0), UI_DELAY_CAP)

    def _sr_H(self):
        h = self.height
        if not h and self.frame is not None:
            h = self.frame.shape[0]
        return h or 1

    def _sr_key(self, box):
        w = max(1, self.width or 1) * SPOT_GRID
        h = self._sr_H() * SPOT_GRID
        return (box.name, int((box.x + box.width / 2) / w), int((box.y + box.height / 2) / h))

    def _sr_may_click(self, st, box, now=None):
        """防连点：同一处 3 秒内最多 2 次；连着点了 6 次它都还在就先不点它（交给后面的规则）。

        按真正点下去的时刻算（不是截图时刻），和 _sr_click 记的时间对得上。
        """
        now = self._now()
        key = self._sr_key(box)
        times = [t for t in st.clicks.get(key, ()) if now - t < SAME_SPOT_WINDOW]
        st.clicks[key] = times
        if len(times) >= SAME_SPOT_MAX:
            return False
        if key == st.streak_key and st.streak_n >= SAME_TARGET_GIVE_UP:
            if key not in st.refused:
                st.refused.add(key)
                self.log_info('「%s」连着点了 %d 次都还在，先不点它了' % (box.name, st.streak_n))
            return False
        return True

    def _sr_click(self, st, box, after, now=None):
        key = self._sr_key(box)
        st.clicks.setdefault(key, []).append(self._now())
        if SKIP_TEXT in (getattr(box, 'name', None) or ''):
            st.skip_t = self._now()
        if key == st.streak_key:
            st.streak_n += 1
        else:
            st.streak_key, st.streak_n = key, 1
        st.streak_gone_t = None
        self.click_box(box, after_sleep=after)

    def _sr_forget_gone(self, st, boxes, now):
        """上次连点的那个按钮连续 2 秒不在了 —— 连点计数清零（一段段剧情的「跳过」都在同一处）。

        只看一帧会被 OCR 时有时无的死按钮骗过去，见 STREAK_GONE_SECONDS。
        """
        if not st.streak_key:
            return
        if any(self._sr_key(b) == st.streak_key for b in boxes if b.name == st.streak_key[0]):
            st.streak_gone_t = None
        elif st.streak_gone_t is None:
            st.streak_gone_t = now
        elif now - st.streak_gone_t >= STREAK_GONE_SECONDS:
            st.refused.discard(st.streak_key)
            st.streak_key, st.streak_n, st.streak_gone_t = None, 0, None

    def _sr_known(self, st, now, deploy=False):
        """认出了一个界面：卡住计时、盲点计数都重来。"""
        st.last_known_t = st.last_progress_t = now
        st.unknown_since = st.zero_since = st.busy_since = None
        st.tap_fail = 0
        if st.tap_given_up:
            st.tap_given_up = False
            self.log_info('认出画面了，恢复「点屏幕唤出跳过」')
        if not deploy:
            st.deploy_seen_t = None

    def _sr_say(self, st, label, msg, notify=False):
        if st.label != label:
            st.label = label
            self.log_info(msg, notify=notify)

    def _sr_thumb(self):
        try:
            frame = self.frame
            if frame is None:
                return None
            small = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
            if small.ndim == 3:
                small = cv2.cvtColor(np.ascontiguousarray(small[..., :3]), cv2.COLOR_BGR2GRAY)
            return small.astype(np.float32)
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return None

    @staticmethod
    def _sr_changed(a, b):
        if a is None or b is None:
            return True                     # 看不出来就当变了，别因此停手
        return float(np.mean(np.abs(a - b))) > THUMB_CHANGE

    @staticmethod
    def _sr_is_loading(boxes):
        return any(LOADING_RE.search(b.name or '') for b in boxes)

    def _sr_ocr(self, **kwargs):
        """补看一眼用的 OCR：WGC 偶尔抓回空帧（上游 skip_dialogs 同样处理），这时当作什么都没看到。"""
        try:
            return self.ocr(**kwargs) or []
        except STOP_EXCEPTIONS:
            raise
        except AttributeError:
            return []

    def _sr_on_story_list(self, boxes):
        """这一帧是不是「剧情战役」章节列表（chapter_nav.story_list_info；认不出 / 出错都当不是）。"""
        try:
            return story_list_info(boxes, self.width, self._sr_H()) is not None
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return False

    def _sr_bottom_skip(self, boxes):
        """「通信中」对话框下面的「跳过」（见 SKIP_BOTTOM_Y）：画面上得有头像旁边的「通信」标签，免得点到别处的同名字。"""
        W, H = max(1, self.width or 1), self._sr_H()
        if not any(COMM_LABEL_RE.search(b.name or '') and COMM_LABEL_Y[0] * H <= b.y + b.height / 2.0 <= COMM_LABEL_Y[1] * H
                   for b in boxes):
            return []
        out = []
        for b in boxes:
            if not SKIP_BOTTOM_RE.match((b.name or '').strip()):
                continue
            cx, cy = b.x + b.width / 2.0, b.y + b.height / 2.0
            if cy >= SKIP_BOTTOM_Y * H and SKIP_BOTTOM_X[0] * W <= cx <= SKIP_BOTTOM_X[1] * W:
                out.append(b)
        return out

    def _sr_deploy_overlay(self, st, boxes, now, busy, cap):
        """部署界面上盖着教学 / 提示（I-43）：先不借人。右上角有叉（连着两帧、画面静止）就点叉，没叉就点中间翻页；
        点了 DEPLOY_OVERLAY_TAP_CAP 下还关不掉就提醒你、只看（卡住计时照走）。"""
        self._sr_say(st, 'deploy_overlay', '部署界面上盖着教学 / 提示（「%s」），先关掉它再借人' % (cap.name or '')[:16])
        if busy:
            self._sr_known(st, now, deploy=True)
            return self._sr_idle()
        r = self._sr_close_x(st, boxes, now, busy, 0, False)
        if r == 'clicked':
            self._sr_known(st, now, deploy=True)
            return None
        if r == 'pending':
            return self._sr_idle(0.3)
        if st.overlay_taps < DEPLOY_OVERLAY_TAP_CAP:
            if st.last_tap_t is not None and now - st.last_tap_t < DEPLOY_OVERLAY_TAP_INTERVAL:
                return self._sr_idle(0.3)
            st.overlay_taps += 1
            st.last_tap_t = now
            self._sr_known(st, now, deploy=True)
            self.log_info('点一下屏幕中央翻教学 / 提示（第 %d 下，最多 %d 下）' % (st.overlay_taps, DEPLOY_OVERLAY_TAP_CAP))
            self.click_relative(*SKIP_REVEAL_POINT, after_sleep=0)
            return self._sr_idle(TAP_SETTLE)
        if not st.overlay_cap_noted:
            st.overlay_cap_noted = True
            msg = '部署界面上的教学 / 提示点了 %d 下中间还没关掉，不再点 —— 你来关一下（关掉之后我接着借人）' % DEPLOY_OVERLAY_TAP_CAP
            alert = getattr(self, 'alert_user', None)
            if alert is not None:
                alert(msg)
            else:
                self.log_info(msg, notify=True)
            self._sr_evidence('deploy_overlay', self._sr_extra(st, boxes))
        return self._sr_stall(st, now, STALL_SECONDS) or self._sr_idle()

    def _sr_top_skip(self, boxes):
        limit = SKIP_TOP_RATIO * self._sr_H()
        exact = [b for b in boxes if b.name == SKIP_TEXT and b.y < limit]
        if exact:
            return exact
        # 10/1 11:34 打完 5-3 的剧情对话：右上角「自动」「跳过」「全部跳过」被 OCR 读成一块「自动跳过全部跳过」，
        # 认不出跳过、开场盲点又用完了，停在那里等你点了 2 分钟。按字在这一块里的位置估出「跳过」那两个字在哪，
        # 优先不是「全部跳过」里的那个。只看顶部这一条里的短字块（按钮），不看台词
        out = []
        W, H = float(self.width or 1), float(self._sr_H() or 1)
        for b in boxes:
            name = b.name or ''
            if b.y >= limit or SKIP_TEXT not in name or len(name) > 12 or b.width <= 0:
                continue
            starts = [i for i in range(len(name)) if name.startswith(SKIP_TEXT, i)]
            plain = [i for i in starts if name[max(0, i - 2):i] != '全部']
            i = (plain or starts)[0]
            n = float(len(name))
            x0, x1 = b.x + b.width * i / n, b.x + b.width * (i + len(SKIP_TEXT)) / n
            out.append(self.box_of_screen(x0 / W, b.y / H, x1 / W, (b.y + b.height) / H, name=SKIP_TEXT))
        return out

    def _save_stuck_image(self, tag='stuck'):
        """停手时把当前画面存下来（debug_tiles/日期/HHMMSS_stuck.jpg），返回路径。存图失败不影响流程。"""
        try:
            frame = self.frame
            if frame is None:
                return None
            folder = os.path.join(STUCK_DEBUG_DIR, time.strftime('%Y%m%d'))
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, time.strftime('%H%M%S') + '_' + tag + '.jpg')
            cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(path)
            note = getattr(self, '_save_case_note', None)
            if note:
                note(path, tag)
            prune_debug_dir(STUCK_DEBUG_DIR, STUCK_DEBUG_KEEP)
            return path
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            self.log_info('存现场图失败（不影响流程）：%r' % (e,))
            return None

    # ---------------------------------------------------------------- 入口

    def run_stage(self, kind='battle', in_battle=False, phase=1):
        """把一关从当前画面一路打到结算完、回到关卡外；或者停手交还给轮询。

        :param kind: 'battle' = 点了「挑战」之后 / 部署界面 / 断线重连回到关卡里；
                     'story' = 点了「观看」的纯剧情关
        :param in_battle: 从「战斗中 / 结算中」接手，一开始就当作已经开打
        :param phase: 从第几阶段接着打（同一关中途重新接管时由轮询传进来：第 2 阶段起不退关换好友）
        :return: dict(result, phases, victory, failed, entered, debug, names)。names 是最后一帧读到的文字
                 （轮询拿它认「还是刚才卡住的那个画面」）。result 是
                 'victory'（看到过任务完成 / 对战胜利，并回到了关卡外）/ 'failed'（看到过失败）/
                 'left'（回到关卡外但没看到胜负：你手动退出了，或者剧情看完了）/
                 'stalled'（长时间认不出画面，debug 是现场图路径）/ 'timeout'（总时长上限）
        :raises TileNotSelectedError, SupportStateError, UserTakeoverError: 借助战抛的原样往外抛
        """
        st = _Run(kind, in_battle, self._now())
        st.phase = max(1, int(phase or 1))
        self._sr_current = st                # 现场记录（_save_case_note）里写第几阶段、到哪一步了
        self._support_done = False
        self._phase_support = None           # 这一阶段场上的助战叫什么（借完读详情卡，_sr_note_support）
        warned_empty = False
        result = None
        names = []
        while result is None:
            now = self._now()
            if now - st.t0 > RUN_STAGE_TIMEOUT:
                result = 'timeout'
                st.note = '一关跑了 %d 分钟还没完' % (RUN_STAGE_TIMEOUT // 60)
                st.debug = self._save_stuck_image('timeout')
                break
            self.next_frame()
            t_frame = self._now()           # 计时都按截图的时刻算，不算 OCR 花的那 0.2 秒
            try:
                boxes = self.ocr()
            except STOP_EXCEPTIONS:
                raise
            except AttributeError:
                # WGC 偶尔抓回空帧（上游 skip_dialogs 同样处理）
                if not warned_empty:
                    warned_empty = True
                    self.log_info('抓帧返回空帧，稍等重试')
                self.sleep(1)
                continue
            names = [b.name for b in boxes or [] if b.name]
            tick = getattr(self, '_stuck_tick', None)      # 卡住提醒（stuck_alert）：关卡里一直没进展也要提醒你
            if tick is not None:
                tick(names, 'stage', True)
            result = self._sr_step(st, boxes or [], t_frame)
        self._sr_current = None
        label = RESULT_NAMES[result]
        if st.kind == 'story' and result == 'left' and st.entered:
            label = '剧情看完了，回到了关卡外'      # 剧情点没有结算，这就是正常结束（以前写成「没看到结算…」）
        self.log_info('本关流程结束：%s（第 %d 阶段，用时 %d 秒）%s' % (
            label, st.phase, int(self._now() - st.t0),
            ('，现场图 %s' % st.debug) if st.debug else ''))
        return dict(result=result, phases=st.phase, victory=st.victory, failed=st.failed,
                    entered=st.entered, debug=st.debug, names=names, note=st.note)

    # ---------------------------------------------------------------- 一轮

    def _sr_idle(self, secs=IDLE_SLEEP):
        self.sleep(secs)
        return None

    def _sr_chain(self, name):
        """告诉轮询「继续前进」连打到哪了（_chain_won / _chain_started 在 RealtimeAssistTask 上；别的宿主没有就不管）。"""
        hook = getattr(self, name, None)
        if hook is not None:
            hook()

    def _sr_step(self, st, boxes, now=None):
        """认一帧、最多做一个动作。返回结果字符串就结束，None 继续下一轮。

        :param now: 这一帧截图的时刻
        """
        now = self._now() if now is None else now
        H = self._sr_H()
        # 你在操作 / 分不清是不是你（见 user_busy）：这一轮只看不点
        busy = self.user_busy() or bool(getattr(self, '_sr_unsure', False))
        if busy:
            st.last_known_t = now          # 你在操作：别把这段算成「卡住」
            st.user_last_t = now
        exit_rounds, st.exit_rounds = st.exit_rounds, 0
        self._sr_forget_gone(st, boxes, now)
        self._sr_motion(st, now)

        # 1) 跳过剧情的确认框：勾「今日不再提示」再点确认（框在最上层，排在顶部「跳过」前面）
        alert = self.find_boxes(boxes, match=NO_ALERT)
        if not alert:
            st.alert_clicks = 0
        else:
            st.entered = True
            self._sr_say(st, 'no_alert', '跳过确认框：勾上「今日不再提示」再确认')
            if busy:
                self._sr_known(st, now)
                return self._sr_idle()
            ok_btn, ticked = None, False
            recent = st.clicks.get(self._sr_key(alert[0]))
            # 3 秒内勾过就别再点（再点一次是取消勾选），同一个框最多勾两次，之后只找确认
            if st.alert_clicks < 2 and not (recent and self._now() - recent[-1] < SAME_SPOT_WINDOW):
                st.alert_clicks += 1
                ticked = True
                self._sr_known(st, now)
                self._sr_click(st, alert[0], NO_ALERT_SLEEP, now)
                self.next_frame()
                ok_btn = self.find_boxes(self._sr_ocr(), match=CONFIRM_NAMES)
            ok_btn = ok_btn or self.find_boxes(boxes, match=CONFIRM_NAMES)
            t = self._now()
            if ok_btn and self._sr_may_click(st, ok_btn[0], t):
                self._sr_known(st, t)
                self._sr_click(st, ok_btn[0], self._sr_pause(0.3), t)
                return None
            if ticked:
                return self._sr_idle(0.3)
            # 勾过了、确认也点不动：往下走

        # 2) 顶部「跳过」
        skip = self._sr_top_skip(boxes)
        if skip and self._sr_may_click(st, skip[0], now):
            st.entered = True
            self._sr_known(st, now)
            if busy:
                return self._sr_idle()
            self._sr_say(st, 'skip', '跳过剧情 / 动画')
            self._sr_click(st, skip[0], SKIP_AFTER_SLEEP, now)
            return None

        # 2b) 「通信中」对话框下面的「跳过」（见 SKIP_BOTTOM_Y 上面）
        low = self._sr_bottom_skip(boxes)
        if low and self._sr_may_click(st, low[0], now):
            st.entered = True
            self._sr_known(st, now)
            if busy:
                return self._sr_idle()
            self._sr_say(st, 'skip_bottom', '点对话框下面的「跳过」')
            self._sr_click(st, low[0], SKIP_AFTER_SLEEP, now)
            return None

        # 3) 阵容没上满的确认框：点底部的「确认」照常开打（上游同样处理）
        if self.find_boxes(boxes, match=LINEUP_NOT_FULL_RE):
            st.entered = True
            self._sr_say(st, 'lineup', '阵容没上满，点「确认」照常开打', notify=True)
            btn = self.find_boxes(boxes, match=CONFIRM_NAMES, boundary=self.box.bottom)
            if not btn:
                # 确认框还在弹出来 / 按钮读不出来：等，但不算「认出来了」—— 一直读不出来就按卡住交还，
                # 别在这里空转到 40 分钟总时限（复核模拟：原来每轮都当认出来了，永远不判卡住）
                return self._sr_stall(st, now, STALL_SECONDS) or self._sr_idle(0.3)
            if busy or self._sr_may_click(st, btn[0], now):
                self._sr_known(st, now)
                if busy:
                    return self._sr_idle(0.3)
                self._sr_click(st, btn[0], self._sr_pause(), now)
                return None
            # 点了没反应：往下走

        # 4) 继续前进：下一阶段会重新部署，也要重新借人（phase > 1 时不退关）
        # 阶段数按「看到」算（你自己点了继续前进也算）；点不点另说
        cont = self.find_boxes(boxes, match=CONTINUE)
        if cont:
            st.entered = True
            if st.last_continue_t is None or now - st.last_continue_t > CONTINUE_DEDUP:
                # 只有打赢的结算页才有「继续前进」：正在打的这关当场记成打完（RealtimeAssistTask._chain_won，stage_memory）
                self._sr_chain('_chain_won')
                st.new_phase()
                self._support_done = False
                self._phase_support = None       # 助战不跟着进下一关（自己的人形会，见 own_units）
                st.label = 'continue'
                self.log_info('继续前进：进入第 %d 阶段' % st.phase, notify=True)
            st.last_continue_t = now
            if busy or self._sr_may_click(st, cont[0], now):
                self._sr_known(st, now)
                if busy:
                    return self._sr_idle()
                self._sr_click(st, cont[0], self._sr_pause(), now)
                return None
            # 连点被拦（点了没反应）：往下走，别每轮都当「认出来了」而永远不判卡住

        # 5) 胜负。有「确认」就先点确认（上游也是 继续前进 > 确认 > 胜负字样）
        win = self.find_boxes(boxes, match=VICTORY_NAMES)
        lose = [] if win else self.find_boxes(boxes, match=DEFEAT_NAMES)
        if win or lose:
            st.entered = True
            if win and not st.victory:
                st.victory = True
            if lose and not st.failed:
                st.failed = True
            # 后面可能是结算动画 / 剧情：允许唤出跳过
            st.battle_started = st.hud_seen = False
            self._sr_say(st, 'verdict', '结算：%s' % (win or lose)[0].name, notify=bool(lose))
            ok_btn = [b for b in self.find_boxes(boxes, match=SETTLE_CONFIRM) if b.y > CONFIRM_MIN_Y * H]
            target = ok_btn[0] if ok_btn else (win or lose)[0]
            if busy or self._sr_may_click(st, target, now):
                self._sr_known(st, now)
                if busy:
                    return self._sr_idle()
                self._sr_click(st, target, self._sr_pause(), now)
                return None
            # 3 秒内点过两次了：往下走，按认不出处理（结算横幅点屏幕中央也一样能过）

        # 6) 部署界面：先借人，再点「作战开始」
        start = self.find_boxes(boxes, match=START_BATTLE, boundary=self.box.bottom)
        cap = overlay_caption(boxes, self.width, H) if start else None
        if cap is not None:
            st.entered = True
            return self._sr_deploy_overlay(st, boxes, now, busy, cap)
        if start:
            st.entered = True
            self._sr_known(st, now, deploy=True)
            if st.deploy_seen_t is None:
                st.deploy_seen_t = now
                self._sr_chain('_chain_started')     # 「继续前进」之后进了部署：开的是下一关
                self._sr_say(st, 'deploy', '部署阶段（第 %d 阶段）' % st.phase)
            if busy:
                return self._sr_idle()
            return self._sr_deploy(st, start[0], now, boxes)

        # 7) 战斗 HUD：真在打了。什么都不点，慢一点轮询。左上角的回合计数也算（自动战斗没有「行动结束」，
        #    以前开场 6 下盲点大多点在正在打的战场上：10/02 一天 107 下）
        if (self.find_boxes(boxes, match=HUD_RE, boundary=self.box.bottom)
                or battle_frame(boxes, self.width, self._sr_H(), comm=False)):
            st.entered = True
            self._sr_known(st, now)
            if not st.hud_seen:
                self._sr_chain('_chain_started')     # 「继续前进」之后直接进了战斗（没见到部署界面）
            st.battle_started = st.hud_seen = True
            st.start_clicks = 0
            st.label = 'battle'
            if st.comm_taps and (st.last_tap_t is None or now - st.last_tap_t >= COMM_RESET_AFTER):
                st.comm_taps = 0            # 上一段通讯对话完了，下一段重新算 12 下
                st.comm_cap_noted = False
            if not st.hud_logged:
                st.hud_logged = True
                self.log_info('开打了，战斗中不点屏幕')
            return self._sr_idle(HUD_POLL)

        # 7b) 关卡里你在用的菜单 / 面板（☰「关卡信息」、挡住了作战开始的面板）：什么都不点，
        #     也不能让下面的关卡外判断把它当成章节地图（顶上有关卡编号），排在「确认」前面 ——
        #     你点了「退出作战」弹出来的确认框要你自己决定。看到胜负之后就不管了（结算画面上
        #     万一有同名的字，别因此卡在「不动」里）
        menu = [] if st.verdict else self.find_boxes(boxes, match=IN_STAGE_MARKERS)
        if menu:
            st.entered = True
            # deploy=True：部署界面「作战开始」OCR 偶尔漏一帧也会落到这里，别因此重新计「刚进部署界面」
            self._sr_known(st, now, deploy=True)
            self._sr_say(st, 'stage_menu', '关卡里的菜单 / 面板开着（「%s」），先不动' % menu[0].name)
            return self._sr_idle()

        # 8) 结算的「确认」（下半屏）
        ok_btn = [b for b in self.find_boxes(boxes, match=SETTLE_CONFIRM) if b.y > CONFIRM_MIN_Y * H]
        on_list = self._sr_on_story_list(boxes)
        if ok_btn and on_list:
            # 「剧情战役」章节列表右下角的「确认」= 进选中的那一章，不是跳过 / 结算的确认框。10/12 00:24 主线最后一关
            # （剧情点）放完、游戏退回列表，它在「刚点过跳过」的 8 秒里点了两下（列表还没反应才没进去，I-42）
            ok_btn = []
        if ok_btn and not st.verdict:
            res = [b for b in boxes if RESOURCE_RE.search(b.name or '')]
            if res:
                # 「情报拼图补充」框：先试用中间那格的补给补上（_sr_refill）。补上了接着走
                why = self._sr_refill(st, boxes, busy)
                if why is None:
                    return None
                # 补不了 / 别的消耗 / 购买类的框：不点「确认」，停手提醒；轮询那边先不再自动开关卡（见 _stamina_hold）
                st.debug = self._save_stuck_image('resource')
                self._stamina_hold = dict(t=self._now(), what=res[0].name, logged=False)
                if st.refill_clicked:
                    did = '用补给补没成（%s），不再点' % why
                else:
                    did = '不点「确认」' + (('（没用补给补：%s）' % why) if why else '')
                st.note = ('弹出了「%s」（多半是情报拼图不够了），%s —— 你补充好之后点开关卡，我接着打'
                           % (res[0].name, did))
                self.log_info('弹出了「%s」（多半是情报拼图不够了），%s，停手交给你 —— 你补充好之后点开关卡，'
                              '我接着打' % (res[0].name, did), notify=True)
                return 'stalled'
            if not (self.find_boxes(boxes, match=VICTORY_NAMES + DEFEAT_NAMES)
                    or (st.skip_t is not None and now - st.skip_t <= SKIP_CONFIRM_WINDOW)):
                if not st.confirm_noted:
                    st.confirm_noted = True
                    self.log_info('画面上有「确认」，但不是跳过剧情 / 结算的确认框，不点：%s'
                                  % ' / '.join(b.name for b in boxes[:10] if b.name))
                ok_btn = []
        if ok_btn and self._sr_may_click(st, ok_btn[0], now):
            st.entered = True
            self._sr_known(st, now)
            if busy:
                return self._sr_idle()
            self._sr_say(st, 'confirm', '点「%s」' % ok_btn[0].name)
            self._sr_click(st, ok_btn[0], self._sr_pause(), now)
            return None

        # 8b) 画面中部的「跳过」：9/25 16:17:16、9/29 10:18:32 / 10:19:19 / 10:55:49 上游都点过
        #     一个 x≈0.52W、y 在 0.27~0.61H 的「跳过」，每次点完关卡就往下走了 —— 顶部规则管不到它
        mid = [b for b in boxes if b.name == SKIP_TEXT and b.y >= SKIP_TOP_RATIO * H]
        if mid and self._sr_may_click(st, mid[0], now):
            st.entered = True
            self._sr_known(st, now)
            if busy:
                return self._sr_idle()
            self._sr_say(st, 'skip_mid', '点画面中间的「跳过」')
            self._sr_click(st, mid[0], SKIP_AFTER_SLEEP, now)
            return None

        # 9) 弹窗：按 Esc 关掉（照搬上游）
        pop = self.find_boxes(boxes, match=pop_ups)
        if pop and self._sr_may_click(st, pop[0], now):
            st.entered = True
            self._sr_known(st, now)
            # 你刚停手不久：这个弹窗（敌人信息之类）多半是你自己点开在看的，别替你关（10/02 08:24 你看了 20 秒）
            if busy or (st.user_last_t is not None and now - st.user_last_t < USER_RECENT_GUARD):
                return self._sr_idle()
            self._sr_say(st, 'popup', '关弹窗（%s）' % pop[0].name)
            key = self._sr_key(pop[0])
            st.clicks.setdefault(key, []).append(self._now())
            self.back(after_sleep=self._sr_pause(0.3))
            return None

        # 10) 关卡外：章节地图 / 关卡详情面板 /（看到胜负之后）地图上的关卡编号
        why = None
        if is_chapter_map(boxes, self.width, self.height):
            why = '回到章节地图'
        elif self.find_boxes(boxes, match=ENTER_STAGE, boundary=self.box.right):
            why = '回到关卡详情面板'
        elif on_list:
            # 主线最后一关打完，游戏放完下一章的解锁动画直接退回「剧情战役」章节列表（剧情点打完时中间不经过章节地图）。
            # 以前这里认不出，当成认不出的画面在列表上盲点了 6 下、干等到 63 秒，你都自己去领奖励了（I-42）
            why = '回到「剧情战役」章节列表'
        elif st.verdict and self.find_boxes(boxes, match=map_re):
            why = '回到地图'
        if why:
            st.exit_rounds = exit_rounds + 1
            # 看到过胜负：立即返回。没看到：多看一帧（面板滑出的那一下不算），
            # 一进来就是关卡外、关卡里什么都没见过的，再多等一会儿（点「挑战」可能还没生效）
            if st.verdict or (st.exit_rounds >= EXIT_CONFIRM_ROUNDS
                              and (st.entered or now - st.t0 >= EXIT_MIN_SECONDS)):
                self.log_info(why)
                return 'victory' if st.victory else 'failed' if st.failed else 'left'
            return self._sr_idle(EXIT_RECHECK)
        st.entered = True

        # 11) 真的在加载：不点（0 个字块不算加载，见下面的盲点）。加载太久就不信它了，按认不出处理
        loading = [b for b in boxes if LOADING_RE.search(b.name or '')]
        if not loading:
            st.loading_since = None
        else:
            if st.loading_since is None:
                st.loading_since = now
            if now - st.loading_since < LOADING_MAX:
                st.unknown_since = st.zero_since = st.busy_since = None
                self._sr_say(st, 'loading', '加载中（%s）' % loading[0].name)
                return self._sr_idle(LOADING_SLEEP)
            self._sr_say(st, 'loading_long', '「加载中」超过 %d 秒还没完，当成普通画面处理（%d 个字块）'
                         % (LOADING_MAX, len(boxes)))

        # 12) 认不出：剧情对话 / 过场动画 / 你打开的别的界面
        return self._sr_unknown(st, boxes, now, busy)

    # ---------------------------------------------------------------- 教学 / 引导（I-13）

    def _sr_motion(self, st, now):
        """每轮记一个「画面自己动了多少」的样本：和上一轮的缩略图比。两帧之间自己点过的不算（那是点出来的变化）。"""
        th = self._sr_thumb()
        prev, pt = st.prev_thumb, st.prev_thumb_t
        st.prev_thumb, st.prev_thumb_t = th, now
        if th is None or prev is None or pt is None or now - pt > MOTION_GAP_MAX:
            return
        own = getattr(self, '_sr_own', None) or ()
        if own and own[-1][1] >= pt - 0.05:
            return                              # 两帧之间自己点过：不算画面自己动
        d = np.abs(th - prev)
        a, f = float(d.mean()), float((d > STATIC_PIX).mean())
        st.motion.append((now, a, f))
        del st.motion[:-MOTION_KEEP]
        if a > STATIC_MEAN or f > STATIC_FRAC:
            mine = max([st.last_tap_t or -1e9] + [b for _, b in own[-1:]])
            if now - mine > OWN_ECHO:
                # 游戏自己在动：没被暂停，不像教学。这一阶段先不多点，等它又长时间停住再说（_sr_overlay_ok）
                st.self_moved_t = now
                st.overlay_noresp = max(st.overlay_noresp, OVERLAY_NORESP_STOP)

    def _sr_static(self, st, now, min_samples=STATIC_MIN_SAMPLES, window=STATIC_WINDOW):
        """最近 window 秒里至少 min_samples 个样本、每个都几乎没变化：画面停着（游戏被教学暂停了）。"""
        s = [m for m in st.motion if now - m[0] <= window + 1e-6]
        return len(s) >= min_samples and all(a <= STATIC_MEAN and f <= STATIC_FRAC for _, a, f in s)

    def _sr_overlay_ok(self, st, n, now):
        """开场 6 下用完之后还能不能接着点中间：画面停着、上一下点了有新内容、没到上限。"""
        if not getattr(self, '_close_tutorial_x', True) or st.kind != 'battle':
            return False                # 开关关了：只点开场那 6 下（和说明一致）
        if st.user_last_t is not None and now - st.user_last_t < USER_RECENT_GUARD:
            return False
        if st.tutorial_taps >= TUTORIAL_TAP_CAP:
            if not st.tut_cap_noted:
                st.tut_cap_noted = True
                msg = '教学画面点了 %d 下中间还没出叉，不再点，只看 —— 你来关一下' % TUTORIAL_TAP_CAP
                alert = getattr(self, 'alert_user', None)   # 要你处理：醒目地提醒（测试桩没有就照旧 notify）
                if alert is not None:
                    alert(msg)
                else:
                    self.log_info(msg, notify=True)
                self._sr_evidence('tutorial_cap', self._sr_extra(st))
            return False
        if st.overlay_noresp >= OVERLAY_NORESP_STOP:
            # 只在「游戏自己动过、然后又停住好久」时再试：一直停着的战场（手动模式等你操作）不算
            if (st.rearms >= OVERLAY_REARM_CAP or st.self_moved_t is None
                    or (st.rearm_t is not None and st.self_moved_t <= st.rearm_t)
                    or (st.last_tap_t is not None and now - st.last_tap_t < OVERLAY_REARM_STILL)
                    or not self._sr_static(st, now, min_samples=3, window=OVERLAY_REARM_STILL)):
                return False
            # 动着动着停住了好久：像是又被暂停了（打到一半弹出的教学）。试一下，这下没反应就又不点
            st.rearms += 1
            st.rearm_t = now
            st.overlay_noresp = OVERLAY_NORESP_STOP - 1
            self.log_info('画面停了 %d 秒没动，再试着点一下中间（这一阶段第 %d 次）' % (OVERLAY_REARM_STILL, st.rearms))
            return True
        return self._sr_static(st, now)

    def _sr_refund(self, st):
        """「继续前进」之后、部署之前：点出了进展的那下不算进开场 6 下（复查 SM-3：多段对话第 7 段起就不点了）。"""
        if st.refunds < PRE_DEPLOY_REFUND_CAP:
            st.refunds += 1
            st.intro_taps -= 1

    def _sr_close_x_cv(self):
        """图像上认右上角的关闭叉（close_x.detect_close_x），返回点击用的框；认不出 / 出错返回 None。测试里换成桩。"""
        if close_x is None:
            return None
        try:
            frame = self.frame
            if frame is None or frame.shape[1] < 640:
                return None
            hits = close_x.detect_close_x(frame, band=CLOSE_X_BAND)
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            return None
        if not hits:
            return None
        h = hits[0]
        x, y, s = h['x'], h['y'], h['size']
        return self.box_of_screen((x - s / 2.0) / self.width, (y - s / 2.0) / self._sr_H(),
                                  (x + s / 2.0) / self.width, (y + s / 2.0) / self._sr_H(), name='close_x')

    def _sr_x_find(self, boxes):
        """叉在哪：先看图像，再看 OCR 有没有恰好一个「X」字落在右上角的带子里。-> (框, 来源) / (None, None)"""
        W, H = max(1, self.width or 1), self._sr_H()
        x0, y0, x1, y1 = CLOSE_X_BAND
        try:
            b = self._sr_close_x_cv()
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            b = None
        if b is not None:
            return b, 'cv'
        for b in boxes:
            if b.name and X_TOKEN_RE.match(b.name):
                cx, cy = b.x + b.width / 2.0, b.y + b.height / 2.0
                if x0 * W <= cx <= x1 * W and y0 * H <= cy <= y1 * H and 0.02 * H <= b.height <= 0.08 * H:
                    return b, 'ocr'
        return None, None

    def _sr_extra(self, st, boxes=()):
        """教学相关的现场记录（写进图旁边的 .json）：读到的字、点了几下、画面变化样本、每下点完的反应。"""
        return {'names': [b.name for b in list(boxes)[:12] if b.name], 'intro_taps': st.intro_taps,
                'tutorial_taps': st.tutorial_taps, 'x_clicks': st.x_clicks_run, 'overlay_noresp': st.overlay_noresp,
                'motion': [(round(t - st.t0, 2), round(a, 3), round(f, 4)) for t, a, f in st.motion[-12:]],
                'responses': list(st.responses), 'x_cand': st.x_cand}

    def _sr_evidence(self, tag, extra=None):
        """存现场图，再把 extra 补进同名的 .json。不改 _save_stuck_image 的签名（测试里好几个桩重写了它）。"""
        path = self._save_stuck_image(tag)
        note = getattr(self, '_save_case_note', None)
        if path and extra and note and os.path.exists(path):
            note(path, tag, extra)
        return path

    def _sr_close_x(self, st, boxes, now, busy, n, dialogue):
        """右上角出现叉（教学 / 引导的关闭按钮）：确认两帧、画面静止之后点它的中心。不盲点。"""
        if not getattr(self, '_close_tutorial_x', True) or st.kind != 'battle' or st.verdict:
            st.x_cand = None
            return None
        box, src = self._sr_x_find(boxes)
        if box is None:
            st.x_cand = None
            st.x_noeffect = 0
            return None
        W = max(1, self.width or 1)
        cx, cy = box.x + box.width / 2.0, box.y + box.height / 2.0
        tol = CLOSE_X_SAME * W

        def same(p):
            return p is not None and abs(cx - p[0]) <= tol and abs(cy - p[1]) <= tol

        if (st.x_last is not None and not st.x_last[4] and same(st.x_last)
                and now - st.x_last[2] <= CLOSE_X_RECHECK):
            # 点完之后第一次又看到它在原处：判一次有没有用（每点一下只判一次；画面或字变了都算有用）
            st.x_last = st.x_last[:4] + (True,)
            if self._sr_changed(st.x_last[3], self._sr_thumb()) or st.x_fresh:
                st.x_noeffect = 0
            else:
                st.x_noeffect += 1
            if st.x_noeffect >= CLOSE_X_NOEFFECT_STOP and not st.x_blocked:
                st.x_blocked = True
                self.log_info('右上角的叉点了没反应，这一阶段不再点它')
                self._sr_evidence('tutorial_x_noeffect', self._sr_extra(st, boxes))
        prev = st.x_cand
        if same(prev):
            first_t, count = prev[2], prev[3] + (1 if now - prev[4] >= CLOSE_X_MIN_GAP else 0)
        else:
            first_t, count = now, 1
        st.x_cand = (cx, cy, first_t, count, now, src)
        gate = None
        target = None
        if busy:
            gate = 'busy'
        elif st.user_last_t is not None and now - st.user_last_t < USER_RECENT_GUARD:
            gate = 'user_recent'
        elif n > self._sr_box_cap(st, dialogue, TAP_MAX_BOXES):
            gate = 'boxes'
        elif st.x_blocked:
            gate = 'noeffect'
        elif st.x_clicks_phase >= CLOSE_X_PHASE_CAP or st.x_clicks_run >= CLOSE_X_RUN_CAP:
            gate = 'cap'
        elif count < 2 or not self._sr_static(st, now, min_samples=1, window=0.0):
            if now - first_t <= CLOSE_X_PENDING:
                return 'pending'
            gate = 'unstable' if count < 2 else 'moving'
        if gate is None:
            target = self.box_of_screen(box.x / W, box.y / self._sr_H(), (box.x + box.width) / W,
                                        (box.y + box.height) / self._sr_H(), name='关闭叉')
            if not self._sr_may_click(st, target, now):
                gate = 'anti_repeat'
        if gate is not None:
            if gate not in ('busy', 'user_recent') and not st.x_refused_saved:
                st.x_refused_saved = True
                ex = self._sr_extra(st, boxes)
                ex['gate'] = gate
                self.log_info('右上角像是有叉，但没点（%s）' % gate)
                self._sr_evidence('tutorial_x_refused', ex)
            return None
        if st.x_clicks_run < 3:
            self._sr_evidence('tutorial_x', self._sr_extra(st, boxes))
        self.log_info('右上角出现「×」（教学 / 引导，%s），点掉' % src)
        st.x_clicks_phase += 1
        st.x_clicks_run += 1
        st.x_last = (cx, cy, now, self._sr_thumb(), False)
        st.x_fresh = False
        st.x_cand = None
        st.unknown_since = st.zero_since = None
        st.tap_fail = 0
        st.label = 'close_x'
        self._sr_click(st, target, SKIP_AFTER_SLEEP, now)
        return 'clicked'

    # ---------------------------------------------------------------- 部署

    def _sr_start_texts(self, st, boxes):
        """点过「作战开始」、又回到了部署界面：记下画面上新冒出来的字。
        9/29 18:12 / 18:18 MA-1-3：只放了助战，点 4 次「作战开始」都没反应，连「阵容没上满」的确认框都没有，
        当时没记下画面上闪过什么提示，查不出为什么。"""
        for b in boxes:
            k = text_skeleton(b.name)
            if k and k not in st.start_base:
                st.start_changed = True
                if b.name not in st.start_new and len(st.start_new) < 12:
                    st.start_new.append(b.name)

    def _sr_start_note(self, st):
        if st.start_new:
            return '；点完之后画面上新出现的字：%s' % ' / '.join(st.start_new)
        return '；点完之后画面上没出现新的字'

    def _sr_min_sortie(self, st, boxes):
        """点完「作战开始」之后弹了「本关至少需要出击N人！」-> N；没弹 -> None。"""
        for t in [b.name for b in boxes if b.name] + list(st.start_new):
            m = MIN_SORTIE_RE.search(t)
            if m:
                return int(m.group(1))
        return None

    def _sr_read_card(self):
        """左下角详情卡 -> own_units.read_unit_card 的结果；读不了当没选人。"""
        reader = getattr(self, 'read_unit_card', None)
        try:
            if reader is not None:
                return reader()
        except UserTakeoverError:
            raise
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            pass
        return dict(support=False, name=None, level=None, candidates=[], reason='unreadable')

    def _sr_fill_extra(self):
        """补人半路停了（你接手 / 出错）：部署数比核对着放的多 -> risky 说明；对得上 / 核对不了 -> None。"""
        check = getattr(self, 'fill_unexplained', None)
        try:
            why = check(self.read_deploy_count()) if check else None
        except UserTakeoverError:
            raise
        except STOP_EXCEPTIONS:
            raise
        except Exception:
            why = None
        return ('多出来的那个人（%s，没核对是谁）' % why) if why else None

    def _sr_lineup_ok(self, need):
        """部署数读得到、而且已经够 need 了吗。"""
        count = self.read_deploy_count()
        return bool(count and count[0] >= need)

    def _sr_note_support(self):
        """借完助战：记下场上的助战叫什么（补人时不上同一个角色）。左下角详情卡上「支援 + 名字」最准；
        读不到就用这次借人选中的名字（借战力最高的替补时那个是 None）。"""
        card = self._sr_read_card()
        name = card['name'] if card['support'] else None
        self._phase_support = name or getattr(self, '_picked_target', None) or None

    def _sr_fill_avoid(self):
        """补人要排除的助战名 -> (名字 / None, 知不知道)。

        - 借完读到的助战名，或者弹提示时详情卡上的「支援 + 名字」-> (名字, True)
        - 详情卡上有「支援」、名字读不出 -> (None, False)：场上多半有助战、不知道是谁，不补
        - 别的情况 -> (None, True)：没认出场上有助战。**不拿「没填助战角色名」当证据**（你可能自己借了一个）——
          fill_own_units 会拿场上的人数核对：部署数和「我的人形」里认准打勾的对不上，就不补
        """
        name = getattr(self, '_phase_support', None)
        if name:
            return name, True
        card = self._sr_read_card()                    # 弹提示时详情卡上多半还是助战（10/10 100537_start）
        if card['support']:
            if card['name']:
                self._phase_support = card['name']
                return card['name'], True
            return None, False
        return None, True

    def _sr_short(self, st, need, count, why, res=None, risky=None):
        """补人停手：把这一阶段几轮补上的 / 没上的都带上（停手提醒里写），抛 LineupShortError。
        risky = 场上可能站着个不能上的人（见 own_units.LineupShortError）。"""
        if res:
            st.fill_added += [n for n in res.get('added', ()) if n not in st.fill_added]
            st.fill_skipped += [x for x in res.get('skipped', ()) if x not in st.fill_skipped]
        raise LineupShortError(need, count, why, st.fill_added, st.fill_skipped, risky=risky)

    def _sr_refill(self, st, boxes, busy):
        """「情报拼图补充」框：选中间那格（补充道具）、点「确认」，补上了就再点「挑战」（你 10/11 说的；认格子见 stamina_refill）。

        右边那格是晶体货币，一下都不点：只点中间那格的位置，点完看橙色选中框确实挪到了中间才点「确认」。
        每关只试一次；试过之后 REFILL_GUARD 秒内又弹出来（补完还不够 / 没生效 / 框没关、轮询又接手）也不再补 ——
        「确认」绝不会因为重试被点第二下。补的途中你动了键鼠就停。

        :return: None = 补上了；否则是没补 / 没补成的原因（'' = 没开这个功能、或者不是这种框），调用方照旧停手
        """
        if not getattr(self, '_refill_stamina', False):
            return ''
        W, H = self.width, self._sr_H()
        title = refill_title(boxes)
        if title is None:
            return ''
        if busy:
            return '你在操作'
        if st.refilled:
            return '这一关已经补过一次，又弹出来了'
        now = self._now()
        last = getattr(self, '_refill_t', None)
        if last is not None and now - last < REFILL_GUARD:
            return '%d 秒前刚试过补，又弹出来了' % int(now - last)
        stock, white, red = slot_stock(self.frame, title, 1, W, H)
        if stock is False:
            return '中间那格是 0 个'
        if stock is None:
            return '中间那格有没有货看不清（白字 %.3f / 红字 %.3f）' % (white, red)
        n = parse_count(boxes, title, 1, W, H)
        before = stamina(boxes, W, H)
        st.refilled = st.refill_clicked = True        # 不管成没成，这一关只试一次
        self._refill_t = now                          # 下一关 / 轮询再接手也一样：REFILL_GUARD 秒内不再试
        self._sr_known(st, now)
        guard = (getattr(self, '_sr_guarding', False), getattr(self, '_sr_guard_what', None))
        self._sr_guarding, self._sr_guard_what = True, '补情报拼图'
        try:
            x, y = icon_point(title, 1, W, H)
            self.click(int(round(x)), int(round(y)), after_sleep=REFILL_PICK_SLEEP, name='refill_item')
            self.next_frame()
            b2 = self._sr_ocr()
            t2 = refill_title(b2)
            if t2 is None:
                return '点了中间那格，框不见了'
            sel, ratios = selected_slot(self.frame, t2, W, H)
            if sel != 1:
                return '点了中间那格，选中框没挪过去（三格的橙框 %s）' % (ratios,)
            if quantity(b2) == 0:
                return '选中了中间那格，数量却是 0'
            ok_btn = confirm_button(b2, t2, H)
            if ok_btn is None:
                return '找不到「确认」'
            self.log_info('情报拼图不够：选了「情报拼图补充」中间那格（补给%s），点「确认」'
                          % ((' %d 个' % n) if n is not None else ''))
            self.click_box(ok_btn, after_sleep=REFILL_CONFIRM_SLEEP)
        except UserTakeoverError:
            return '补的途中你动了键鼠'
        finally:
            self._sr_guarding, self._sr_guard_what = guard
        b3 = []
        for i in range(REFILL_CLOSE_TRIES):
            self.next_frame()
            b3 = self._sr_ocr()
            if refill_title(b3) is None:
                break
            if i + 1 < REFILL_CLOSE_TRIES:
                self.sleep(REFILL_CLOSE_WAIT)
        if refill_title(b3) is not None:
            return '点了「确认」框还在'
        if any(RESOURCE_RE.search(b.name or '') for b in b3):
            return '点了「确认」又弹出了别的框'
        self._refill_t = self._now()
        self._refill_used = getattr(self, '_refill_used', 0) + 1
        self._stamina_hold = None
        after = stamina(b3, W, H)

        def fmt(s):
            return '%d/%d' % s if s else '读不到'
        self.log_info('用补给补上了情报拼图：%s -> %s（这次运行用了 %d 个）'
                      % (fmt(before), fmt(after), self._refill_used), notify=True)
        go = [b for b in self.find_boxes(b3, match=ENTER_STAGE, boundary=self.box.right) if '挑战' in b.name]
        t = self._now()
        if go and self._sr_may_click(st, go[0], t):
            self._sr_known(st, t)
            self.log_info('补完了，再点「挑战」')
            self._sr_click(st, go[0], self._sr_pause(), t)
        return None

    def _sr_fill(self, st, need):
        """「本关至少需要出击N人」：从「我的人形」补到 N 个（own_units.fill_own_units），补完下一帧再点「作战开始」。
        补人关着 / 补了两轮还弹 / 补不上：抛 LineupShortError，轮询停手、说清差几个人
        （以前接着点满 4 次、5 分钟后重试又去借一次已经在场上的助战）。"""
        count = self.read_deploy_count()
        have = ('%d/%d' % count) if count else '读不到'
        cfg = getattr(self, 'config', None) or {}
        filler = getattr(self, 'fill_own_units', None)
        if not cfg.get('缺人时从我的人形补人', True) or filler is None:
            self._sr_short(st, need, count, '「缺人时从我的人形补人」没开')
        if st.fill_rounds >= OWN_FILL_ROUNDS:
            self._sr_short(st, need, count, '从「我的人形」补了 %d 轮，点「作战开始」还是提示' % st.fill_rounds)
        if count and count[1] < need:
            # 提示要 N 个、这一关最多放的却比 N 少：读数多半错了。不补，也别让轮询「一看够了就开打」（会来回弹）
            self._sr_short(st, need, count, '提示要 %d 个人，可部署上限读成 %d（读数可能不对）' % (need, count[1]))
        avoid, known = self._sr_fill_avoid()
        if not known:
            self._sr_short(st, need, count, '详情卡上是助战、读不出名字 —— 怕补上和助战同一个角色，先不补')
        st.fill_rounds += 1
        self.log_info('点「作战开始」弹了「本关至少需要出击%d人」（现在 %s）：从「我的人形」补人%s'
                      % (need, have, ('，不上和助战「%s」同一个角色的' % avoid) if avoid else ''), notify=True)
        self._sr_guarding = True
        self._sr_guard_what = '补人'
        try:
            res = filler(need, avoid=avoid)
        except UserTakeoverError as e:
            e.need = need                      # 轮询停手等你：放够 N 个才替你开打
            # 刚点了卡、还没核对你就接手了 / 部署数比核对着放的多：场上可能有没核对的人
            e.risky = getattr(self, '_fill_risky', None) or self._sr_fill_extra()
            e.added = st.fill_added + [n for n in getattr(self, '_fill_added', ()) if n not in st.fill_added]
            raise
        except LineupShortError as e:
            # own_units 停手（场上可能多了个没核对的人 risky）：带上前几轮补上的人
            self._sr_short(st, need, e.count, e.why, dict(added=e.added, skipped=e.skipped), risky=e.risky)
        except OverlayBlockedError as e:
            # 补到一半弹出了教学 / 提示（I-43）：场上没有没核对的人就先停下，关掉它之后再点「作战开始」、再补；
            # 有的话照补人的规矩停手（risky：不碰部署界面、不替你开打）
            risky = getattr(self, '_fill_risky', None)
            if risky:
                self._sr_short(st, need, self.read_deploy_count(), '补人途中%s' % str(e).replace('部署界面上', ''),
                               dict(added=getattr(self, '_fill_added', ()), skipped=getattr(self, '_fill_skipped', ())),
                               risky=risky)
            self.log_info('补人途中%s，先停下关掉它，之后再补' % str(e).replace('部署界面上', ''))
            self._sr_known(st, self._now(), deploy=True)
            return None
        except (SupportStateError, TileNotSelectedError):
            raise
        except STOP_EXCEPTIONS:
            raise
        except Exception as e:
            self.log_error('补人出错了', e)
            self._sr_short(st, need, self.read_deploy_count(), '补人出错了：%r' % (e,),
                           dict(added=getattr(self, '_fill_added', ()), skipped=getattr(self, '_fill_skipped', ())),
                           risky=getattr(self, '_fill_risky', None) or self._sr_fill_extra())
        finally:
            self._sr_guarding = False
            self._sr_guard_what = None
        if not res.get('added') or res.get('why'):
            # 一个都没补上，或者半路停了（why）—— 都是人不够：别再点「作战开始」，停手
            self._sr_short(st, need, res.get('count', count), res.get('why') or '没补上人', res)
        st.fill_added += [n for n in res['added'] if n not in st.fill_added]
        st.fill_skipped += [x for x in res.get('skipped', ()) if x not in st.fill_skipped]
        # 补上了：重新开始点「作战开始」（又是 4 次机会）。界面其实还没开打
        st.start_clicks = 0
        st.last_start_t = None
        st.start_base = set()
        st.start_new = []
        st.start_changed = False
        st.battle_started = False
        self._sr_known(st, self._now(), deploy=True)
        return None

    def _sr_deploy(self, st, start, now, boxes=()):
        skip_once = bool(getattr(self, '_skip_borrow_once', False))
        borrow = self._should_borrow([start])
        if skip_once:
            # 你手动放好人之后替你点「作战开始」：这一阶段都别再借（_should_borrow 只拦一次）
            self._support_done = True
        if borrow:
            self._support_done = True
            opts = dict(self.support_options() or {})
            if st.phase > 1:
                opts['reenter_times'] = 0        # 前面的阶段已经打赢了，退关会把战果一起丢掉
            # 借人期间每次键鼠输入之前都看你一眼（见 _sr_input）。自己的动作逐次记时间窗，
            # 不再把整段借人都算成「自己的动作」—— 那样会把你中途的操作一起盖掉
            self._sr_guarding = True
            self._picked_target = None           # 以前不清：会带着上一关借的人的名字（补人时拿它排除同一个角色）
            got = False
            try:
                got = self.borrow_support(**opts)
            except OverlayBlockedError as e:
                # 借到一半弹出了教学 / 提示（I-43）：这一阶段还没借上，关掉它之后重新借（不然会不带助战直接开打）
                self._support_done = False
                self.log_info('借助战途中%s，先停下关掉它，再重新借人' % str(e).replace('部署界面上', ''))
                self._sr_known(st, self._now(), deploy=True)
                return None
            except (SupportStateError, TileNotSelectedError, UserTakeoverError):
                raise
            except STOP_EXCEPTIONS:
                raise
            except Exception as e:
                self.log_error('借助战出错，按原流程继续', e)
            finally:
                self._sr_guarding = False
            if got:
                self._sr_note_support()
            # 借人过程中界面动过：重新截图再判断
            self._sr_known(st, self._now(), deploy=True)
            return None
        if st.start_clicks:
            self._sr_start_texts(st, boxes)
            need = self._sr_min_sortie(st, boxes)
            if need and not self._sr_lineup_ok(need):
                return self._sr_fill(st, need)
            # 提示还留在 start_new 里、可部署数已经够了（提示是上一下点的 / 你刚放够）：照常等、照常重点
        wait = START_RETRY_SECONDS + (START_TRANSITION_EXTRA if st.start_changed else 0.0)
        if st.last_start_t is not None and now - st.last_start_t < wait:
            return self._sr_idle()
        if now - st.deploy_seen_t < START_SETTLE:
            return self._sr_idle(0.3)
        if st.start_clicks >= START_CLICK_LIMIT:
            st.debug = self._save_stuck_image('start')
            st.note = '点了 %d 次「作战开始」还停在部署界面%s' % (st.start_clicks, self._sr_start_note(st))
            self.log_info('点了 %d 次「作战开始」还停在部署界面，停手交还给轮询%s'
                          % (st.start_clicks, self._sr_start_note(st)), notify=True)
            return 'stalled'
        st.start_clicks += 1
        st.last_start_t = now
        if st.start_clicks > 1:
            self.log_info('点了「作战开始」界面没反应，再点一次（第 %d 次）%s'
                          % (st.start_clicks, self._sr_start_note(st)))
        st.start_base |= set(k for k in (text_skeleton(b.name) for b in boxes) if k)
        st.start_changed = False
        self.click_box(start, after_sleep=self._sr_pause())
        if st.post_continue:
            # 「继续前进」之后部署前那段（对话等）用掉的开场 / 教学额度，别算到真正开打之后
            st.post_continue = False
            st.intro_taps = 0
            st.reset_tutorial()
        st.battle_started = True
        return None

    # ---------------------------------------------------------------- 认不出

    def _sr_dialogue_like(self, boxes):
        """像不像剧情对话：除去顶部按钮条（自动 / 记录 / 跳过），字块大多在下半屏的对话框里。
        菜单、活动地图、战场上的名字条都是满屏散开的。"""
        H = self._sr_H()
        body = [b for b in boxes if b.y >= SKIP_TOP_RATIO * H]
        if len(body) < 2:
            return False
        low = [b for b in body if b.y + b.height / 2.0 >= DIALOG_LOW_RATIO * H]
        return len(low) >= DIALOG_LOW_SHARE * len(body)

    def _sr_unknown(self, st, boxes, now, busy):
        n = len(boxes)
        if st.unknown_since is None:
            st.unknown_since = now
        if n == 0:
            if st.zero_since is None:
                st.zero_since = now
        else:
            st.zero_since = None
        # 读到的字变了 = 有进展（对话换了一句）。只看缩略图的话，对话框里换一行字整幅画面几乎不变，
        # 点几下就会被当成「点了没用」而不再点，多段剧情就卡在第二段。
        # 但倒计时、伤害数字、OCR 抖动（一个字认错、字块时有时无、一句被拆成两块）不算：以前只要字集合
        # 不一样就算进展，这种画面上「点几下没用就不点」永远到不了，一直点到 150 秒卡住（见 text_skeleton）
        sig = frozenset(k for k in (text_skeleton(b.name) for b in boxes) if k)
        fresh = [k for k in sig if not any(_seen(k, h) for h in st.sig_hist)]
        if st.sig_hist and fresh:
            st.last_progress_t = now
            st.tap_fail = 0
            if (st.post_continue and not st.battle_started and st.intro_taps > 0 and st.last_tap_t is not None
                    and now - st.last_tap_t <= OWN_ECHO + TAP_SETTLE and self._sr_dialogue_like(boxes)):
                self._sr_refund(st)     # 部署前的对话点出了下一句
            elif (st.hud_seen and not st.verdict and st.battle_taps > 0 and st.last_tap_t is not None
                  and now - st.last_tap_t <= OWN_ECHO + TAP_SETTLE and st.battle_refunds < COMM_TAP_CAP
                  and self._sr_dialogue_like(boxes)):
                st.battle_taps -= 1     # 战斗里的对话点出了下一句：这下不算进 3 下（见 COMM_RESET_AFTER 上面）
                st.battle_refunds += 1
            if st.tap_given_up:
                st.tap_given_up = False
                self.log_info('画面上的字在变（%s），恢复「点屏幕唤出跳过」' % fresh[0])
        st.sig_hist.append(sig)
        del st.sig_hist[:-SIG_HISTORY]
        ov_fresh = [k for k in sig if not _seen(k, st.ov_seen)]
        if st.resp_pending is not None and ov_fresh:
            st.resp_pending['fresh'] = True
        if ov_fresh and st.x_last is not None and not st.x_last[4]:
            st.x_fresh = True
        st.ov_seen.extend(ov_fresh)
        del st.ov_seen[:-200]
        if st.resp_pending is not None and st.motion and st.motion[-1][0] == now:
            _t, _a, _f = st.motion[-1]
            _settled = _a <= STATIC_MEAN and _f <= STATIC_FRAC
            # 点之前画面就停着、点完换了新内容又停稳：才像翻了一页教学（点在正在打的战场上不算，复查 BS-2）
            _ok = (st.resp_pending['still'] and _settled
                   and (st.resp_pending['novel'] or st.resp_pending['fresh']))
            st.overlay_noresp = 0 if _ok else st.overlay_noresp + 1
            st.responses.append((st.resp_pending['diff'], st.resp_pending['novel'], st.resp_pending['fresh'],
                                 round(_a, 3), _ok))
            del st.responses[:-12]
            st.resp_pending = None
        st.unknown_sig = sig
        dialogue = self._sr_dialogue_like(boxes)

        # 排查用：一关里第一次连续 20 秒认不出、字也没变，存一张图、记下读到的字（见 STUCK_UNKNOWN_SAVE）
        if (not st.unknown_saved and not (st.battle_started and not st.verdict)
                and now - st.last_progress_t >= STUCK_UNKNOWN_SAVE):
            st.unknown_saved = True
            path = st.unknown_path = self._save_stuck_image('unknown')
            st.unknown_known_t = st.last_known_t
            self.log_info('连续 %d 秒认不出画面、也没变化（%d 个字块%s）：%s%s' % (
                int(now - st.last_progress_t), n, '，像对话' if dialogue else '',
                ' / '.join(b.name for b in boxes[:12] if b.name) or '（没有字）',
                '，现场图 %s' % path if path else ''))

        # 看到胜负之后，字多又认不出的界面持续 5 秒：多半已经回到了非标准的关卡外界面（活动地图等）。
        # 对话框布局的不算（打完之后的剧情）
        if st.verdict and n > TAP_MAX_BOXES and not dialogue:
            if st.busy_since is None:
                st.busy_since = now
            if now - st.busy_since >= AFTER_VERDICT_EXIT_SECONDS:
                self.log_info('结算之后一直是认不出的界面，当作已经回到关卡外')
                return 'victory' if st.victory else 'failed'
        else:
            st.busy_since = None

        st.comm = any(comm_box(b, self.width, self._sr_H()) for b in boxes)
        xr = self._sr_close_x(st, boxes, now, busy, n, dialogue)
        if xr == 'clicked':
            return None

        stalled = self._sr_stall(st, now)
        if stalled:
            return stalled

        at = None if (busy or xr == 'pending') else self._sr_tap_at(st, n, dialogue, now)
        if at is not None and now >= at:
            return self._sr_tap(st, n, now, boxes)
        # 快到能点的时候只睡到那一刻：每轮还有 ~0.2s 的 OCR，按整轮睡会把「跳过」拖过 3 秒
        nap = IDLE_SLEEP if at is None else min(IDLE_SLEEP, max(0.05, at - self._now()))
        return self._sr_idle(nap)

    def _sr_stall(self, st, now, limit=None):
        """连续这么久没认出任何界面：存现场图、返回 'stalled'；还没到返回 None。

        默认开打之后、结算之前用 BATTLE_STALL_SECONDS，其余 STALL_SECONDS。「继续前进」之后没点作战开始的，
        只有画面最近自己在动（像直接开打了）才按战斗算；停住不动的还是 150 秒交还（复查 SM-3）。
        """
        if limit is None:
            fighting = st.battle_started or (st.post_continue and st.self_moved_t is not None
                                             and now - st.self_moved_t < STALL_SECONDS)
            limit = BATTLE_STALL_SECONDS if (fighting and not st.verdict) else STALL_SECONDS
        if now - st.last_known_t < limit:
            return None
        if st.unknown_path and st.unknown_known_t == st.last_known_t:
            st.debug = st.unknown_path        # 还是存过图的那一段认不出，别再存一张一样的
        else:
            st.debug = self._save_stuck_image('stuck')
        st.note = '连续 %d 秒认不出画面' % int(now - st.last_known_t)
        self.log_info('连续 %d 秒认不出画面，停手交还给轮询' % int(now - st.last_known_t), notify=True)
        return 'stalled'

    def _sr_box_cap(self, st, dialogue, cap):
        """这种画面最多几个字块还肯点：剧情关、对话框布局放宽（见 STORY_TAP_MAX_BOXES）。"""
        if st.kind == 'story':
            return max(cap, STORY_TAP_MAX_BOXES)
        if dialogue:
            return max(cap, DIALOG_TAP_MAX_BOXES)
        return cap

    def _sr_tap_at(self, st, n, dialogue=False, now=None):
        """这种画面最早什么时候可以盲点（时间戳，和帧时间比）；不该点返回 None。"""
        if not getattr(self, '_tap_to_reveal_skip', True) or st.tap_given_up:
            return None
        gap = TAP_INTERVAL
        if st.comm and not st.verdict:
            # 战斗里的通讯对话：点一下才继续，不受开场 / HUD 的限制
            if st.comm_taps >= COMM_TAP_CAP:
                if not st.comm_cap_noted:
                    st.comm_cap_noted = True
                    self.log_info('战斗里的通讯对话点了 %d 下还没完，先不点了 —— 卡住的话你点一下' % COMM_TAP_CAP, notify=True)
                return None
            at = st.unknown_since + COMM_TAP_WAIT
            if st.last_tap_t is not None:
                at = max(at, st.last_tap_t + COMM_TAP_INTERVAL)
            return at
        if (st.battle_started or st.post_continue) and not st.verdict:
            if st.hud_seen:
                # 真在打：只点战斗里的过场（几乎没字、持续好几秒），每阶段最多 3 次
                if (n > BATTLE_TAP_MAX_BOXES and not dialogue) or st.battle_taps >= BATTLE_TAP_CAP:
                    return None
                start, wait = st.unknown_since, BATTLE_TAP_WAIT
            else:
                # 点了作战开始、还没见过 HUD：开场过场 / 对话，比开打前克制
                if n > self._sr_box_cap(st, dialogue, TAP_MAX_BOXES):
                    return None
                gap = INTRO_TAP_INTERVAL
                if st.intro_taps >= INTRO_TAP_CAP:
                    if not self._sr_overlay_ok(st, n, self._now() if now is None else now):
                        return None
                    gap = TUTORIAL_TAP_INTERVAL
                start, wait = ((st.zero_since, max(INTRO_TAP_WAIT, TAP_ZERO_WAIT)) if n == 0
                               else (st.unknown_since, INTRO_TAP_WAIT))
        else:
            cap = self._sr_box_cap(st, dialogue, TAP_MAX_BOXES)
            if n > cap:
                # 字多：菜单 / 你打开的界面，一律不点（每关只说一次，排查用）
                if not st.cap_logged:
                    st.cap_logged = True
                    self.log_info('认不出、字又多的画面（%d 个字块），当成菜单不点' % n)
                return None
            start, wait = ((st.zero_since, TAP_ZERO_WAIT) if n == 0
                           else (st.unknown_since, TAP_TEXT_WAIT))
        at = start + wait
        if st.last_tap_t is not None:
            at = max(at, st.last_tap_t + gap)
        return at

    def _sr_tap(self, st, n, now, boxes=()):
        """点一下屏幕中央，0.35 秒后只 OCR 顶部一条找「跳过」，找到立刻点。"""
        before = self._sr_thumb()
        still = self._sr_static(st, now)
        st.last_tap_t = now
        st.tap_total += 1
        if st.comm and not st.verdict:
            st.comm_taps += 1
            if st.comm_taps == 1:
                self.log_info('战斗里的通讯对话（通信中），点一下继续')
        elif (st.battle_started or st.post_continue) and not st.verdict:
            if st.hud_seen:
                st.battle_taps += 1
            elif st.intro_taps >= INTRO_TAP_CAP:
                st.tutorial_taps += 1
                if not st.tut_noted:
                    st.tut_noted = True
                    self.log_info('画面停着、点了有反应（像教学 / 引导）：开场盲点用完后接着点中间，最多 %d 下' % TUTORIAL_TAP_CAP)
                    self._sr_evidence('tutorial', self._sr_extra(st, boxes))
            else:
                st.intro_taps += 1
        self.log_info('认不出画面（%d 个字块），点屏幕中央唤出「跳过」（第 %d 次）：%s' % (
            n, st.tap_total, ' / '.join(b.name for b in list(boxes)[:6] if b.name) or '（没有字）'))
        self.click_relative(*SKIP_REVEAL_POINT, after_sleep=0)
        self.sleep(TAP_SETTLE)
        self.next_frame()
        skip = self._sr_top_skip(self._sr_ocr(box=self.box_of_screen(*SKIP_BAND)))
        t = self._now()
        if skip and self._sr_may_click(st, skip[0], t):
            self.log_info('「跳过」出来了，点掉')
            if st.post_continue and not st.battle_started and st.intro_taps > 0:
                self._sr_refund(st)     # 部署前的对话：这下点出了跳过，不算进开场 6 下
            st.label = 'skip'
            self._sr_known(st, t)
            self._sr_click(st, skip[0], SKIP_AFTER_SLEEP, t)
            return None
        after = self._sr_thumb()
        resp = self._sr_changed(before, after)
        diff = None if (before is None or after is None) else round(float(np.mean(np.abs(before - after))), 2)
        novel = bool(resp and after is not None and all(
            float(np.mean(np.abs(after - p))) > THUMB_CHANGE for p in st.tap_thumbs))
        st.tap_thumbs = (st.tap_thumbs + [x for x in (before, after) if x is not None])[-4:]
        st.resp_pending = {'diff': diff, 'novel': novel, 'fresh': False, 'still': still}
        st.prev_thumb, st.prev_thumb_t = after, self._now()
        if st.battle_started and not st.verdict and not st.hud_seen and st.intro_taps >= INTRO_TAP_CAP \
                and not st.cap_noted:
            st.cap_noted = True
            self.log_info('开场盲点用完（%d 次），之后只在像教学的画面上接着点' % INTRO_TAP_CAP)
        if resp:
            st.tap_fail = 0
        else:
            st.tap_fail += 1
            if st.tap_fail >= TAP_GIVE_UP:
                st.tap_given_up = True
                self.log_info('连点 %d 次没进展，不再盲点，只看' % st.tap_fail)
        return None
