# -*- coding: utf-8 -*-
"""打赢的关卡记进文件：重开任务、重启程序也记得（ok-gf2 扩展）。不定义 BaseTask 子类。

10/10 20:35「镜茧与囚蝶」：LA-1-3～1-5 是它自己用「继续前进」连着打完的，记忆里只有 LA-1-3；你 20:35 重开了一次任务，
连这一条也没了。卡片上的勾又没认出来（I-28），就把 LA-1-3、LA-1-4 又打了一遍。
认勾总会碰上认不出的新样式，所以它自己打赢的关按「游戏客户端 + 活动名 + 编号」记进 configs/stage_memory.json（客户端 = 游戏 exe 的路径，两个服 / 两个号分开，I-54），
选关时记录里有的一律算打过，不看卡片上有没有勾：
  - 结算页出现「继续前进」= 这一关打赢了，当场就记（后面卡住、你重开任务也不丢）
  - 「继续前进」开的是同一前缀的下一关，编号按顺序往后推（10/02 日志 28 次全是这样：UA-1-3 一路连打到 UA-1-7、
    7-9 -> 7-10、UA-1-8 -> UA-1-10；下一关是剧情点时结算页只有「确认」，连不过去）
  - 只记战斗关：剧情点重看一遍不花情报拼图，照旧只记这次运行
  - 读不出活动名 / 章节名的地图只记这次运行：不同活动的编号可能撞上
删掉这个文件 = 全忘掉（不用重启）；设置里「记住打过的关卡」关掉 = 不读也不写。
"""
import json
import os
import re
import shutil
import time
import unicodedata

LEDGER_ENV = 'GF2_STAGE_MEMORY'       # 测试用：'off' = 不读不写；也可以指向别的文件
LEDGER_DEFAULT = os.path.join('configs', 'stage_memory.json')
_CODE_TAIL_RE = re.compile(r'^(.*?)(\d+)$')
_TITLE_VARIANTS = str.maketrans({'煙': '烟', '·': '', '・': '', '.': '', '•': '', ' ': ''})
# OCR 有时把简体字读成繁体：10/11 18:07 章节列表上的「疊影效应」（地图上是「叠影效应」，换章停了两次）、记录里的「虛境新始」、
# 10/14 物资页签上的「逃离賽博格」（剧情页签上读成「逃离赛博格」，当成两个活动，两个页签来回切了 4 次，I-59）。
# 比较时两边都转成简体，只会让「同一章」更容易对上。前面一行是碰到过的，后面是活动名 / 章节名里常见、繁简不同的字
_TITLE_T2S = str.maketrans({p[0]: p[1] for p in (
    '疊叠 應应 異异 簡简 諧谐 週周 駐驻 變变 於于 澀涩 荊荆 與与 車车 織织 進进 繭茧 遊游 樂乐 園园 沒没 員员 銀银 '
    '虛虚 紀纪 禮礼 讚赞 贊赞 訴诉 隱隐 時时 爾尔 錯错 錨锚 遠远 點点 頻频 戰战 鏡镜 賽赛 '
    '離离 費费 戲戏 夢梦 華华 歸归 劍剑 鬥斗 龍龙 風风 雲云 靈灵 書书 聲声 會会 歲岁 節节 鏈链 識识 憶忆 '
    '鳥鸟 魚鱼 貓猫 獵猎 擊击 歡欢 號号 響响 飛飞 門门 開开 關关 長长 東东 陽阳 陰阴 燈灯 熱热 電电 '
    '國国 動动 對对 葉叶 島岛 嶼屿 灣湾 際际 憶忆 歷历 傳传 說说 語语 謎谜 錄录 終终 結结 線线 絲丝 '
    '紅红 綠绿 藍蓝 黃黄 驅驱 騎骑 驚惊 鬱郁 寶宝 幣币 獎奖 慶庆 願愿 劇剧 場场 環环 塵尘 滅灭 '
    '槍枪 彈弹 機机 艦舰 軍军 衛卫 護护 險险 嚴严 雙双 雜杂 難难 鍵键 鐵铁 鋼钢 輪轮 輕轻 壓压 縛缚 '
    '燼烬 輝辉 愛爱 戀恋 舊旧 頁页 題题 顏颜 漁渔 灘滩 濱滨 豐丰 麗丽 億亿 萬万 貳贰 參参').split()})


def norm_title(s):
    """活动名比较用：全角半角统一、去掉间隔号和空白，OCR 常见的异体字（煙 / 烟、疊 / 叠）统一。"""
    return unicodedata.normalize('NFKC', s or '').translate(_TITLE_VARIANTS).translate(_TITLE_T2S).strip()


def same_title(a, b):
    """是不是同一个活动。不能按相似度算：「密米尔的返演·上篇」和「·下篇」只差一个字，相似度 0.88。
    只容 OCR 丢一个字（「共诉隐匿」/「共诉隐匿时」）；认错字的（煙 / 烟）在 norm_title 里统一。"""
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = sorted((na, nb), key=len)
    return len(short) >= 4 and len(long_) - len(short) == 1 and short in long_


def chain_code(code, k):
    """从 code 开始用「继续前进」连着打，第 k 关之后是哪一关（k=0 是它自己）：最后一段数字往后数 k。

    'UA-1-8', 2 -> 'UA-1-10'；'7-9', 1 -> '7-10'；末尾不是数字推不出，返回 None。
    """
    if not code or k < 0:
        return None
    if k == 0:
        return code
    m = _CODE_TAIL_RE.match(code)
    if not m:
        return None
    return '%s%d' % (m.group(1), int(m.group(2)) + k)


def ledger_path():
    """记录文件的路径（相对工作目录，和 configs 里的设置放在一起）；环境变量关掉了返回 None。"""
    p = os.environ.get(LEDGER_ENV)
    if p is None:
        return LEDGER_DEFAULT
    if p.strip().lower() in ('', 'off', '0', 'none'):
        return None
    return p


def client_key(exe_path):
    """游戏客户端的记号：exe 的完整路径（小写、正斜杠）。两个服 / 两个号装在不同目录（10/13：Epic 国际服
    C:\\Program Files\\Epic Games\\GIRLSFRONTLINE2EXILIUM、国服「少女前线2：追放」D:\\Program Files\\GF2Exilium），
    打过的关分开记（I-54）。拿不到返回 ''。"""
    if not exe_path:
        return ''
    return os.path.normcase(os.path.abspath(str(exe_path))).replace('\\', '/')


LEDGER_VERSION = 2


def ledger_doc(data):
    """{客户端: {活动名: {编号: …}}}（'' = 共用）-> 写进文件的样子（第 2 版）：
      - shared：共用那份；clients：各客户端各一份；
      - stages：**所有**记录合在一起 —— 给 I-54 之前的旧版本看（它只认 stages）。旧版本开着、又写了一次文件时，
        丢掉的只是「哪条是哪个客户端的」，记录本身一条不少（下次新版本读到的是第 1 版，全算共用）。"""
    union = {}
    for books in data.values():
        for title, stages in books.items():
            key = next((k for k in union if same_title(k, title)), title)
            for code, info in stages.items():
                union.setdefault(key, {}).setdefault(code, info)
    return {'version': LEDGER_VERSION, 'shared': data.get('', {}), 'clients': {k: v for k, v in data.items() if k},
            'stages': union}


def read_ledger_doc(d):
    """文件内容 -> {客户端: {活动名: {编号: …}}}。第 1 版（只有 stages）/ 旧版本覆盖回去的：全算共用。"""
    def stages_of(v):
        return {k: dict(x) for k, x in v.items() if isinstance(x, dict)} if isinstance(v, dict) else {}
    if not isinstance(d, dict):
        return {}
    ver = d.get('version')
    v2 = isinstance(ver, int) and ver >= 2 and isinstance(d.get('shared'), dict)
    data = {'': stages_of(d.get('shared') if v2 else d.get('stages'))}
    clients = d.get('clients') if v2 else None
    for k, v in (clients.items() if isinstance(clients, dict) else ()):
        if k:
            data[k] = stages_of(v)
    return data


class StageLedger:
    """{客户端: {活动名: {编号: {at, how}}}}。活动名按 same_title 比（容 OCR 丢一个字、煙 / 烟）。

    客户端是游戏 exe 的路径（client_key）：两个客户端（两个服 / 两个号）的记录分开（I-54）。
    第 1 版文件（只有「stages」、不分客户端）和拿不到客户端时记的，放在「共用」那份：每个客户端都参考 ——
    对不上这个号时由选关那边点开面板核实（I-53）。文件的样子见 ledger_doc。

    每次读都看一眼文件的修改时间：你删了 / 改了文件，不用重启就生效。读写出错抛给调用方。
    """

    def __init__(self, path):
        self.path = path
        self._data = None             # {客户端: {活动名: {编号: …}}}；'' = 共用
        self._mtime = None
        self._file_version = None     # 读到的文件是第几版（第 1 版第一次升级时留个备份）

    def _load(self):
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            if self._mtime is not None or self._data is None:
                self._data = {}               # 没有文件 / 被你删了 = 什么都没记
            self._mtime = None
            return self._data
        if self._data is not None and mtime == self._mtime:
            return self._data
        with open(self.path, encoding='utf-8') as f:
            d = json.load(f)
        self._data = read_ledger_doc(d)
        self._file_version = d.get('version') if isinstance(d, dict) else None
        self._mtime = mtime
        return self._data

    def _key(self, stages, title):
        for key in stages:
            if same_title(key, title):
                return key
        return norm_title(title)

    def codes(self, title, client=''):
        """这个客户端在这个活动 / 这一章记下的编号（连同共用那份）。"""
        if not title:
            return set()
        out = set()
        data = self._load()
        for c in {client or '', ''}:
            for key, stages in data.get(c, {}).items():
                if same_title(key, title):
                    out.update(stages)
        return out

    def add(self, title, code, how, client=''):
        """记一关（记在这个客户端名下；拿不到客户端记共用）；已经记过返回 False（不重写文件）。
        how：'won' 自己打赢的 / 'chain' 「继续前进」连过去、编号推算的 / 'panel' 点开面板看到打过。"""
        if not title or not code:
            return False
        self._load()
        books = self._data.setdefault(client or '', {})
        stages = books.setdefault(self._key(books, title), {})
        if code in stages:                    # 只看自己这份：共用那份里有（多半是另一个客户端打的）也照记一笔
            return False
        stages[code] = dict(at=time.strftime('%Y-%m-%d %H:%M'), how=how)
        self._save()
        return True

    def _save(self):
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if self._file_version != LEDGER_VERSION and os.path.exists(self.path) and not os.path.exists(self.path + '.v1.bak'):
            shutil.copyfile(self.path, self.path + '.v1.bak')     # 第 1 版第一次升级：留一份原样的
        out = ledger_doc(self._data)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)            # 写到一半断电也不会留下半个文件
        self._mtime = os.path.getmtime(self.path)
        self._file_version = LEDGER_VERSION
