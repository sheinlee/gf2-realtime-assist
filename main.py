# -*- coding: utf-8 -*-
"""GF2 实时辅助 独立版入口。

    双击桌面快捷方式，或：
    <ok-gf2>\\python\\pythonw.exe main.py

工作目录必须钉在本文件所在目录：configs/ logs/ screenshots/ assets/ 都按相对路径找。
ok-script 的单实例互斥名是 ``md5(os.getcwd())``，工作目录不同 ——
所以本程序和 ok-gf2 本体可以同时开着。

**不自带 ok-script，也不复制 ok-gf2 的源码**：启动时把已安装的 ok-gf2
``data/apps/ok-gf2/working`` 挂到 sys.path 上，直接用它的 ``ok`` 和 ``src``。
好处是版本永远和上游一致，坏处是 ok-gf2 必须装着。
"""
import os
import sys

# Windows 控制台默认 cp1252，中文日志会刷 UnicodeEncodeError（不致命但很吵）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8')
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tasks'))   # 任务之间是平铺 import


def _msgbox(text, title='GF2 实时辅助'):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
    except Exception:
        print(text)


def find_ok_gf2():
    """定位已安装的 ok-gf2 工作目录（里面有 ok/ 和 src/）。

    顺序和 启动.bat 找 Python 的顺序一致（两边得是同一份 ok-gf2）：
    环境变量 OK_GF2_DIR → 本文件夹放在 ok-gf2 的文件夹里（解压版，和 ok-gf2.exe 同一层）→ 安装版的默认位置。
    """
    override = os.environ.get('OK_GF2_DIR')
    local = os.environ.get('LOCALAPPDATA') or os.path.expanduser(r'~\AppData\Local')
    for cand in [override,
                 os.path.join(os.path.dirname(HERE), 'data', 'apps', 'ok-gf2', 'working'),
                 os.path.join(local, 'ok-gf2', 'data', 'apps', 'ok-gf2', 'working'),
                 os.path.join(local, 'ok-gf2', 'data', 'apps', 'ok-gf2', 'repo')]:
        if cand and os.path.isdir(os.path.join(cand, 'ok')) \
                and os.path.isdir(os.path.join(cand, 'src')):
            return cand
    return None


def ensure_admin():
    """没有管理员权限就以管理员身份重启自己。

    GF2 客户端是提权运行的。Windows 的 UIPI（用户界面特权隔离）不允许低完整性进程
    向高完整性窗口发消息，而 ok-gf2 的 Genshin 交互走的正是 PostMessage —— 截图和
    OCR 用的是另一套 API 不受影响，表现就是「什么都认得出来，就是点不动」，很有迷惑性。
    ok-gf2 本体能用，也是因为它自己就是提权启动的（pyappify.yml 里 uac/admin: true）。
    """
    import ctypes
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return
    except Exception:
        return
    params = ' '.join('"%s"' % a for a in sys.argv)
    rc = ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable, params, HERE, 1)
    if rc <= 32:          # <=32 表示失败，最常见是用户在 UAC 弹窗里点了「否」
        _msgbox('本程序需要管理员权限才能操作游戏窗口。\n\n'
                'GF2 客户端以管理员身份运行，Windows 不允许普通权限的程序向它发送\n'
                '鼠标/键盘消息。没有权限时：画面能识别，但所有点击都会被丢弃。\n\n'
                '请重新启动并在 UAC 提示中选择「是」。')
    sys.exit(0)


def prefer_trigger_tab():
    """开始运行后跳到「Triggers」页（实时辅助的开关和状态在那），不跳到「Tasks」页。

    ok-script 开始运行时，界面还停在第一页 Capture 的话，会切到 MainWindow.startup_task_tab() 给的页：
    有一次性任务就是「Tasks」页，没有才是「Triggers」页。本程序的一次性任务只有「测试:好友助战」，
    平时用的是实时辅助（触发任务），所以改成先给「Triggers」页。上游改了接口就照旧，不拦启动。
    :return: 改上了没有
    """
    try:
        from ok.ui.qt.MainWindow import MainWindow
    except Exception:
        return False
    orig = getattr(MainWindow, 'startup_task_tab', None)
    if orig is None or getattr(orig, '_prefer_trigger', False):
        return orig is not None

    def startup_task_tab(self):
        tab = getattr(self, 'trigger_tab', None)
        return tab if tab is not None else orig(self)
    startup_task_tab._prefer_trigger = True
    MainWindow.startup_task_tab = startup_task_tab
    return True


def main():
    app_dir = find_ok_gf2()
    if not app_dir:
        _msgbox('找不到已安装的 ok-gf2。\n\n'
                '本程序复用 ok-gf2 的 ok-script 运行时和 src/ 源码，ok-gf2 要先装好、正常打开过一次。\n'
                r'  · 安装包版：默认在 %LOCALAPPDATA%\ok-gf2，能直接找到' '\n'
                '  · 解压版：把本程序的文件夹放进 ok-gf2 的文件夹，和 ok-gf2.exe 同一层\n'
                r'  · 也可以设环境变量 OK_GF2_DIR，指向 ok-gf2 的 data\apps\ok-gf2\working 目录')
        sys.exit(1)
    # 追加在末尾：本目录的 config.py / tasks 优先，其余回落到 ok-gf2
    sys.path.append(app_dir)

    ensure_admin()

    from config import config
    from ok import OK
    prefer_trigger_tab()
    OK(config).start()


if __name__ == '__main__':
    main()
