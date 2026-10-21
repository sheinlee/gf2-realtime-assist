# -*- coding: utf-8 -*-
"""GF2 实时辅助 独立版的 ok-script 配置。

与 ok-gf2 自己的 src/config.py 的关键差异：

  · **没有 template_matching**。ok-gf2 靠 assets/coco_annotations.json + assets/images/
    做模板匹配，那批图是游戏截图；而本程序只跑纯 OCR 的任务（实时辅助、测试助战），
    一个模板特征都用不到，整块省掉，也就不用随包带游戏截图。
    代价：依赖模板的任务（推图/一键日常/周常）在独立版里跑不了 —— 那些请用 ok-gf2 本体。

  · 注册实时辅助、测试助战和机要密报批量解锁任务。

  · config_folder / logs / screenshots 都在本目录下，与 ok-gf2 完全隔离。
    ok-script 的单实例互斥名是 md5(cwd)，所以两边可以同时开着。

窗口/OCR/分辨率这几项是照抄 ok-gf2 的，改了会导致抓不到窗口或点不准。
"""
import os
import re

version = '0.1.0'

config = {
    'debug': False,
    'use_gui': True,
    'config_folder': 'configs',
    'gui_title': 'GF2 实时辅助',
    # 必填：ok-script 会把它直接喂给 qfluentwidgets 的 IconWidget，给 None 会崩
    'gui_icon': 'icon.png',
    'version': version,

    'wait_until_before_delay': 0,
    'wait_until_check_delay': 0,
    'wait_until_settle_time': 0,

    'ocr': {'lib': 'onnxocr', 'params': {'use_openvino': True}},

    # 不配 template_matching：本程序不做模板匹配

    'windows': {
        'exe': ['GF2_Exilium.exe'],
        'hwnd_class': 'UnityWndClass',
        'interaction': 'Genshin',
        'capture_method': ['WGC', 'BitBlt_RenderFull'],
        'check_hdr': True,
        'force_no_hdr': False,
        'require_bg': True,
    },
    'start_timeout': 60,
    'window_size': {'width': 1000, 'height': 720, 'min_width': 600, 'min_height': 450},
    'supported_resolution': {'min_size': (1280, 720)},

    'log_file': os.path.join('logs', 'gf2assist.log'),
    'error_log_file': os.path.join('logs', 'gf2assist_error.log'),
    'screenshots_folder': 'screenshots',

    'onetime_tasks': [
        ['tasks.IntelArchiveTask', 'IntelArchiveTask'],
        ['tasks.TestSupportPickTask', 'TestSupportPickTask'],
    ],
    'trigger_tasks': [
        ['tasks.RealtimeAssistTask', 'RealtimeAssistTask'],
    ],

    'about': (
        '<h3>GF2 实时辅助</h3>'
        '<p>关卡你点，剩下的它打：进本 / 跳剧情 / 借好友助战 / 部署 / 开打 / 结算。</p>'
        '<p>在「实时触发」页打开「实时辅助:点关卡自动打」即可。</p>'
        '<p>基于 <a href="https://github.com/alicejump/ok-gf2">ok-gf2</a> 与 '
        '<a href="https://github.com/ok-oldking/ok-script">ok-script</a>，复用其运行时，不含游戏素材。</p>'
    ),
}

# OCR 错字兼容：让 match 文本容忍形近/繁简错字。错字表用 ok-gf2 自己的
# assets/ocr_fix/ocr_text_fix.json（补丁先找本目录，没有再找 ok-gf2 的 working 目录），跟着上游更新。
try:
    from src.patches.ocr_text_fix_patch import install_ocr_text_fix_patch
    install_ocr_text_fix_patch()
except Exception as _e:  # 表缺失或上游改了接口，不该拦住启动
    print('OCR 错字兼容补丁未启用: %r' % (_e,))

_ = re  # config 里保留 re 以便照抄上游时扩 top_hwnd_class
