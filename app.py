"""兼容入口：转发到 ui/app.py。"""

import os
import subprocess
import sys


def _run_streamlit_ui() -> int:
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", ui_path]
    print("[NetGuard] 检测到使用 python 直接启动，自动切换为 Streamlit 模式...")
    return subprocess.call(cmd)


if __name__ == "__main__":
    launcher_name = os.path.basename(sys.argv[0]).lower()
    if "streamlit" not in launcher_name:
        raise SystemExit(_run_streamlit_ui())

# Streamlit 执行该脚本时，直接导入 UI 模块即可。
from ui.app import *  # noqa: F401,F403
