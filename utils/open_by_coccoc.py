# utils/open_by_coccoc.py

import gradio as gr
import subprocess
import threading
import time


def open_coccoc():
    time.sleep(1.5)

    coccoc_path = r"C:\Program Files\CocCoc\Browser\Application\browser.exe"
    url = "http://127.0.0.1:7860"

    try:
        subprocess.Popen([coccoc_path, url])
    except Exception as e:
        print(f"Không thể mở Cốc Cốc. Hãy truy cập {url} thủ công. Lỗi: {e}")
