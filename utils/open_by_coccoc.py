# utils/open_by_coccoc.py

import gradio as gr
import subprocess
import threading
import time


def open_coccoc(url):
    time.sleep(1.5)

    coccoc_path = r"C:\Program Files\CocCoc\Browser\Application\browser.exe"
    url1 = "http://127.0.0.1:7860"
    url2 = "http://127.0.0.1:7861"

    if url == "faceswap":
        subprocess.Popen([coccoc_path, url1])

    elif url == "multi_faceswap":
        subprocess.Popen([coccoc_path, url2])

    else:
        print("Không thể mở Cốc Cốc. Hãy truy cập {url} thủ công.")
