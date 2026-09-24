import threading

import gradio as gr
import os
import cv2
import subprocess
import sys
import random as _random
import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis
from utils.open_by_coccoc import open_coccoc
from utils.rename_file import rename_by_current_time
from utils.delete_file import clear_all_files

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

ort.preload_dlls()
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError(
        "ONNX Runtime CUDAExecutionProvider is unavailable. "
        "Install the CUDA and cuDNN extras for onnxruntime-gpu."
    )

MODEL_PATH = r"models\inswapper_128.onnx"
CODEFORMER = "CodeFormer"
OUTPUT_DIR = "output"
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_results")
CROPPED_DIR = os.path.join(OUTPUT_DIR, "cropped_faces")
RESTORED_DIR = os.path.join(OUTPUT_DIR, "restored_faces")
TEMP_DIR = "temp"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

for d in (FINAL_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=PROVIDERS)
if "CUDAExecutionProvider" not in swapper.session.get_providers():
    raise RuntimeError(
        "The inswapper ONNX session did not initialize CUDAExecutionProvider."
    )


# ─── FaceSwap ────────────────────────────────────────────────────────────────
def _swap_faces(src_path, tgt_path):
    src_img = cv2.imread(src_path)
    tgt_img = cv2.imread(tgt_path)
    src_faces = app.get(src_img)
    tgt_faces = app.get(tgt_img)
    if not src_faces:
        raise ValueError("No face found in input_face")
    if not tgt_faces:
        raise ValueError("No face found in input_data")
    result = tgt_img.copy()
    for face in tgt_faces:
        result = swapper.get(result, face, src_faces[0], paste_back=True)
    return result


def _run_codeformer(img_path, weight=1.0):
    cmd = (
        f'cd {CODEFORMER} && "{sys.executable}" inference_codeformer.py '
        f'-w {weight} -i "../{img_path}" -o "../{OUTPUT_DIR}" --face_upsample'
    )
    subprocess.run(cmd, shell=True, check=True)


def _post_process():
    rename_by_current_time(FINAL_DIR)
    rename_by_current_time(RESTORED_DIR)
    clear_all_files(CROPPED_DIR)


def _latest_image(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    return max(files, key=os.path.getmtime) if files else None


def process_faceswap(source_img_path, target_img_path):
    if not source_img_path or not target_img_path:
        return None
    try:
        result = _swap_faces(source_img_path, target_img_path)

        temp_path = os.path.join(TEMP_DIR, "temp_swap.png")
        cv2.imwrite(temp_path, result)

        _run_codeformer(temp_path)
        _post_process()

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return _latest_image(FINAL_DIR)

    except Exception:
        return None


# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    with gr.Tabs():

        with gr.TabItem("FaceSwap"):
            with gr.Row():
                with gr.Column(scale=7):
                    output_image = gr.Image(
                        type="filepath",
                        label="Output",
                        interactive=False,
                        height=500,
                    )
                with gr.Column(scale=3):
                    input_face = gr.Image(
                        type="filepath",
                        label="Input Face",
                        height=250,
                        sources=["upload"],
                    )
                    input_data = gr.Image(
                        type="filepath",
                        label="Input Data",
                        height=250,
                        sources=["upload"],
                    )
                    run_btn = gr.Button(
                        "SwapFace", variant="primary", size="lg", elem_id="run-btn"
                    )

            run_btn.click(
                fn=process_faceswap,
                inputs=[input_face, input_data],
                outputs=[output_image],
            )


if __name__ == "__main__":

    threading.Thread(target=open_coccoc, args=(
        "faceswap",), daemon=True).start()

    demo.launch(server_port=7861, inbrowser=False)
