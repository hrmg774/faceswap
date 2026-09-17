import threading

import gradio as gr
import os
import cv2
import subprocess
import sys
import insightface
from insightface.app import FaceAnalysis
from utils.open_by_coccoc import open_coccoc
from utils.rename_file import rename_by_current_time
from utils.delete_file import clear_all_files

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

try:
    import nvidia.cudnn
    import nvidia.cuda_runtime

    os.add_dll_directory(os.path.join(nvidia.cudnn.__path__[0], "bin"))
    os.add_dll_directory(os.path.join(nvidia.cuda_runtime.__path__[0], "bin"))
except Exception:
    pass

MODEL_PATH = r"models\inswapper_128.onnx"
CODEFORMER = "CodeFormer"
OUTPUT_DIR = "output"
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_results")
CROPPED_DIR = os.path.join(OUTPUT_DIR, "cropped_faces")
RESTORED_DIR = os.path.join(OUTPUT_DIR, "restored_faces")
TEMP_DIR = "temp"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

for d in (FINAL_DIR, TEMP_DIR, CROPPED_DIR, RESTORED_DIR):
    os.makedirs(d, exist_ok=True)

# Initialize models
app = FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(
    MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

MAX_FACES = 10


def _run_codeformer(img_path):
    cmd = (
        f'cd {CODEFORMER} && "{sys.executable}" inference_codeformer.py '
        f'-w 0.5 -i "../{img_path}" -o "../{OUTPUT_DIR}" --face_upsample'
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


def detect_faces(src_path):
    if not src_path:
        updates = []
        for i in range(MAX_FACES):
            updates.extend(
                [gr.update(visible=False), gr.update(value=None), gr.update(value=None)]
            )
        return updates

    img = cv2.imread(src_path)
    if img is None:
        updates = []
        for i in range(MAX_FACES):
            updates.extend(
                [gr.update(visible=False), gr.update(value=None), gr.update(value=None)]
            )
        return updates

    faces = app.get(img)
    # Sort faces from left to right
    faces = sorted(faces, key=lambda f: f.bbox[0])

    updates = []
    for i in range(MAX_FACES):
        if i < len(faces):
            bbox = faces[i].bbox.astype(int)
            # Expand bbox slightly for better visualization
            y1 = max(0, bbox[1] - 20)
            y2 = min(img.shape[0], bbox[3] + 20)
            x1 = max(0, bbox[0] - 20)
            x2 = min(img.shape[1], bbox[2] + 20)

            crop = img[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=crop))
            updates.append(gr.update(value=None))  # Reset replacement upload
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None))
            updates.append(gr.update(value=None))

    return updates


def execute_swap(src_path, *replace_paths):
    if not src_path:
        return None

    img = cv2.imread(src_path)
    if img is None:
        return None

    faces = app.get(img)
    faces = sorted(faces, key=lambda f: f.bbox[0])

    result_img = img.copy()
    swapped = False

    for i in range(min(len(faces), MAX_FACES)):
        repl_path = replace_paths[i]
        if repl_path is not None:
            repl_img = cv2.imread(repl_path)
            if repl_img is not None:
                repl_faces = app.get(repl_img)
                if repl_faces:
                    # Use the largest face in the replacement image
                    repl_face = sorted(
                        repl_faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        reverse=True,
                    )[0]
                    result_img = swapper.get(
                        result_img, faces[i], repl_face, paste_back=True
                    )
                    swapped = True

    if not swapped:
        return src_path  # Return original if nothing swapped

    temp_path = os.path.join(TEMP_DIR, "temp_multiswap.png")
    cv2.imwrite(temp_path, result_img)

    try:
        _run_codeformer(temp_path)
        _post_process()
    except Exception as e:
        print(f"Error during CodeFormer/PostProcess: {e}")
        # If codeformer fails, just return the swapped image directly
        return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return _latest_image(FINAL_DIR)


# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Face Swap App")
    gr.Markdown(
        "Thay thế từng khuôn mặt trong một bức ảnh chứa nhiều người một cách tùy chọn."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Bước 1: Tải ảnh gốc")
            src_image = gr.Image(
                type="filepath",
                label="Ảnh gốc (Nhiều khuôn mặt)",
                height=300,
                sources=["upload"],
            )
            detect_btn = gr.Button("Bước 2: Phát hiện khuôn mặt", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### Bước 4: Thực thi và Kết quả")
            result_image = gr.Image(
                type="filepath",
                label="Kết quả cuối cùng",
                interactive=False,
                height=300,
            )
            swap_btn = gr.Button("Bắt đầu Swap", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### Bước 3: Cấu hình Swap từng khuôn mặt")
    gr.Markdown(
        "Các khuôn mặt phát hiện được sẽ hiển thị bên dưới. Tải lên một bức ảnh mới vào vị trí tương ứng để thay thế. Nếu để trống, khuôn mặt đó sẽ được giữ nguyên."
    )

    face_rows = []
    crop_images = []
    replace_images = []

    # Pre-create UI components for a maximum number of faces
    for i in range(MAX_FACES):
        with gr.Row(visible=False) as row:
            with gr.Column(scale=1, min_width=150):
                gr.Markdown(f"**Khuôn mặt #{i+1}**")
            with gr.Column(scale=2):
                crop = gr.Image(
                    type="numpy",
                    label="Khuôn mặt đã cắt",
                    interactive=False,
                    height=150,
                )
            with gr.Column(scale=2):
                replace = gr.Image(
                    type="filepath",
                    label="Ảnh thay thế",
                    sources=["upload"],
                    height=150,
                )

            face_rows.append(row)
            crop_images.append(crop)
            replace_images.append(replace)

    # Bundle the updates
    detect_outputs = []
    for i in range(MAX_FACES):
        detect_outputs.extend([face_rows[i], crop_images[i], replace_images[i]])

    detect_btn.click(fn=detect_faces, inputs=[src_image], outputs=detect_outputs)

    swap_btn.click(
        fn=execute_swap, inputs=[src_image] + replace_images, outputs=[result_image]
    )

if __name__ == "__main__":

    threading.Thread(target=open_coccoc, daemon=True).start()

    demo.launch(inbrowser=False)
