import gradio as gr
import os
import cv2
import subprocess
import sys
import random as _random
import insightface
from insightface.app import FaceAnalysis
from utils.rename_file import rename_by_current_time

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


try:
    import nvidia.cudnn
    import nvidia.cuda_runtime

    os.add_dll_directory(os.path.join(nvidia.cudnn.__path__[0], "bin"))
    os.add_dll_directory(os.path.join(nvidia.cuda_runtime.__path__[0], "bin"))
except Exception:
    pass

model_path = r"models\inswapper_128.onnx"
codeformer_dir = "CodeFormer"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Khởi tạo mô hình một lần khi chạy app để tăng tốc quá trình bấm nút
print("Đang nạp mô hình FaceAnalysis...")
try:
    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    print(f"Không thể khởi tạo FaceAnalysis: {e}")

print("Đang nạp mô hình inswapper...")
try:
    swapper = insightface.model_zoo.get_model(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
except Exception as e:
    print(f"Không thể khởi tạo Inswapper: {e}")


# ─── Hàm xử lý FaceSwap ──────────────────────────────────────────────────────
def process_faceswap(source_img_path, target_img_path):
    if not source_img_path or not target_img_path:
        return None, "⚠️ Vui lòng chọn cả ảnh nguồn và ảnh đích!"

    if not os.path.exists(model_path):
        return None, f"❌ Không tìm thấy mô hình: {model_path}"

    try:
        src_img = cv2.imread(source_img_path)
        tgt_img = cv2.imread(target_img_path)

        src_faces = app.get(src_img)
        tgt_faces = app.get(tgt_img)

        if len(src_faces) == 0:
            return None, "❌ Lỗi: Không tìm thấy khuôn mặt trong ảnh Nguồn"
        if len(tgt_faces) == 0:
            return None, "❌ Lỗi: Không tìm thấy khuôn mặt trong ảnh Đích"

        source_face = src_faces[0]
        result_swap = tgt_img.copy()

        for face in tgt_faces:
            result_swap = swapper.get(
                result_swap, face, source_face, paste_back=True)

        temp_swap_path = os.path.join("temp", "temp_swap_result.png")
        cv2.imwrite(temp_swap_path, result_swap)

        input_for_codeformer = f"../{temp_swap_path}"
        output_for_codeformer = f"../{output_dir}"

        cmd = f'cd {codeformer_dir} && "{sys.executable}" inference_codeformer.py -w 0.5 -i "{input_for_codeformer}" -o "{output_for_codeformer}" --face_upsample'
        subprocess.run(cmd, shell=True, check=True)

        final_results_dir = os.path.join(output_dir, "final_results")
        rename_by_current_time(final_results_dir)

        files = [
            os.path.join(final_results_dir, f)
            for f in os.listdir(final_results_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            return None, "❌ Lỗi: CodeFormer không sinh ra ảnh kết quả"

        latest_file = max(files, key=os.path.getmtime)

        try:
            os.remove(temp_swap_path)
        except Exception:
            pass

        return latest_file, "✅ Thành công! Ảnh đã được lưu trong thư mục output."

    except Exception as e:
        return None, f"❌ Lỗi trong quá trình xử lý: {str(e)}"


# ─── Hàm xử lý Bộ sưu tập ────────────────────────────────────────────────────
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_gallery_images(folder_path: str):
    """Quét thư mục và trả về danh sách đường dẫn ảnh."""
    if not folder_path or not os.path.isdir(folder_path):
        return [], "⚠️ Đường dẫn không hợp lệ hoặc không tồn tại!", None
    files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(IMAGE_EXTS)
        ]
    )
    if not files:
        return [], "⚠️ Không tìm thấy ảnh nào trong thư mục!", None
    return files, f"✅ Tìm thấy {len(files)} ảnh.", files[0]


def gallery_navigate(files, current_idx, direction, shuffle_on):
    """Điều hướng ảnh: tiếp theo / trước đó / ngẫu nhiên."""
    if not files:
        return current_idx, None
    n = len(files)
    if shuffle_on:
        new_idx = _random.randint(0, n - 1)
    else:
        new_idx = (current_idx + direction) % n
    return new_idx, files[new_idx]


def gallery_autoplay_next(files, current_idx, shuffle_on):
    """Hàm được Timer gọi để tự động chuyển ảnh."""
    return gallery_navigate(files, current_idx, 1, shuffle_on)


# ─── Xây dựng giao diện Web (Gradio) ─────────────────────────────────────────

with gr.Blocks() as demo:

    with gr.Tabs():

        with gr.TabItem("FaceSwap"):
            with gr.Row():
                with gr.Column(scale=7):
                    output_image = gr.Image(
                        type="filepath",
                        label="Out Put",
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
                        "SwapFace",
                        variant="primary",
                        size="lg",
                        elem_id="run-btn",
                    )

            run_btn.click(
                fn=process_faceswap,
                inputs=[input_face, input_data],
                outputs=[output_image],
            )

        with gr.TabItem("Gallery"):

            gallery_files = gr.State([])
            gallery_idx = gr.State(0)

            with gr.Row():
                with gr.Column(scale=7):
                    gallery_img = gr.Image(
                        type="filepath",
                        label="",
                        interactive=False,
                        height=500,
                    )
                with gr.Column(scale=3):
                    folder_input = gr.Textbox(
                        label="Đường dẫn thư mục ảnh",
                        placeholder="VD: C:/Users/h1oo7/Desktop/hrmg774/faceswap/output/final_results",
                        scale=1,
                    )
                    load_btn = gr.Button(
                        "Tải thư mục", variant="secondary", scale=1
                    )
                    with gr.Row():
                        prev_btn = gr.Button("⬅️ Trước", size="lg", scale=1)
                        next_btn = gr.Button("Tiếp ➡️", size="lg", scale=1)
                    with gr.Row():
                        shuffle_chk = gr.Checkbox(
                            label="🔀 Ngẫu nhiên", value=False, scale=1
                        )
                        autoplay_chk = gr.Checkbox(
                            label="▶️ Tự động phát", value=False, scale=1
                        )
                    interval_sl = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=0.5,
                        label="Tốc độ (giây / ảnh)",
                        scale=2,
                    )

            # Tải ảnh từ thư mục
            load_btn.click(
                fn=load_gallery_images,
                inputs=[folder_input],
                outputs=[gallery_files, gallery_img],
            ).then(fn=lambda: 0, outputs=[gallery_idx])

            # Nút điều hướng thủ công
            prev_btn.click(
                fn=lambda files, idx, shuf: gallery_navigate(
                    files, idx, -1, shuf),
                inputs=[gallery_files, gallery_idx, shuffle_chk],
                outputs=[gallery_idx, gallery_img],
            )
            next_btn.click(
                fn=lambda files, idx, shuf: gallery_navigate(
                    files, idx, 1, shuf),
                inputs=[gallery_files, gallery_idx, shuffle_chk],
                outputs=[gallery_idx, gallery_img],
            )

            # Autoplay bằng gr.Timer
            timer = gr.Timer(value=3, active=False)

            # Bật/tắt autoplay hoặc đổi tốc độ → cập nhật timer
            autoplay_chk.change(
                fn=lambda active, interval: gr.Timer(
                    value=interval, active=active),
                inputs=[autoplay_chk, interval_sl],
                outputs=[timer],
            )
            interval_sl.change(
                fn=lambda active, interval: gr.Timer(
                    value=interval, active=active),
                inputs=[autoplay_chk, interval_sl],
                outputs=[timer],
            )

            # Mỗi lần timer tick → chuyển ảnh kế tiếp (hoặc ngẫu nhiên)
            timer.tick(
                fn=gallery_autoplay_next,
                inputs=[gallery_files, gallery_idx, shuffle_chk],
                outputs=[gallery_idx, gallery_img],
            )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
    )
