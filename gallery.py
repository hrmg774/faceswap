import threading
import gradio as gr
import os
import random as _random
from utils.open_by_coccoc import open_coccoc


os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

OUTPUT_DIR = "output"
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_results")
CROPPED_DIR = os.path.join(OUTPUT_DIR, "cropped_faces")
RESTORED_DIR = os.path.join(OUTPUT_DIR, "restored_faces")
GALLERY_DIR = r"C:\hrmg774\momhoa\x"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp",
              ".bmp", ".mp4", ".avi", ".mov", ".mkv")


# ─── Gallery ─────────────────────────────────────────────────────────────────
def load_gallery(folder_path):
    if not os.path.isdir(folder_path):
        return [], 0, None, ""
    files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(IMAGE_EXTS)
        ]
    )
    if not files:
        return [], 0, None, ""
    return files, 0, files[0], os.path.basename(files[0])


def gallery_navigate(files, idx, direction, shuffle_on):
    if not files:
        return idx, None, ""
    n = len(files)
    new_idx = _random.randint(
        0, n - 1) if shuffle_on else (idx + direction) % n
    return new_idx, files[new_idx], os.path.basename(files[new_idx])


def gallery_tick(files, idx, shuffle_on):
    return gallery_navigate(files, idx, 1, shuffle_on)


# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    with gr.Tabs():

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
                    img_label = gr.Textbox(
                        interactive=False, show_label=False, container=False
                    )
                with gr.Column(scale=3):
                    folder_drop = gr.Dropdown(
                        choices=[GALLERY_DIR, RESTORED_DIR,
                                 FINAL_DIR, CROPPED_DIR],
                        value=GALLERY_DIR,
                        label="Select Folder",
                        allow_custom_value=False,
                    )
                    load_btn = gr.Button("Load", variant="secondary")
                    with gr.Row():
                        prev_btn = gr.Button("⬅️ Prev", size="lg")
                        next_btn = gr.Button("Next ➡️", size="lg")
                    with gr.Row():
                        shuffle_chk = gr.Checkbox(
                            label="🔀 Shuffle", value=False)
                        autoplay_chk = gr.Checkbox(
                            label="▶️ Autoplay", value=False)
                    interval_sl = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=0.5,
                        label="Speed (sec/img)",
                    )

            load_btn.click(
                fn=load_gallery,
                inputs=[folder_drop],
                outputs=[gallery_files, gallery_idx, gallery_img, img_label],
            )
            prev_btn.click(
                fn=lambda f, i, s: gallery_navigate(f, i, -1, s),
                inputs=[gallery_files, gallery_idx, shuffle_chk],
                outputs=[gallery_idx, gallery_img, img_label],
            )
            next_btn.click(
                fn=lambda f, i, s: gallery_navigate(f, i, 1, s),
                inputs=[gallery_files, gallery_idx, shuffle_chk],
                outputs=[gallery_idx, gallery_img, img_label],
            )

            timer = gr.Timer(value=3, active=False)

            autoplay_chk.change(
                fn=lambda active, iv: gr.Timer(value=iv, active=active),
                inputs=[autoplay_chk, interval_sl],
                outputs=[timer],
            )
            interval_sl.change(
                fn=lambda active, iv: gr.Timer(value=iv, active=active),
                inputs=[autoplay_chk, interval_sl],
                outputs=[timer],
            )
            timer.tick(
                fn=gallery_tick,
                inputs=[gallery_files, gallery_idx, shuffle_chk],
                outputs=[gallery_idx, gallery_img, img_label],
            )


if __name__ == "__main__":

    threading.Thread(target=open_coccoc, args=(
        "gallery",), daemon=True).start()

    demo.launch(server_port=7860, inbrowser=False, allowed_paths=[GALLERY_DIR])
