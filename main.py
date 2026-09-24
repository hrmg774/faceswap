import threading
import gradio as gr
import os
from faceswap_app import process_faceswap
from multi_faceswap_app import detect_faces, execute_swap, MAX_FACES
from video_faceswap_app import process_video
from utils.open_by_coccoc import open_coccoc

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

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

        with gr.TabItem("Multi FaceSwap"):
            with gr.Row():
                with gr.Column(scale=7):
                    result_image = gr.Image(
                        type="filepath",
                        label="Output",
                        interactive=False,
                        height=500,
                    )
                with gr.Column(scale=3):
                    src_image = gr.Image(
                        type="filepath",
                        label="Input Data",
                        height=250,
                        sources=["upload"],
                    )
                    detect_btn = gr.Button("Detect Face", variant="secondary")
                    swap_btn = gr.Button(
                        "SwapFace", variant="primary", size="lg")

            gr.Markdown("---")

            face_rows = []
            crop_images = []
            replace_images = []

            # Pre-create UI components for a maximum number of faces
            for i in range(MAX_FACES):
                with gr.Row(visible=False) as row:
                    with gr.Column(scale=1, min_width=150):
                        gr.Markdown(f"**Face #{i+1}**")
                    with gr.Column(scale=3):
                        crop = gr.Image(
                            type="numpy",
                            label="Input Data",
                            interactive=False,
                            height=150,
                        )
                    with gr.Column(scale=3):
                        replace = gr.Image(
                            type="filepath",
                            label="Input Face",
                            sources=["upload"],
                            height=150,
                        )

                    face_rows.append(row)
                    crop_images.append(crop)
                    replace_images.append(replace)

            # Bundle the updates
            detect_outputs = []
            for i in range(MAX_FACES):
                detect_outputs.extend(
                    [face_rows[i], crop_images[i], replace_images[i]])

            detect_btn.click(fn=detect_faces, inputs=[
                             src_image], outputs=detect_outputs)

            swap_btn.click(
                fn=execute_swap, inputs=[src_image] + replace_images, outputs=[result_image]
            )

        with gr.TabItem("Video FaceSwap"):
            with gr.Row():
                with gr.Column(scale=7):
                    output_video = gr.Video(
                        label="Output Video", interactive=False, height=500)
                with gr.Column(scale=3):
                    input_face = gr.Image(
                        type="filepath",
                        label="Input Face",
                        sources=["upload"],
                        height=250,
                    )
                    input_video = gr.Video(
                        label="Input Data",
                        sources=["upload"],
                        height=250,
                    )
                    run_btn = gr.Button("SwapFace",
                                        variant="primary", size="lg")

            run_btn.click(
                fn=process_video,
                inputs=[input_face, input_video],
                outputs=[output_video]
            )

if __name__ == "__main__":

    threading.Thread(target=open_coccoc, args=(
        "all",), daemon=True).start()

    demo.launch(server_port=7860, inbrowser=False)
