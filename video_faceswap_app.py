import threading
import gradio as gr
import os
import cv2
import subprocess
import sys
import uuid
import shutil
import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Use the existing utils
from utils.open_by_coccoc import open_coccoc
from utils.rename_file import rename_by_current_time

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

ort.preload_dlls()
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
if "CUDAExecutionProvider" not in ort.get_available_providers():
    print("WARNING: CUDAExecutionProvider is unavailable. Processing video on CPU will be very slow.")

MODEL_PATH = r"models\inswapper_128.onnx"
OUTPUT_DIR = "output"
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_results")
TEMP_DIR = "temp"
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")

def _latest_video(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(VIDEO_EXTS)
    ]
    return max(files, key=os.path.getmtime) if files else None

for d in (FINAL_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=PROVIDERS)
if "CUDAExecutionProvider" not in swapper.session.get_providers():
    print("WARNING: The inswapper ONNX session did not initialize CUDAExecutionProvider.")


def get_face(img_path):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        return None
    return faces[0]


def process_video(source_img_path, target_video_path, progress=gr.Progress()):
    if not source_img_path or not target_video_path:
        return None

    progress(0, desc="Analyzing source face...")
    source_face = get_face(source_img_path)
    if source_face is None:
        raise gr.Error("No face found in input face image.")

    session_id = str(uuid.uuid4())

    # 1. Extract Audio
    progress(0.05, desc="Extracting audio...")
    audio_path = os.path.join(TEMP_DIR, f"{session_id}.aac")

    # Try copying audio stream first
    subprocess.run(["ffmpeg", "-y", "-i", target_video_path, "-vn",
                   "-acodec", "copy", audio_path], stderr=subprocess.DEVNULL)
    has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    if not has_audio:
        # If copy fails, try encoding to aac
        subprocess.run(["ffmpeg", "-y", "-i", target_video_path, "-vn",
                       "-c:a", "aac", audio_path], stderr=subprocess.DEVNULL)
        has_audio = os.path.exists(
            audio_path) and os.path.getsize(audio_path) > 0

    # 2. Extract and Process Frames
    progress(0.1, desc="Processing video frames...")
    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_out_path = os.path.join(TEMP_DIR, f"{session_id}_no_audio.mp4")
    out = cv2.VideoWriter(temp_video_out_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tgt_faces = app.get(frame)
        res_frame = frame.copy()
        if tgt_faces:
            for face in tgt_faces:
                res_frame = swapper.get(
                    res_frame, face, source_face, paste_back=True)

        out.write(res_frame)
        frame_count += 1

        if total_frames > 0 and frame_count % 10 == 0:
            progress(0.1 + 0.8 * (frame_count / total_frames),
                     desc=f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()

    # 3. Combine Audio and Video
    progress(0.9, desc="Merging audio and video...")
    final_output_path = os.path.join(FINAL_DIR, f"swapped_{session_id}.mp4")

    if has_audio:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_video_out_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            final_output_path
        ], check=True, stderr=subprocess.DEVNULL)
    else:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_video_out_path,
            "-c:v", "libx264",
            final_output_path
        ], check=True, stderr=subprocess.DEVNULL)

    # Cleanup
    progress(1.0, desc="Cleaning up...")
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass
    if os.path.exists(temp_video_out_path):
        try:
            os.remove(temp_video_out_path)
        except Exception:
            pass

    rename_by_current_time(FINAL_DIR)
    
    return _latest_video(FINAL_DIR)


# --- UI ---
with gr.Blocks() as demo:
    with gr.Tabs():
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
        "video_faceswap",), daemon=True).start()

    demo.launch(server_port=7863, inbrowser=False)
