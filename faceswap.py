import os
import cv2
import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
from utils.rename_file import rename_by_current_time

try:
    import nvidia.cudnn, nvidia.cuda_runtime
    os.add_dll_directory(os.path.join(nvidia.cudnn.__path__[0], "bin"))
    os.add_dll_directory(os.path.join(nvidia.cuda_runtime.__path__[0], "bin"))
except Exception:
    pass

source_path = r"input_face\input_face_20260913_174734_01.jpg"
target_path = r"input_data\input_data_20260913_175550_12.jpg"
model_path = r"models\inswapper_128.onnx"

codeformer_dir = "CodeFormer"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)
os.makedirs("temp", exist_ok=True)

for p in [source_path, target_path, model_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Không tìm thấy file: {os.path.abspath(p)}")

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

src_img = cv2.imread(source_path)
tgt_img = cv2.imread(target_path)

src_faces = app.get(src_img)
tgt_faces = app.get(tgt_img)

if len(src_faces) == 0:
    raise ValueError("Không tìm thấy mặt trong ảnh nguồn")
if len(tgt_faces) == 0:
    raise ValueError("Không tìm thấy mặt trong ảnh đích")

source_face = src_faces[0]
result_swap = tgt_img.copy()

for face in tgt_faces:
    result_swap = swapper.get(result_swap, face, source_face, paste_back=True)

temp_swap_path = os.path.join("temp", "temp_swap_result.png")
cv2.imwrite(temp_swap_path, result_swap)

input_for_codeformer = f"../{temp_swap_path}"
output_for_codeformer = f"../{output_dir}"

cmd = f'cd {codeformer_dir} && "{sys.executable}" inference_codeformer.py -w 0.5 -i "{input_for_codeformer}" -o "{output_for_codeformer}" --face_upsample'

subprocess.run(cmd, shell=True, check=True)

final_results_dir = os.path.join(output_dir, "final_results")
rename_by_current_time(final_results_dir)

try:
    os.remove(temp_swap_path)
except Exception:
    pass

