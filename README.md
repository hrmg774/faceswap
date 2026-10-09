# FaceSwap + CodeFormer (Gradio)

Ứng dụng đổi mặt chạy hoàn toàn trên máy cá nhân:

1. **Đổi mặt** bằng InsightFace (`inswapper_128.onnx`) chạy qua ONNX Runtime GPU.
2. **Làm nét mặt** bằng CodeFormer (PyTorch).
3. Giao diện web bằng Gradio tại `http://127.0.0.1:7861`.

> Đã chạy thành công trên: Windows, Python 3.11.9, NVIDIA RTX 3050 (4 GB VRAM), driver hỗ trợ CUDA 13.x.

---

## 1. Yêu cầu

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Windows (code dùng đường dẫn `models\...`) |
| Python | **3.11** (khuyến nghị 3.11.x) |
| GPU | NVIDIA, driver mới (chạy `nvidia-smi`, dòng `CUDA Version` nên là 13.x) |
| Dung lượng trống | khoảng 10 GB (thư viện ~5 GB, model ~1 GB) |
| Git | để clone repo |
| Microsoft C++ Build Tools | chỉ cần nếu `insightface` báo lỗi biên dịch khi cài |

Không cần cài CUDA Toolkit riêng: thư viện CUDA/cuDNN được cài kèm qua pip (`onnxruntime-gpu[cuda,cudnn]`).

---

## 2. Cài đặt nhanh (khuyến nghị)

Mở **PowerShell** và chạy:

```powershell
git clone https://github.com/hrmg774/faceswap.git
cd faceswap
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Sau đó làm tiếp **mục 4 (tải model)** rồi **mục 5 (chạy app)**.

### Nội dung `setup.ps1`

Nếu repo chưa có file này, tạo `setup.ps1` với nội dung sau:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip uv

uv pip install -r requirements.txt --index-strategy unsafe-best-match

# Gỡ bản onnxruntime CPU (insightface kéo về) để không đè bản GPU
uv pip uninstall onnxruntime onnxruntime-gpu
uv pip install "onnxruntime-gpu[cuda,cudnn]==1.30.0"

# Tạo basicsr/version.py nếu chưa có
$v = "CodeFormer\basicsr\version.py"
if (-not (Test-Path $v)) {
@"
__version__ = '1.3.2'
__gitsha__ = 'unknown'
version_info = (1, 3, 2)
"@ | Set-Content -Encoding utf8 $v
}

Write-Host "Xong. Chay app bang: python faceswap_app.py"
```

---

## 3. Cài đặt thủ công (từng bước)

### Bước 1. Tạo môi trường ảo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip uv
```

Nếu PowerShell chặn script, chạy trước: `Set-ExecutionPolicy -Scope Process Bypass`.

### Bước 2. Cài thư viện

```powershell
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

- Cờ `--index-strategy unsafe-best-match` là **bắt buộc** với `uv`, vì `torch` lấy từ kho PyTorch còn các gói khác lấy từ PyPI.
- Lần đầu tải khoảng 4-5 GB (riêng `torch` ~2,5 GB), có thể mất 10-30 phút tùy mạng. `uv` có cache nên chạy lại sẽ nhanh.
- Có thể dùng `pip install -r requirements.txt` thay cho `uv`, nhưng chậm hơn.

### Bước 3. Sửa lại `onnxruntime` (quan trọng)

`insightface` kéo theo bản **CPU** của `onnxruntime`, bản này đè lên bản GPU và làm mất `CUDAExecutionProvider`. Gỡ cả hai rồi chỉ cài bản GPU:

```powershell
uv pip uninstall onnxruntime onnxruntime-gpu
uv pip install "onnxruntime-gpu[cuda,cudnn]==1.30.0"
```

Kiểm tra:

```powershell
python -c "import onnxruntime as ort; ort.preload_dlls(); print(ort.get_available_providers())"
```

Kết quả phải có `CUDAExecutionProvider`.

> Không chạy `uv pip uninstall` với các gói `nvidia-*` sau bước này, vì sẽ làm mất file `cudnn64_9.dll`.

### Bước 4. Tạo file `basicsr/version.py`

Thư mục `CodeFormer/basicsr` cần file `version.py` (thường không có trong git). Tạo file `CodeFormer\basicsr\version.py` với nội dung:

```python
__version__ = '1.3.2'
__gitsha__ = 'unknown'
version_info = (1, 3, 2)
```

---

## 4. Tải model

Các file model **không nằm trong git** vì quá nặng.

### Model bạn phải tự đặt vào

| File | Đặt tại |
|---|---|
| `inswapper_128.onnx` | `models\inswapper_128.onnx` |
| `codeformer.pth` | `CodeFormer\weights\CodeFormer\codeformer.pth` |

Với `codeformer.pth`, có thể tải bằng script của CodeFormer:

```powershell
cd CodeFormer
python scripts/download_pretrained_models.py CodeFormer
cd ..
```

Mẹo: nếu đã có sẵn các file này từ máy cũ, chỉ cần copy sang đúng thư mục.

### Model tự tải ở lần chạy đầu

| Model | Dung lượng | Lưu tại |
|---|---|---|
| `buffalo_l` (InsightFace) | ~300 MB | `C:\Users\<tên>\.insightface` |
| `RealESRGAN_x2plus.pth` | 64 MB | `CodeFormer\weights\realesrgan\` |
| `detection_Resnet50_Final.pth` | 104 MB | `CodeFormer\weights\facelib\` |
| `parsing_parsenet.pth` | 81 MB | `CodeFormer\weights\facelib\` |

Cần có mạng ở lần chạy đầu. Từ lần sau không phải tải lại, **đừng xóa thư mục `weights`**.

---

## 5. Chạy ứng dụng

```powershell
.\.venv\Scripts\Activate.ps1
python faceswap_app.py
```

Mở trình duyệt tại `http://127.0.0.1:7861`:

1. Tải ảnh lên **Input Face** (khuôn mặt muốn ghép vào).
2. Tải ảnh lên **Input Data** (ảnh đích).
3. Bấm **SwapFace**, đợi kết quả hiện ở ô Output.

Ảnh kết quả cũng được lưu trong `output\final_results`.