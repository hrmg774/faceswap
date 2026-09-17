# FaceSwap Project

Dự án thực hiện hoán đổi khuôn mặt (Face Swap) sử dụng thư viện **InsightFace** (để trích xuất và ghép mặt) và **CodeFormer** (để phục hồi và tăng cường chất lượng khuôn mặt sau khi ghép).

## Yêu cầu hệ thống
- Python 3.8 - 3.10
- Khuyến nghị sử dụng GPU (NVIDIA) với CUDA để tăng tốc độ xử lý.

## Cài đặt

1. Tạo môi trường ảo và kích hoạt:
   ```bash
   python -m venv .venv
   ```
   ```bash
   .venv\Scripts\activate
   ```

2. Cài đặt các thư viện từ file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   `onnxruntime-gpu==1.30.0` được cài cùng các extra CUDA 13 và cuDNN 9
   cần thiết cho CUDA Execution Provider. Ứng dụng sẽ preload các DLL này
   từ các package NVIDIA trong môi trường `.venv`; không cần copy DLL vào
   `System32` hoặc cài thêm CUDA Toolkit chỉ để chạy FaceSwap.

3. Cài đặt mô hình và mã nguồn phụ:
   - Tải mô hình `inswapper_128.onnx` và đặt vào thư mục `models/`.

   https://huggingface.co/hrmg774/inswapper_128.onnx/blob/main/inswapper_128.onnx

   - **Cài đặt CodeFormer:**
     Clone dự án CodeFormer vào thư mục hiện tại và cài đặt các mô hình cần thiết:
     ```bash
     git clone https://github.com/hrmg774/CodeFormer.git
     cd CodeFormer
     pip install -r requirements.txt
     python basicsr/setup.py develop
     python scripts/download_pretrained_models.py facelib
     python scripts/download_pretrained_models.py CodeFormer
     cd ..
     ```
   - **Cài đặt Model cho CodeFormer:**
     https://huggingface.co/hrmg774/codeformer/blob/main/codeformer.pth

     Đặt file codeformer.pth vào thư mục `CodeFormer/experiments/codeformer/models/`

## Cấu trúc thư mục

- `input_face/`: Chứa ảnh gốc lấy khuôn mặt để ghép.
- `input_data/`: Chứa ảnh đích mà bạn muốn thay thế khuôn mặt.
- `output/`: Chứa kết quả hoán đổi khuôn mặt (bao gồm ảnh gốc và ảnh đã qua CodeFormer).
- `models/`: Chứa các mô hình AI phục vụ hoán đổi.
- `temp/`: Chứa file tạm trong quá trình xử lý.

## Hướng dẫn sử dụng

Chỉnh sửa các đường dẫn `source_path` và `target_path` trong file `faceswap.py` theo nhu cầu, sau đó chạy script:

```bash
python faceswap.py
```

### Chạy CodeFormer (Độc lập)

Nếu bạn chỉ muốn sử dụng CodeFormer để tăng cường chất lượng khuôn mặt cho một ảnh bất kỳ, bạn có thể chạy script của CodeFormer như sau:

```bash
cd CodeFormer
python inference_codeformer.py -w 0.5 -has_aligned --input_path ../output/ảnh_của_bạn.jpg --output_path ../output/kết_quả_codeformer/
cd ..
```
*(Lưu ý: Thay đổi `input_path` và `output_path` cho phù hợp với đường dẫn của bạn, tham số `-w` điều chỉnh mức độ phục hồi)*
