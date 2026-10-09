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