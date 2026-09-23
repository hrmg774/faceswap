
from faceswap_app import process_faceswap


for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    process_faceswap("momhoa.jpg", "dis7uyp-fe006fd8-6a9c-4a79-b70f-332f593161e3.png", weight=w)