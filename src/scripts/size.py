import cv2
import os
from pathlib import Path

RAW_DIR = Path("data/raw")

print("Размеры фото витрин:")
print("-" * 30)

for filename in RAW_DIR.glob("*.jpg"):
    img = cv2.imread(str(filename))
    if img is not None:
        h, w = img.shape[:2]
        print(f"{filename.name}: {w}x{h}")
    else:
        print(f"{filename.name}: ERROR загрузки")
