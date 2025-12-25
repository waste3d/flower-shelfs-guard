import cv2
import pandas as pd
from pathlib import Path
from config import IMAGE_ZONES

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/zones")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for filename, zones in IMAGE_ZONES.items():
    img_path = RAW_DIR / filename
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read {img_path}")
        continue

    h, w, _ = img.shape
    print(f"{filename}: {w}x{h}")

    for zone in zones:
        x1, y1, x2, y2 = zone.box
        crop = img[y1:y2, x1:x2]

        zone_filename = f"{Path(filename).stem}_{zone.name}.jpg"
        out_path = OUT_DIR / zone_filename
        cv2.imwrite(str(out_path), crop)

        label = "full" if "full" in filename else "empty"

        rows.append({
            'orig_image': filename,
            'zone_name': zone.name,
            'zone_image': zone_filename,
            'label': label,
        })

df = pd.DataFrame(rows)
df.to_csv("data/processed/zones.csv", index=False)
print(df)