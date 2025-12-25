from typing import Dict
import cv2
import numpy as np
from pathlib import Path

from config import IMAGE_ZONES
from infer_zone import extract_features
import joblib

MODEL_PATH = Path("models/baseline_full_empty.pkl")
RAW_DIR = Path("data/raw")

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_shelf_from_image(filename: str) -> Dict[str, str]:
    f"""
    возвращает словарь:  'zone_name': 'full/empty', 'status': 'ok/not ok'
    """

    img_path = RAW_DIR / filename
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")

    return predict_shelf_from_array(img, filename)

def predict_shelf_from_array(img: np.ndarray, filename: str) -> Dict[str, str]:
    """
    img: BGR-изображение (как читает cv2.imread или cv2.imdecode)
    """

    zones = IMAGE_ZONES[filename]
    if not zones:
        raise ValueError(f"No zones found for {filename}")

    model = get_model()

    result: Dict[str, str] = {}
    has_empty = False

    for zone in zones:
        x1, y1, x2, y2 = zone.box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            raise RuntimeError(f"Empty crop for zone {zone.name} in {filename}")

        X = extract_features(crop)
        pred = model.predict(X)[0]
        label = "full" if pred == 1 else "empty"
        result[zone.name] = label
        if label == "empty":
            has_empty = True

    result['status'] = 'ok' if not has_empty else 'not ok'
    return result

if __name__ == "__main__":
    print(predict_shelf_from_image("shelf_full_1.jpg"))
    print(predict_shelf_from_image("shelf_empty_1.jpg"))