from typing import Dict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import IMAGE_ZONES

MODEL_PATH = Path("models/cnn_full_empty.keras")
RAW_DIR = Path("data/raw")

IMG_SIZE = 128
_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=0)  # (1, H, W, 3)
    return crop


def predict_shelf_from_image(filename: str) -> Dict[str, str]:
    """
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

        X = preprocess_crop(crop)
        p = model.predict(X, verbose=0)[0][0]
        label = "full" if p >= 0.5 else "empty"

        result[zone.name] = label
        if label == "empty":
            has_empty = True

    result["status"] = "ok" if not has_empty else "not ok"
    return result


if __name__ == "__main__":
    print(predict_shelf_from_image("shelf_full_1.jpg"))
    print(predict_shelf_from_image("shelf_empty_1.jpg"))
