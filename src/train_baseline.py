import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

DATA_CSV = Path("data/processed/zones.csv")
IMAGES_DIR = Path("data/processed/zones")
MODEL_PATH = Path("models/baseline_full_empty.pkl")

df = pd.read_csv(DATA_CSV)

X = []
y = []

for _, row in df.iterrows():
    img_path = IMAGES_DIR / row['zone_image']
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    features = [
        np.mean(s),
        np.std(s),
        np.mean(v),
        np.std(v)
    ]
    X.append(features)
    y.append(1 if row['label'] == 'full' else 0)

X = np.array(X)
y = np.array(y)

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

print(f"Train accuracy: {acc:.2f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")