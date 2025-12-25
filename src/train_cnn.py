import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

CSV_PATH = Path("data/processed/zones.csv")
IMAGES_DIR = Path("data/processed/zones")
IMG_SIZE = 128

def load_data():
    df = pd.read_csv(CSV_PATH)
    X = []
    y = []

    for _, row in df.iterrows():
        img_path = IMAGES_DIR / row["zone_image"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0

        X.append(img)
        y.append(1 if row["label"] == "full" else 0)

    X = np.array(X)
    y = np.array(y, dtype="float32")
    return X, y

X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42
)

print(X_train.shape, X_val.shape)

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
]

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Val accuracy: {val_acc:.3f}")

MODEL_PATH = Path("models/cnn_full_empty.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
