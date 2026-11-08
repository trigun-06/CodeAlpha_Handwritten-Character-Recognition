import os
import sys
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emnist_alphanumeric_cnn.keras")
BACKUP_PATH = os.path.join(BASE_DIR, "emnist_alphanumeric_cnn_pretrained_backup.keras")
HANDWRITING_IMAGE_PATH = "/Users/trigun06/Desktop/an.jpeg"
DIGITS_CSV = os.path.join(BASE_DIR, "dataset/emnist-digits-train.csv")
LETTERS_CSV = os.path.join(BASE_DIR, "dataset/emnist-letters-train.csv")

IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 36

# Backup original model if not already backed up
if not os.path.exists(BACKUP_PATH) and os.path.exists(MODEL_PATH):
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    print(f"[INFO] Created backup of pretrained model at: {BACKUP_PATH}")


def preprocess_char_crop(crop_gray: np.ndarray) -> np.ndarray:
    """Standardizes a cropped character into a centered 28x28 normalized float array."""
    h, w = crop_gray.shape
    size = max(h, w) + 12
    square_img = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = crop_gray
    resized = cv2.resize(square_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def augment_character(img_28x28: np.ndarray, num_variations: int = 90) -> List[np.ndarray]:
    """Generates realistic variations: rotations, shear slants, shifts, scale, dilation, and erosion."""
    augmented = [img_28x28]
    h, w = img_28x28.shape

    for _ in range(num_variations):
        # 1. Random rotation [-24 to +24 degrees]
        angle = random.uniform(-24, 24)
        # 2. Random translation [-3 to +3 pixels]
        tx = random.uniform(-3, 3)
        ty = random.uniform(-3, 3)
        # 3. Random scale [0.82 to 1.18]
        scale = random.uniform(0.82, 1.18)
        # 4. Random shear / slant [-0.22 to +0.22]
        shear = random.uniform(-0.22, 0.22)

        cos_a = np.cos(np.radians(angle)) * scale
        sin_a = np.sin(np.radians(angle)) * scale
        M = np.array([
            [cos_a - shear * sin_a, -sin_a - shear * cos_a, tx + (1 - cos_a + shear * sin_a) * (w / 2)],
            [sin_a, cos_a, ty + (1 - cos_a) * (h / 2) - sin_a * (w / 2)]
        ], dtype=np.float32)

        warped = cv2.warpAffine(img_28x28, M, (w, h), borderValue=0)

        # 5. Pen stroke thickness variations (simulating both thin pen & thick 20px canvas brush)
        binary_uint8 = (warped * 255).astype(np.uint8)
        r = random.random()
        if r < 0.20:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary_uint8 = cv2.erode(binary_uint8, k, iterations=1)
        elif r < 0.65:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary_uint8 = cv2.dilate(binary_uint8, k, iterations=1)
        elif r > 0.85:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary_uint8 = cv2.dilate(binary_uint8, k, iterations=2)

        aug_float = binary_uint8.astype("float32") / 255.0
        augmented.append(aug_float)

    return augmented


def extract_user_dataset(image_path: str):
    """Segments all 62 characters from the user's uploaded handwriting sheet."""
    print(f"[INFO] Extracting characters from user sheet: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def extract_band(y1, y2):
        strip = thresh[y1:y2, :]
        cnts, _ = cv2.findContours(strip.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] >= 15 and cv2.boundingRect(c)[3] >= 25]

        merged = []
        for b in sorted(raw, key=lambda b: b[0]):
            if not merged:
                merged.append(b)
                continue
            px, py, pw, ph = merged[-1]
            cx, cy, cw, ch = b
            overlap = max(0, min(px + pw, cx + cw) - max(px, cx))
            if overlap > 0:
                merged[-1] = (min(px, cx), min(py, cy), max(px + pw, cx + cw) - min(px, cx), max(py + ph, cy + ch) - min(py, cy))
            else:
                merged.append(b)

        return strip, merged

    # Define the 5 rows
    bands_def = [
        ("A-W", 600, 900, list(range(10, 33))),       # 'A'=10 ... 'W'=32 (23 chars)
        ("X-Z", 900, 1200, list(range(33, 36))),      # 'X'=33 ... 'Z'=35 (3 chars)
        ("a-t", 1250, 1580, list(range(10, 30))),     # 'a'=10 ... 't'=29 (20 chars)
        ("u-z", 1600, 1900, list(range(30, 36))),     # 'u'=30 ... 'z'=35 (6 chars)
        ("0-9", 1950, 2300, list(range(0, 10))),      # '0'=0 ... '9'=9 (10 chars)
    ]

    X_user = []
    y_user = []

    for name, y1, y2, labels in bands_def:
        strip, boxes = extract_band(y1, y2)
        print(f"  -> Band '{name}': Extracted {len(boxes)} characters (expected {len(labels)})")

        for idx, (bx, by, bw, bh) in enumerate(boxes[:len(labels)]):
            crop = strip[by:by + bh, bx:bx + bw]
            proc = preprocess_char_crop(crop)
            label = labels[idx]

            # Augment each character
            aug_imgs = augment_character(proc, num_variations=90)
            for a_img in aug_imgs:
                X_user.append(a_img)
                y_user.append(label)

    # Also augment real canvas hand-drawn strokes if present
    canvas_file = os.path.join(BASE_DIR, "user_canvas.png")
    if os.path.exists(canvas_file):
        print(f"[INFO] Adding user canvas hand-drawn samples from: {canvas_file}")
        c_img = cv2.imread(canvas_file, cv2.IMREAD_GRAYSCALE)
        _, c_thresh = cv2.threshold(c_img, 128, 255, cv2.THRESH_BINARY_INV)
        # Slanted H: (10, 82, 99, 86) -> label 17 ('H')
        crop_H = c_thresh[82:82+86, 10:10+99]
        for a_img in augment_character(preprocess_char_crop(crop_H), num_variations=160):
            X_user.append(a_img)
            y_user.append(17)
        # i stem: (103, 80, 40, 45) -> label 18 ('I')
        crop_i_stem = c_thresh[80:80+45, 103:103+40]
        for a_img in augment_character(preprocess_char_crop(crop_i_stem), num_variations=160):
            X_user.append(a_img)
            y_user.append(18)
        # i full dot+stem: (87, 62, 56, 63) -> label 18 ('I')
        crop_i_full = c_thresh[62:62+63, 87:87+56]
        for a_img in augment_character(preprocess_char_crop(crop_i_full), num_variations=160):
            X_user.append(a_img)
            y_user.append(18)

    X_user = np.array(X_user)
    y_user = np.array(y_user)
    print(f"[INFO] Total Augmented User Samples: {len(X_user)} across {NUM_CLASSES} classes.")
    return X_user, y_user


def load_emnist_mix(num_samples=5000):
    """Loads a balanced mix of original EMNIST samples to maintain generalization."""
    print(f"[INFO] Loading {num_samples} background EMNIST samples...")
    d_df = pd.read_csv(DIGITS_CSV, header=None, nrows=num_samples // 2)
    l_df = pd.read_csv(LETTERS_CSV, header=None, nrows=num_samples // 2)

    y_d = d_df.iloc[:, 0].values
    X_d = d_df.iloc[:, 1:].values
    y_l = l_df.iloc[:, 0].values + 9
    X_l = l_df.iloc[:, 1:].values

    X_emnist = np.vstack((X_d, X_l)).astype("float32") / 255.0
    y_emnist = np.concatenate((y_d, y_l))

    X_emnist = X_emnist.reshape(-1, IMG_HEIGHT, IMG_WIDTH)
    X_emnist = np.transpose(X_emnist, (0, 2, 1))

    return X_emnist, y_emnist


def train_and_update_model():
    # 1. Extract and augment user handwriting
    X_user, y_user = extract_user_dataset(HANDWRITING_IMAGE_PATH)

    # 2. Load EMNIST general data mix
    X_emnist, y_emnist = load_emnist_mix(num_samples=5000)

    # 3. Combine datasets (heavily weighting the user's specific handwriting)
    X_combined = np.concatenate((X_user, X_emnist), axis=0)
    y_combined = np.concatenate((y_user, y_emnist), axis=0)

    # Add channel dimension (N, 28, 28, 1)
    X_combined = np.expand_dims(X_combined, axis=-1)
    y_cat = to_categorical(y_combined, NUM_CLASSES)

    # Shuffle
    X_train, y_train = shuffle(X_combined, y_cat, random_state=42)

    # 4. Load pretrained CNN model from BACKUP_PATH to ensure a clean base
    base_model_source = BACKUP_PATH if os.path.exists(BACKUP_PATH) else MODEL_PATH
    print(f"[INFO] Loading base model from: {base_model_source}")
    model = load_model(base_model_source)

    # Compile with low learning rate for gentle fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n[INFO] Starting Personalization Training on User Handwriting...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=7,
        batch_size=64,
        verbose=1
    )

    # 5. Save the fine-tuned model
    print(f"\n[INFO] Saving fine-tuned personalized model to: {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("[SUCCESS] Model fine-tuning complete!")


if __name__ == "__main__":
    train_and_update_model()
