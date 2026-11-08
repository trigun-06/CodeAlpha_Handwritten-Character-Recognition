import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
DIGITS_TRAIN_CSV = "dataset/emnist-digits-train.csv"
DIGITS_TEST_CSV = "dataset/emnist-digits-test.csv"
LETTERS_TRAIN_CSV = "dataset/emnist-letters-train.csv"
LETTERS_TEST_CSV = "dataset/emnist-letters-test.csv"

MODEL_SAVE_PATH = "emnist_alphanumeric_cnn.keras"

IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 36  # 10 Digits (0-9) + 26 Letters (A-Z)

# Map class indices: 0-9 -> '0'-'9', 10-35 -> 'A'-'Z'
LABEL_MAP = {i: str(i) for i in range(10)}
LABEL_MAP.update({i + 10: chr(65 + i) for i in range(26)})


# ==========================================
# 1. LOAD & MERGE DIGITS + LETTERS DATASETS
# ==========================================
def load_and_merge_csv_data(
    digits_train_path, digits_test_path, letters_train_path, letters_test_path, max_digits_per_class=3500
):
    """
    Loads both EMNIST Digits and EMNIST Letters from external CSV files,
    shifts letter labels to 10-35, fixes image orientation, and merges them.
    """
    required_files = [digits_train_path, digits_test_path, letters_train_path, letters_test_path]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: '{path}'. Please ensure both digits and letters train/test CSVs are in 'dataset/'."
            )

    print("[INFO] Loading Digits CSV files...")
    d_train_df = pd.read_csv(digits_train_path, header=None)
    d_test_df = pd.read_csv(digits_test_path, header=None)

    # Pandas 3.x safe sampling: preserve all 785 columns (1 label + 784 pixels)
    if max_digits_per_class:
        sampled_indices = (
            d_train_df.groupby(0)
            .apply(lambda x: x.sample(min(len(x), max_digits_per_class), random_state=42), include_groups=False)
            .index.get_level_values(1)
        )
        d_train_df = d_train_df.loc[sampled_indices].reset_index(drop=True)

    print("[INFO] Loading Letters CSV files...")
    l_train_df = pd.read_csv(letters_train_path, header=None)
    l_test_df = pd.read_csv(letters_test_path, header=None)

    # Digits labels are already 0-9
    y_d_train = d_train_df.iloc[:, 0].values
    X_d_train = d_train_df.iloc[:, 1:].values
    y_d_test = d_test_df.iloc[:, 0].values
    X_d_test = d_test_df.iloc[:, 1:].values

    # Letters labels are 1-26 in EMNIST -> shift by +9 so 'A'=10, 'B'=11, ..., 'Z'=35
    y_l_train = l_train_df.iloc[:, 0].values + 9
    X_l_train = l_train_df.iloc[:, 1:].values
    y_l_test = l_test_df.iloc[:, 0].values + 9
    X_l_test = l_test_df.iloc[:, 1:].values

    # Concatenate Digits and Letters (both now guaranteed to have 784 pixel columns)
    X_train = np.vstack((X_d_train, X_l_train))
    y_train = np.concatenate((y_d_train, y_l_train))

    X_test = np.vstack((X_d_test, X_l_test))
    y_test = np.concatenate((y_d_test, y_l_test))

    # Shuffle combined datasets
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape flattened arrays to (N, 28, 28)
    X_train = X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH)
    X_test = X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH)

    # Fix EMNIST transposition (rotate 90 degrees & flip to make upright)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    # Add channel dimension for CNN: (N, 28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # One-hot encode labels across all 36 classes
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    print(f"[INFO] Combined Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    print(f"[INFO] Total Classes: {NUM_CLASSES} (0-9 and A-Z)")
    return X_train, y_train_cat, X_test, y_test_cat, y_test


# ==========================================
# 2. BUILD 36-CLASS CNN MODEL
# ==========================================
def build_cnn_model():
    """Builds a Deep CNN architecture for 36-class Alphanumeric Recognition."""
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

        # Block 1
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==========================================
# 3. EXTERNAL IMAGE PROCESSING & INFERENCE
# ==========================================
def preprocess_single_char_image(img_gray):
    """Pads an extracted character crop into a centered 28x28 square image."""
    h, w = img_gray.shape
    size = max(h, w) + 10
    square_img = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img_gray

    resized = cv2.resize(square_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))


def predict_single_character(model, image_path):
    """Predicts a single handwritten digit (0-9) or letter (A-Z) from an external image."""
    if not os.path.exists(image_path):
        print(f"[WARNING] Image '{image_path}' not found.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    processed_input = preprocess_single_char_image(thresh)
    predictions = model.predict(processed_input, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100

    predicted_char = LABEL_MAP[class_idx]
    print(f"[RESULT] Predicted Character/Digit: '{predicted_char}' (Confidence: {confidence:.2f}%)")
    return predicted_char


def predict_word_or_number_sequence(model, image_path):
    """
    EXTENDABLE FEATURE: Full Alphanumeric Sequence / Word Recognition.
    Segments an image containing letters and numbers left-to-right and predicts the sequence.
    """
    if not os.path.exists(image_path):
        print(f"[WARNING] Image '{image_path}' not found.")
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 8 and h > 15:
            char_boxes.append((x, y, w, h))

    # Sort bounding boxes left-to-right
    char_boxes = sorted(char_boxes, key=lambda b: b[0])

    recognized_sequence = []
    for (x, y, w, h) in char_boxes:
        char_crop = thresh[y:y + h, x:x + w]
        processed_char = preprocess_single_char_image(char_crop)

        preds = model.predict(processed_char, verbose=0)
        char_label = LABEL_MAP[np.argmax(preds[0])]
        recognized_sequence.append(char_label)

    final_output = "".join(recognized_sequence)
    print(f"[SEQUENCE RESULT] Recognized Alphanumeric String from '{image_path}': {final_output}")
    return final_output


# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Step 1: Load and Merge Digits + Letters CSVs
    X_train_full, y_train_full, X_test, y_test_cat, y_test_raw = load_and_merge_csv_data(
        DIGITS_TRAIN_CSV, DIGITS_TEST_CSV, LETTERS_TRAIN_CSV, LETTERS_TEST_CSV
    )

    # Split into 90% Train, 10% Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # Step 2: Build CNN Model
    model = build_cnn_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    # Step 3: Train Model
    print("\n[INFO] Starting 36-Class Alphanumeric CNN training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks
    )

    # Step 4: Evaluate on Combined Test Set
    print("\n[INFO] Evaluating on combined Digits + Letters test dataset...")
    y_pred_probs = model.predict(X_test, batch_size=128)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_test_raw, y_pred_classes)
    print(f"\n[FINAL METRIC] Alphanumeric Test Accuracy (0-9, A-Z): {test_acc * 100:.2f}%\n")

    target_names = [LABEL_MAP[i] for i in range(NUM_CLASSES)]
    print("Classification Report (0-9 & A-Z):")
    print(classification_report(y_test_raw, y_pred_classes, target_names=target_names))

    # Step 5: Save Training Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Alphanumeric CNN Accuracy (36 Classes)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Alphanumeric CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("alphanumeric_training_metrics.png")
    print("[INFO] Saved training curves to 'alphanumeric_training_metrics.png'")

    # Step 6: Optional Inference on External Images
    # predict_single_character(model, "sample_char_or_digit.png")
    # predict_word_or_number_sequence(model, "sample_alphanumeric.png")
