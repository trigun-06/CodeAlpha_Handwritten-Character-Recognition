import os
import io
import base64
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 36-Class Alphanumeric Label Map
LABEL_MAP = {i: str(i) for i in range(10)}
LABEL_MAP.update({i + 10: chr(65 + i) for i in range(26)})
NUM_CLASSES = 36
IMG_HEIGHT = 28
IMG_WIDTH = 28


class HandwritingModelEngine:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "emnist_alphanumeric_cnn.keras"))

        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        print(f"[INFO] Loading Keras model from: {self.model_path}")
        self.model = load_model(self.model_path)
        print("[INFO] Model loaded successfully.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess_char_crop(self, crop_gray: np.ndarray) -> np.ndarray:
        """
        Pads a cropped character into a centered 28x28 square image,
        matching the preprocessing pipeline of the EMNIST training data.
        """
        h, w = crop_gray.shape
        size = max(h, w) + 12
        square_img = np.zeros((size, size), dtype=np.uint8)

        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square_img[y_offset:y_offset + h, x_offset:x_offset + w] = crop_gray

        resized = cv2.resize(square_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        return np.expand_dims(normalized, axis=(0, -1))

    def decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decodes raw image bytes into a BGR numpy image."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image. Unsupported or corrupted format.")
        return img

    def binarize_image(self, gray: np.ndarray) -> np.ndarray:
        """
        Applies Otsu binarization and automatically detects whether the background
        is light or dark, ensuring strokes are white (255) on black (0).
        """
        T, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        is_dark_ink = np.sum(gray >= T) > np.sum(gray < T)
        if is_dark_ink:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def detect_document_roi(self, bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detects if the image contains a paper sheet placed on a background surface
        (such as a bedsheet, desk, or table) and extracts the paper's bounding box.
        If no external boundary is found (e.g. close-up note), returns the full image.
        """
        img_h, img_w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        thresh_doc = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)[1]
        cnts_doc, _ = cv2.findContours(thresh_doc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = img_h * img_w
        for c in sorted(cnts_doc, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            # Distinct document takes between 15% and 92% of the total photo area
            if (total_area * 0.15) < area < (total_area * 0.92):
                x, y, w, h = cv2.boundingRect(c)
                mx = int(w * 0.03)
                my = int(h * 0.04)
                rx = max(0, x + mx)
                ry = max(0, y + my)
                rw = min(img_w - rx, w - 2 * mx)
                rh = min(img_h - ry, h - 2 * my)
                if rw > 100 and rh > 100:
                    return (rx, ry, rw, rh)

        return (0, 0, img_w, img_h)

    def predict_single_character(self, image_bytes: bytes, top_k: int = 3, mode: str = "auto") -> Dict[str, Any]:
        """
        Intelligent prediction for drawing pad and isolated character inputs.
        Automatically handles:
        - Single drawn character or digit (returns top predictions & distribution)
        - Multi-character words/drawings (e.g. 'Hi', '42', 'AI') with dot-to-stem matching for 'i'/'j'
        - Generates annotated image with detected bounding boxes and confidences
        """
        bgr = self.decode_image_bytes(image_bytes)
        img_h, img_w = bgr.shape[:2]
        annotated_bgr = bgr.copy()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        thresh = self.binarize_image(gray)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter noise & canvas specks
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) >= 20 and not (w < 4 and h < 4):
                boxes.append((x, y, w, h))

        if not boxes:
            return {
                "character": "",
                "confidence": 0.0,
                "top_predictions": [],
                "characters": [],
                "annotated_image": None,
                "message": "No handwritten strokes detected on pad."
            }

        # Dot-to-stem matching (e.g. lowercase 'i' or 'j' diacritic dot)
        dots = []
        main_boxes = []
        for b in boxes:
            x, y, w, h = b
            if h <= 36 and w <= 36 and (w * h) <= 750:
                dots.append(b)
            else:
                main_boxes.append(list(b))

        # If only dots exist (e.g. dot or period), retain it
        if not main_boxes and dots:
            main_boxes = [list(d) for d in dots]
            dots = []

        for d in dots:
            dx, dy, dw, dh = d
            dcx = dx + dw / 2.0
            best_stem_idx = None
            min_dist = float('inf')

            for idx, s in enumerate(main_boxes):
                sx, sy, sw, sh = s
                scx = sx + sw / 2.0
                vert_gap = sy - (dy + dh)
                # Dot must be directly above stem or slightly overlapping top
                if -10 <= vert_gap <= max(60, sh * 1.3):
                    h_dist = abs(dcx - scx)
                    if h_dist <= max(35, sw * 0.8) and h_dist < min_dist:
                        min_dist = h_dist
                        best_stem_idx = idx

            if best_stem_idx is not None:
                sx, sy, sw, sh = main_boxes[best_stem_idx]
                nx = min(sx, dx)
                ny = min(sy, dy)
                nw = max(sx + sw, dx + dw) - nx
                nh = max(sy + sh, dy + dh) - ny
                main_boxes[best_stem_idx] = [nx, ny, nw, nh]
            else:
                if dw * dh >= 40:
                    main_boxes.append(list(d))

        # Sort left-to-right
        main_boxes = sorted(main_boxes, key=lambda b: b[0])

        # Merge heavily overlapping horizontal boxes (broken strokes)
        merged_boxes = []
        for b in main_boxes:
            if not merged_boxes:
                merged_boxes.append(b)
                continue
            px, py, pw, ph = merged_boxes[-1]
            cx, cy, cw, ch = b
            overlap_x = max(0, min(px + pw, cx + cw) - max(px, cx))
            if overlap_x > 0.5 * min(pw, cw):
                nx = min(px, cx)
                ny = min(py, cy)
                nw = max(px + pw, cx + cw) - nx
                nh = max(py + ph, cy + ch) - ny
                merged_boxes[-1] = [nx, ny, nw, nh]
            else:
                merged_boxes.append(b)

        # CASE 1: Single character drawn
        if len(merged_boxes) <= 1:
            x, y, w, h = merged_boxes[0] if merged_boxes else (0, 0, img_w, img_h)
            char_crop = thresh[y:y + h, x:x + w]
            processed_input = self.preprocess_char_crop(char_crop)
            raw_preds = self.model.predict(processed_input, verbose=0)[0].copy()

            if mode == "digits":
                raw_preds[10:] = 0.0
                if np.sum(raw_preds) > 0:
                    raw_preds /= np.sum(raw_preds)
            elif mode == "letters":
                raw_preds[:10] = 0.0
                if np.sum(raw_preds) > 0:
                    raw_preds /= np.sum(raw_preds)

            top_indices = np.argsort(raw_preds)[::-1][:top_k]
            top_predictions = [
                {
                    "character": LABEL_MAP[idx],
                    "confidence": round(float(raw_preds[idx]) * 100, 2)
                }
                for idx in top_indices
            ]
            best_idx = top_indices[0]
            best_char = LABEL_MAP[best_idx]
            best_conf = round(float(raw_preds[best_idx]) * 100, 2)

            # Draw visual annotation
            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (79, 70, 229), 2)
            cv2.putText(annotated_bgr, f"{best_char} ({int(best_conf)}%)", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (67, 56, 202), 2, cv2.LINE_AA)
            _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            return {
                "character": best_char,
                "confidence": best_conf,
                "top_predictions": top_predictions,
                "characters": [{
                    "character": best_char,
                    "confidence": best_conf,
                    "box": [int(x), int(y), int(w), int(h)]
                }],
                "annotated_image": annotated_base64,
                "message": "Successfully predicted character."
            }

        # CASE 2: Multi-character drawing (e.g. 'Hi', '42', 'AI', 'CODE')
        crops_data = []
        for (x, y, w, h) in merged_boxes:
            crop = thresh[y:y + h, x:x + w]
            proc = self.preprocess_char_crop(crop)
            preds = self.model.predict(proc, verbose=0)[0]
            crops_data.append((x, y, w, h, preds))

        # Check line context: numeric vs alphabetic vs mixed
        digit_conf_sum = sum(sum(p[:10]) for _, _, _, _, p in crops_data)
        letter_conf_sum = sum(sum(p[10:]) for _, _, _, _, p in crops_data)
        is_predominantly_numeric = (mode == "digits") or (mode == "auto" and digit_conf_sum > (letter_conf_sum * 1.25))
        is_predominantly_alphabetic = (mode == "letters") or (mode == "auto" and letter_conf_sum > (digit_conf_sum * 1.25))

        char_details = []
        full_chars = []
        top_preds_list = []

        for x, y, w, h, preds in crops_data:
            p_copy = preds.copy()
            if mode == "digits" or is_predominantly_numeric:
                p_copy[10:] = 0.0
                if np.sum(p_copy) > 0:
                    p_copy /= np.sum(p_copy)
            elif mode == "letters" or is_predominantly_alphabetic:
                p_copy[:10] = 0.0
                if np.sum(p_copy) > 0:
                    p_copy /= np.sum(p_copy)

            top_idx = np.argsort(p_copy)[::-1]
            ch = LABEL_MAP[top_idx[0]]
            cands = [LABEL_MAP[k] for k in top_idx[:3]]

            # Disambiguate common handwriting confusions
            if is_predominantly_alphabetic:
                if ch == "4" and "H" in cands:
                    ch = "H"
                elif ch == "0":
                    ch = "O"
                elif ch == "1" and "I" in cands:
                    ch = "I"
                elif ch == "L" and "I" in cands and (w / float(h)) < 0.65:
                    ch = "I"
            elif is_predominantly_numeric:
                if ch in ["G", "Q"]:
                    ch = "9"
                elif ch in ["I", "L"]:
                    ch = "1"
                elif ch == "O":
                    ch = "0"
                elif ch == "S":
                    ch = "5"
                elif ch == "Z":
                    ch = "2"
                elif ch == "B":
                    ch = "8"

            conf = round(float(p_copy[top_idx[0]]) * 100, 2)
            full_chars.append(ch)
            char_details.append({
                "character": ch,
                "confidence": conf,
                "box": [int(x), int(y), int(w), int(h)]
            })
            top_preds_list.append({
                "character": ch,
                "confidence": conf
            })

            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (79, 70, 229), 2)
            cv2.putText(annotated_bgr, f"{ch} ({int(conf)}%)", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (67, 56, 202), 2, cv2.LINE_AA)

        assembled_text = "".join(full_chars)
        avg_conf = round(sum(c["confidence"] for c in char_details) / len(char_details), 2)

        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return {
            "character": assembled_text,
            "confidence": avg_conf,
            "top_predictions": top_preds_list,
            "characters": char_details,
            "annotated_image": annotated_base64,
            "message": "Successfully predicted drawing sequence."
        }

    def predict_note_sequence(self, image_bytes: bytes, annotate: bool = True, mode: str = "auto") -> Dict[str, Any]:
        """
        Universal handwriting transcription pipeline:
        - mode='auto': Automatically detects whether word/line is numeric or alphabetic.
        - mode='digits': Constrains output strictly to digits (0-9).
        - mode='letters': Constrains output strictly to letters (A-Z).
        - mode='alphanumeric': Unconstrained 0-9 and A-Z.
        """
        bgr = self.decode_image_bytes(image_bytes)
        annotated_bgr = bgr.copy()
        img_h, img_w = bgr.shape[:2]

        # 1. Document / Paper Boundary Isolation
        rx, ry, rw, rh = self.detect_document_roi(bgr)

        roi_bgr = bgr[ry:ry + rh, rx:rx + rw]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        mean_val = np.mean(roi_gray)
        if mean_val < 60:
            # Inverted / dark chalkboard mode
            _, thresh = cv2.threshold(roi_gray, 40, 255, cv2.THRESH_BINARY)
        else:
            # Morphological Illumination Flattening (removes shadows, dim lighting & lens vignetting)
            ksize = max(25, (min(rw, rh) // 30) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            bg = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
            norm = cv2.divide(roi_gray, bg, scale=255)
            _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_w = max(6, int(rw * 0.003))
        if rw < 600:
            max_w = max(int(rw * 0.90), 200)
        else:
            max_w = max(int(rw * 0.40), 300)
        min_h = max(10, int(rh * 0.008))
        max_h = int(rh * 0.85)
        min_area = max(20, int((rw * rh) * 0.00003))

        raw_boxes = []
        dots = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if (h <= 36 and w <= 36 and area <= 750) or (h <= int(rh * 0.05) and w <= int(rw * 0.05) and area <= 750):
                if area >= 15:
                    dots.append((x, y, w, h))
                continue
            if min_w <= w <= max_w and min_h <= h <= max_h and area >= min_area:
                aspect = w / float(h)
                if 0.08 <= aspect <= 3.2:
                    raw_boxes.append([x, y, w, h])

        # Merge dots with main character stems (e.g. lowercase i, j)
        for d in dots:
            dx, dy, dw, dh = d
            dcx = dx + dw / 2.0
            best_stem_idx = None
            min_dist = float('inf')

            for idx, s in enumerate(raw_boxes):
                sx, sy, sw, sh = s
                scx = sx + sw / 2.0
                vert_gap = sy - (dy + dh)
                if -10 <= vert_gap <= max(60, sh * 1.3):
                    h_dist = abs(dcx - scx)
                    if h_dist <= max(35, sw * 0.8) and h_dist < min_dist:
                        min_dist = h_dist
                        best_stem_idx = idx

            if best_stem_idx is not None:
                sx, sy, sw, sh = raw_boxes[best_stem_idx]
                nx = min(sx, dx)
                ny = min(sy, dy)
                nw = max(sx + sw, dx + dw) - nx
                nh = max(sy + sh, dy + dh) - ny
                raw_boxes[best_stem_idx] = [nx, ny, nw, nh]
            else:
                if dw * dh >= min_area and min_w <= dw <= max_w and min_h <= dh <= max_h:
                    raw_boxes.append([dx, dy, dw, dh])

        raw_boxes = [tuple(b) for b in raw_boxes]

        if not raw_boxes:
            return {
                "recognized_text": "",
                "total_characters": 0,
                "characters": [],
                "annotated_image": None,
                "message": "No handwritten characters detected in the note."
            }

        # Remove nested/enclosed sub-boxes
        clean_boxes = []
        for b in sorted(raw_boxes, key=lambda b: b[2] * b[3], reverse=True):
            x1, y1, w1, h1 = b
            is_nested = False
            for cb in clean_boxes:
                x2, y2, w2, h2 = cb
                if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                    is_nested = True
                    break
            if not is_nested:
                clean_boxes.append(b)

        # Merge broken strokes belonging to the same character (horizontal overlap)
        merged_boxes = []
        for b in sorted(clean_boxes, key=lambda b: b[0]):
            if not merged_boxes:
                merged_boxes.append(b)
                continue
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            curr_x, curr_y, curr_w, curr_h = b
            overlap_x = max(0, min(prev_x + prev_w, curr_x + curr_w) - max(prev_x, curr_x))
            min_w_box = min(prev_w, curr_w)
            if overlap_x > 0.45 * min_w_box:
                nx = min(prev_x, curr_x)
                ny = min(prev_y, curr_y)
                nw = max(prev_x + prev_w, curr_x + curr_w) - nx
                nh = max(prev_y + prev_h, curr_y + curr_h) - ny
                merged_boxes[-1] = (nx, ny, nw, nh)
            else:
                merged_boxes.append(b)

        # Cluster lines
        boxes = sorted(merged_boxes, key=lambda b: b[1])
        median_h = np.median([b[3] for b in boxes]) if boxes else 20
        line_threshold = max(15, median_h * 0.6)

        lines = []
        current_line = [boxes[0]]

        for b in boxes[1:]:
            prev_b = current_line[-1]
            if abs(b[1] - prev_b[1]) < line_threshold:
                current_line.append(b)
            else:
                lines.append(sorted(current_line, key=lambda box: box[0]))
                current_line = [b]
        if current_line:
            lines.append(sorted(current_line, key=lambda box: box[0]))

        recognized_lines = []
        all_char_details = []

        for line in lines:
            line_chars = []
            median_char_w = np.median([b[2] for b in line]) if line else 20
            space_threshold = max(20, median_char_w * 0.45)

            # Pre-pass: Collect raw probabilities for all characters on this line
            line_crops_data = []
            for (x, y, w, h) in line:
                crop = thresh[y:y + h, x:x + w]
                proc = self.preprocess_char_crop(crop)
                preds = self.model.predict(proc, verbose=0)[0]
                line_crops_data.append((x, y, w, h, preds))

            # Determine line context: numeric vs alphabetic vs mixed
            digit_conf_sum = sum(sum(p[:10]) for _, _, _, _, p in line_crops_data)
            letter_conf_sum = sum(sum(p[10:]) for _, _, _, _, p in line_crops_data)
            is_predominantly_numeric = (mode == "digits") or (mode == "auto" and digit_conf_sum > (letter_conf_sum * 1.25))
            is_predominantly_alphabetic = (mode == "letters") or (mode == "auto" and letter_conf_sum > (digit_conf_sum * 1.25))

            if is_predominantly_numeric or mode == "digits":
                space_threshold = max(35, median_char_w * 1.05)
            else:
                space_threshold = max(22, median_char_w * 0.50)

            for i, (x, y, w, h, preds) in enumerate(line_crops_data):
                if i > 0 and (x - (line[i - 1][0] + line[i - 1][2])) > space_threshold:
                    line_chars.append(" ")

                preds_copy = preds.copy()
                if mode == "digits" or is_predominantly_numeric:
                    # Mask to digits 0-9
                    preds_copy[10:] = 0.0
                    if np.sum(preds_copy) > 0:
                        preds_copy /= np.sum(preds_copy)
                elif mode == "letters" or is_predominantly_alphabetic:
                    # Mask to letters A-Z
                    preds_copy[:10] = 0.0
                    if np.sum(preds_copy) > 0:
                        preds_copy /= np.sum(preds_copy)

                top_indices = np.argsort(preds_copy)[::-1]
                char_str = LABEL_MAP[top_indices[0]]
                cand_chars = [LABEL_MAP[k] for k in top_indices[:3]]

                # Disambiguate common handwriting confusions
                if is_predominantly_alphabetic:
                    if char_str == "4" and "H" in cand_chars:
                        char_str = "H"
                    elif char_str == "0":
                        char_str = "O"
                    elif char_str == "1" and "I" in cand_chars:
                        char_str = "I"
                    elif char_str == "L" and "I" in cand_chars and (w / float(h)) < 0.62:
                        char_str = "I"
                elif is_predominantly_numeric:
                    if char_str in ["G", "Q"]:
                        char_str = "9"
                    elif char_str in ["I", "L"]:
                        char_str = "1"
                    elif char_str == "O":
                        char_str = "0"
                    elif char_str == "S":
                        char_str = "5"
                    elif char_str == "Z":
                        char_str = "2"
                    elif char_str == "B":
                        char_str = "8"

                conf = round(float(preds_copy[top_indices[0]]) * 100, 2)
                line_chars.append(char_str)

                gx = rx + x
                gy = ry + y

                all_char_details.append({
                    "character": char_str,
                    "confidence": conf,
                    "box": [int(gx), int(gy), int(w), int(h)]
                })

                if annotate:
                    cv2.rectangle(annotated_bgr, (gx, gy), (gx + w, gy + h), (79, 70, 229), max(2, int(img_h * 0.002)))
                    label_text = f"{char_str} ({int(conf)}%)"
                    cv2.putText(
                        annotated_bgr,
                        label_text,
                        (gx, max(25, gy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.6, img_h * 0.0003),
                        (67, 56, 202),
                        max(2, int(img_h * 0.001)),
                        cv2.LINE_AA
                    )

            # Contextual Word Post-Processing
            text = "".join(line_chars)
            words = text.split(" ")
            cleaned_words = []
            for word in words:
                if sum(1 for ch in word if ch.isalpha()) >= 2:
                    word = word.replace("0", "O")
                cleaned_words.append(word)

            recognized_lines.append(" ".join(cleaned_words))

        full_recognized_text = "\n".join(recognized_lines)

        annotated_base64 = None
        if annotate:
            preview_img = annotated_bgr
            if max(img_h, img_w) > 1600:
                scale = 1600.0 / max(img_h, img_w)
                preview_img = cv2.resize(annotated_bgr, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA)

            _, buffer = cv2.imencode(".jpg", preview_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return {
            "recognized_text": full_recognized_text,
            "total_characters": len(all_char_details),
            "characters": all_char_details,
            "annotated_image": annotated_base64,
            "message": "Success"
        }


# Singleton engine instance
_engine_instance = None


def get_engine() -> HandwritingModelEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HandwritingModelEngine()
    return _engine_instance
