# -*- coding: utf-8 -*-
"""
Scrawl — Neural Handwritten OCR & Intelligent Transcription (Standalone Runner)
Engineered by Trigun Parvathapuram (CodeAlpha ML Track Task 3)
Combines the Fine-Tuned CNN Model, FastAPI Server, and Complete Animated Frontend UI in a Single Executable File.
Usage:
    python textreader_standalone.py
"""
import os
import sys
import io
import base64
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

LABEL_MAP = {i: str(i) for i in range(10)}
LABEL_MAP.update({i + 10: chr(65 + i) for i in range(26)})
NUM_CLASSES = 36
IMG_HEIGHT = 28
IMG_WIDTH = 28

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emnist_alphanumeric_cnn.keras")
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model weights not found at: {MODEL_FILE}")
        print(f"[INFO] Loading model from {MODEL_FILE}...")
        _MODEL = load_model(MODEL_FILE)
        print("[INFO] Model loaded successfully!")
    return _MODEL

def preprocess_char_crop(crop_gray: np.ndarray) -> np.ndarray:
    h, w = crop_gray.shape
    size = max(h, w) + 12
    square_img = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = crop_gray
    resized = cv2.resize(square_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

def decode_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data.")
    return img

def binarize(gray: np.ndarray) -> np.ndarray:
    mean_val = np.mean(gray)
    if mean_val > 127:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def predict_single(image_bytes: bytes, top_k: int = 3, mode: str = "auto") -> Dict[str, Any]:
    model = get_model()
    bgr = decode_image(image_bytes)
    img_h, img_w = bgr.shape[:2]
    annotated = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = binarize(gray)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w * h) >= 20 and not (w < 4 and h < 4):
            boxes.append((x, y, w, h))

    if not boxes:
        return {"character": "", "confidence": 0.0, "top_predictions": [], "characters": [], "annotated_image": None}

    dots, main_boxes = [], []
    for b in boxes:
        x, y, w, h = b
        if h <= 36 and w <= 36 and (w * h) <= 750:
            dots.append(b)
        else:
            main_boxes.append(list(b))

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
        elif dw * dh >= 40:
            main_boxes.append(list(d))

    main_boxes = sorted(main_boxes, key=lambda b: b[0])
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

    if len(merged_boxes) <= 1:
        x, y, w, h = merged_boxes[0] if merged_boxes else (0, 0, img_w, img_h)
        crop = thresh[y:y + h, x:x + w]
        proc = preprocess_char_crop(crop)
        preds = model.predict(proc, verbose=0)[0].copy()
        if mode == "digits":
            preds[10:] = 0.0
            if np.sum(preds) > 0: preds /= np.sum(preds)
        elif mode == "letters":
            preds[:10] = 0.0
            if np.sum(preds) > 0: preds /= np.sum(preds)
        top_indices = np.argsort(preds)[::-1][:top_k]
        best_char = LABEL_MAP[top_indices[0]]
        best_conf = round(float(preds[top_indices[0]]) * 100, 2)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (79, 70, 229), 2)
        cv2.putText(annotated, f"{best_char} ({int(best_conf)}%)", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (67, 56, 202), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
        return {
            "character": best_char,
            "confidence": best_conf,
            "top_predictions": [{"character": LABEL_MAP[i], "confidence": round(float(preds[i]) * 100, 2)} for i in top_indices],
            "characters": [{"character": best_char, "confidence": best_conf, "box": [int(x), int(y), int(w), int(h)]}],
            "annotated_image": ann_b64
        }

    crops_data = []
    for (x, y, w, h) in merged_boxes:
        crop = thresh[y:y + h, x:x + w]
        preds = model.predict(preprocess_char_crop(crop), verbose=0)[0]
        crops_data.append((x, y, w, h, preds))

    digit_sum = sum(sum(p[:10]) for _, _, _, _, p in crops_data)
    letter_sum = sum(sum(p[10:]) for _, _, _, _, p in crops_data)
    is_num = (mode == "digits") or (mode == "auto" and digit_sum > (letter_sum * 1.25))
    is_alpha = (mode == "letters") or (mode == "auto" and letter_sum > (digit_sum * 1.25))

    char_details, full_chars, top_preds = [], [], []
    for x, y, w, h, preds in crops_data:
        p = preds.copy()
        if mode == "digits" or is_num:
            p[10:] = 0.0
            if np.sum(p) > 0: p /= np.sum(p)
        elif mode == "letters" or is_alpha:
            p[:10] = 0.0
            if np.sum(p) > 0: p /= np.sum(p)
        top_idx = np.argsort(p)[::-1]
        ch = LABEL_MAP[top_idx[0]]
        cands = [LABEL_MAP[k] for k in top_idx[:3]]
        if is_alpha:
            if ch == "4" and "H" in cands: ch = "H"
            elif ch == "0": ch = "O"
            elif ch == "1" and "I" in cands: ch = "I"
            elif ch == "L" and "I" in cands and (w / float(h)) < 0.65: ch = "I"
        elif is_num:
            if ch in ["G", "Q"]: ch = "9"
            elif ch in ["I", "L"]: ch = "1"
            elif ch == "O": ch = "0"
            elif ch == "S": ch = "5"
            elif ch == "Z": ch = "2"
            elif ch == "B": ch = "8"
        conf = round(float(p[top_idx[0]]) * 100, 2)
        full_chars.append(ch)
        char_details.append({"character": ch, "confidence": conf, "box": [int(x), int(y), int(w), int(h)]})
        top_preds.append({"character": ch, "confidence": conf})
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (79, 70, 229), 2)
        cv2.putText(annotated, f"{ch} ({int(conf)}%)", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (67, 56, 202), 2, cv2.LINE_AA)

    assembled = "".join(full_chars)
    avg_conf = round(sum(c["confidence"] for c in char_details) / len(char_details), 2)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
    return {
        "character": assembled,
        "confidence": avg_conf,
        "top_predictions": top_preds,
        "characters": char_details,
        "annotated_image": ann_b64
    }

def detect_document_roi(bgr: np.ndarray):
    img_h, img_w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh_doc = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)[1]
    cnts_doc, _ = cv2.findContours(thresh_doc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = img_h * img_w
    for c in sorted(cnts_doc, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if (total_area * 0.15) < area < (total_area * 0.92):
            x, y, w, h = cv2.boundingRect(c)
            mx, my = int(w * 0.03), int(h * 0.04)
            rx, ry = max(0, x + mx), max(0, y + my)
            rw, rh = min(img_w - rx, w - 2 * mx), min(img_h - ry, h - 2 * my)
            if rw > 100 and rh > 100:
                return (rx, ry, rw, rh)
    return (0, 0, img_w, img_h)

def predict_sequence(image_bytes: bytes, annotate: bool = True, mode: str = "auto") -> Dict[str, Any]:
    model = get_model()
    bgr = decode_image(image_bytes)
    annotated = bgr.copy()
    img_h, img_w = bgr.shape[:2]

    rx, ry, rw, rh = detect_document_roi(bgr)
    roi_bgr = bgr[ry:ry + rh, rx:rx + rw]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    mean_val = np.mean(roi_gray)
    if mean_val < 60:
        _, thresh = cv2.threshold(roi_gray, 40, 255, cv2.THRESH_BINARY)
    else:
        ksize = max(25, (min(rw, rh) // 30) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        bg = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
        norm = cv2.divide(roi_gray, bg, scale=255)
        _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_w = max(6, int(rw * 0.003))
    if rw < 600:
        max_w = max(int(rw * 0.90), 200)
    else:
        max_w = max(int(rw * 0.40), 300)
    min_h = max(10, int(rh * 0.008))
    max_h = int(rh * 0.85)
    min_area = max(20, int((rw * rh) * 0.00003))

    raw_boxes, dots = [], []
    for c in cnts:
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
        elif dw * dh >= min_area and min_w <= dw <= max_w and min_h <= dh <= max_h:
            raw_boxes.append([dx, dy, dw, dh])

    raw_boxes = [tuple(b) for b in raw_boxes]

    if not raw_boxes:
        return {"recognized_text": "", "total_characters": 0, "characters": [], "annotated_image": None}

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

    boxes = sorted(merged_boxes, key=lambda b: b[1])
    median_h = np.median([b[3] for b in boxes]) if boxes else 20
    line_threshold = max(15, median_h * 0.6)

    lines, current_line = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current_line[-1][1]) < line_threshold:
            current_line.append(b)
        else:
            lines.append(sorted(current_line, key=lambda box: box[0]))
            current_line = [b]
    if current_line:
        lines.append(sorted(current_line, key=lambda box: box[0]))

    recognized_lines, char_details = [], []
    for line in lines:
        line_chars = []
        median_char_w = np.median([b[2] for b in line]) if line else 20

        line_crops_data = []
        for (x, y, w, h) in line:
            crop = thresh[y:y + h, x:x + w]
            preds = model.predict(preprocess_char_crop(crop), verbose=0)[0]
            line_crops_data.append((x, y, w, h, preds))

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
                preds_copy[10:] = 0.0
                if np.sum(preds_copy) > 0:
                    preds_copy /= np.sum(preds_copy)
            elif mode == "letters" or is_predominantly_alphabetic:
                preds_copy[:10] = 0.0
                if np.sum(preds_copy) > 0:
                    preds_copy /= np.sum(preds_copy)

            top_indices = np.argsort(preds_copy)[::-1]
            char_str = LABEL_MAP[top_indices[0]]
            cand_chars = [LABEL_MAP[k] for k in top_indices[:3]]

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

            gx, gy = rx + x, ry + y
            char_details.append({"character": char_str, "confidence": conf, "box": [int(gx), int(gy), int(w), int(h)]})

            if annotate:
                cv2.rectangle(annotated, (gx, gy), (gx + w, gy + h), (79, 70, 229), max(2, int(img_h * 0.002)))
                cv2.putText(annotated, f"{char_str} ({int(conf)}%)", (gx, max(25, gy - 10)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img_h * 0.0003), (67, 56, 202), max(2, int(img_h * 0.001)), cv2.LINE_AA)

        text = "".join(line_chars)
        words = text.split(" ")
        cleaned_words = []
        for word in words:
            if sum(1 for ch in word if ch.isalpha()) >= 2:
                word = word.replace("0", "O")
            cleaned_words.append(word)
        recognized_lines.append(" ".join(cleaned_words))

    annotated_base64 = None
    if annotate:
        preview_img = annotated
        if max(img_h, img_w) > 1600:
            scale = 1600.0 / max(img_h, img_w)
            preview_img = cv2.resize(annotated, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", preview_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "recognized_text": chr(10).join(recognized_lines),
        "total_characters": len(char_details),
        "characters": char_details,
        "annotated_image": annotated_base64
    }

app = FastAPI(title="Scrawl Standalone", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Base64Payload(BaseModel):
    image: str
    top_k: Optional[int] = 3
    mode: Optional[str] = "auto"

def parse_b64(data_url: str) -> bytes:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "application": "Scrawl",
        "owner": "Trigun Parvathapuram",
        "model_loaded": get_model() is not None,
        "num_classes": NUM_CLASSES
    }

@app.get("/api/ownership")
def ownership():
    return {
        "project": "Scrawl",
        "author": "Trigun Parvathapuram",
        "track": "CodeAlpha ML Internship - Task 3",
        "validation_accuracy": "96.32%"
    }

@app.post("/api/predict/character")
async def api_predict_char(file: UploadFile = File(...), top_k: int = Form(3), mode: str = Form("auto")):
    data = await file.read()
    return predict_single(data, top_k=top_k, mode=mode)

@app.post("/api/predict/character/base64")
def api_predict_char_b64(payload: Base64Payload):
    return predict_single(parse_b64(payload.image), top_k=payload.top_k or 3, mode=payload.mode or "auto")

@app.post("/api/predict/sequence")
async def api_predict_seq(file: UploadFile = File(...), mode: str = Form("auto")):
    data = await file.read()
    return predict_sequence(data, annotate=True, mode=mode)

@app.post("/api/predict/sequence/base64")
def api_predict_seq_b64(payload: Base64Payload):
    return predict_sequence(parse_b64(payload.image), annotate=True, mode=payload.mode or "auto")

EMBEDDED_HTML_B64 = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CiAgPG1ldGEgY2hhcnNldD0iVVRGLTgiPgogIDxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyI+CiAgPHRpdGxlPlNjcmF3bCDigJQgTmV1cmFsIEhhbmR3cml0dGVuIE9DUiAmIEludGVsbGlnZW50IFRyYW5zY3JpcHRpb248L3RpdGxlPgogIDxsaW5rIHJlbD0icHJlY29ubmVjdCIgaHJlZj0iaHR0cHM6Ly9mb250cy5nb29nbGVhcGlzLmNvbSI+CiAgPGxpbmsgcmVsPSJwcmVjb25uZWN0IiBocmVmPSJodHRwczovL2ZvbnRzLmdzdGF0aWMuY29tIiBjcm9zc29yaWdpbj4KICA8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PVBsdXMrSmFrYXJ0YStTYW5zOndnaHRANDAwOzUwMDs2MDA7NzAwOzgwMCZmYW1pbHk9SmV0QnJhaW5zK01vbm86d2dodEA0MDA7NTAwOzYwMCZkaXNwbGF5PXN3YXAiIHJlbD0ic3R5bGVzaGVldCI+CiAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSIvc3RhdGljL3N0eWxlLmNzcyI+CjwvaGVhZD4KPGJvZHk+CiAgPCEtLSBBbWJpZW50IEFuaW1hdGVkIEJhY2tncm91bmQgR2xvd3MgLS0+CiAgPGRpdiBjbGFzcz0iYW1iaWVudC1nbG93IGdsb3ctMSI+PC9kaXY+CiAgPGRpdiBjbGFzcz0iYW1iaWVudC1nbG93IGdsb3ctMiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0iYW1iaWVudC1nbG93IGdsb3ctMyI+PC9kaXY+CgogIDwhLS0gSGVhZGVyIC0tPgogIDxoZWFkZXIgY2xhc3M9ImFwcC1oZWFkZXIiPgogICAgPGRpdiBjbGFzcz0iaGVhZGVyLWNvbnRhaW5lciI+CiAgICAgIDxkaXYgY2xhc3M9ImJyYW5kIj4KICAgICAgICA8ZGl2IGNsYXNzPSJsb2dvLWljb24gc2NyYXdsLWdsb3ciPgogICAgICAgICAgPHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMi4yIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPgogICAgICAgICAgICA8cGF0aCBkPSJNMTIgMjBoOSIvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTYuNSAzLjVhMi4xMiAyLjEyIDAgMCAxIDMgM0w3IDE5bC00IDEgMS00WiIvPgogICAgICAgICAgPC9zdmc+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iYnJhbmQtdGV4dCI+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJsb2dvLXRpdGxlLXJvdyI+CiAgICAgICAgICAgIDxoMSBjbGFzcz0ibG9nby10aXRsZSI+U2NyYXdsPHNwYW4gY2xhc3M9ImJyYW5kLWRvdCI+Ljwvc3Bhbj48L2gxPgogICAgICAgICAgICA8c3BhbiBjbGFzcz0idmVyc2lvbi10YWciPnYyLjA8L3NwYW4+CiAgICAgICAgICA8L2Rpdj4KICAgICAgICAgIDxwIGNsYXNzPSJsb2dvLXN1YnRpdGxlIj5OZXVyYWwgSGFuZHdyaXR0ZW4gT0NSICYgSW50ZWxsaWdlbnQgVHJhbnNjcmlwdGlvbjwvcD4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CgogICAgICA8ZGl2IGNsYXNzPSJoZWFkZXItYWN0aW9ucyI+CiAgICAgICAgPCEtLSBNb2RlbCBTdGF0dXMgUGlsbCAtLT4KICAgICAgICA8ZGl2IGlkPSJiYWNrZW5kU3RhdHVzIiBjbGFzcz0ic3RhdHVzLXBpbGwgc3RhdHVzLWNoZWNraW5nIj4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdGF0dXMtZG90Ij48L3NwYW4+CiAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RhdHVzLXRleHQiPkNvbm5lY3RpbmcgdG8gbW9kZWwuLi48L3NwYW4+CiAgICAgICAgPC9kaXY+CgogICAgICAgIDwhLS0gT3duZXIgUHJvZmlsZSAmIENvbnRyb2xzIFRyaWdnZXIgLS0+CiAgICAgICAgPGJ1dHRvbiBpZD0ib3Blbk93bmVyTW9kYWxCdG4iIGNsYXNzPSJvd25lci1waWxsLWJ0biIgdGl0bGU9IlZpZXcgT3duZXJzaGlwICYgTW9kZWwgQ29udHJvbHMiPgogICAgICAgICAgPHNwYW4gY2xhc3M9Im93bmVyLWF2YXRhciI+VFA8L3NwYW4+CiAgICAgICAgICA8c3BhbiBjbGFzcz0ib3duZXItbmFtZS1zaG9ydCI+VHJpZ3VuIFAuPC9zcGFuPgogICAgICAgICAgPHNwYW4gY2xhc3M9Im93bmVyLXZlcmlmaWVkLWNoZWNrIiB0aXRsZT0iVmVyaWZpZWQgUHJvamVjdCBPd25lciI+CiAgICAgICAgICAgIDxzdmcgd2lkdGg9IjEyIiBoZWlnaHQ9IjEyIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9ImN1cnJlbnRDb2xvciI+CiAgICAgICAgICAgICAgPHBhdGggZD0iTTkgMTYuMTdMNC44MyAxMmwtMS40MiAxLjQxTDkgMTkgMjEgN2wtMS40MS0xLjQxeiIvPgogICAgICAgICAgICA8L3N2Zz4KICAgICAgICAgIDwvc3Bhbj4KICAgICAgICA8L2J1dHRvbj4KCiAgICAgICAgPCEtLSBBUEkgRG9jdW1lbnRhdGlvbiAtLT4KICAgICAgICA8YSBocmVmPSIvZG9jcyIgdGFyZ2V0PSJfYmxhbmsiIGNsYXNzPSJhcGktZG9jcy1idG4iIHRpdGxlPSJPcGVuIFN3YWdnZXIgQVBJIGRvY3VtZW50YXRpb24iPgogICAgICAgICAgPHN2ZyB3aWR0aD0iMTUiIGhlaWdodD0iMTUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4KICAgICAgICAgICAgPHBvbHlsaW5lIHBvaW50cz0iMTYgMTggMjIgMTIgMTYgNiIvPgogICAgICAgICAgICA8cG9seWxpbmUgcG9pbnRzPSI4IDYgMiAxMiA4IDE4Ii8+CiAgICAgICAgICA8L3N2Zz4KICAgICAgICAgIEFQSQogICAgICAgIDwvYT4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICA8L2hlYWRlcj4KCiAgPCEtLSBNYWluIENvbnRlbnQgQ29udGFpbmVyIC0tPgogIDxtYWluIGNsYXNzPSJtYWluLWNvbnRlbnQiPgogICAgPCEtLSBUb3AgTW9kZSBTd2l0Y2hlciAtLT4KICAgIDxkaXYgY2xhc3M9Im5hdi13cmFwcGVyIj4KICAgICAgPG5hdiBjbGFzcz0ibW9kZS10YWJzIiByb2xlPSJ0YWJsaXN0Ij4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJ0YWItYnRuIGFjdGl2ZSIgaWQ9InRhYk5vdGVCdG4iIGRhdGEtdGFyZ2V0PSJub3RlTW9kZSIgcm9sZT0idGFiIiBhcmlhLXNlbGVjdGVkPSJ0cnVlIj4KICAgICAgICAgIDxzdmcgd2lkdGg9IjE3IiBoZWlnaHQ9IjE3IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xNCAySDZhMiAyIDAgMCAwLTIgMnYxNmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAwIDAgMCAyLTJWOHoiLz4KICAgICAgICAgICAgPHBvbHlsaW5lIHBvaW50cz0iMTQgMiAxNCA4IDIwIDgiLz4KICAgICAgICAgICAgPGxpbmUgeDE9IjE2IiB5MT0iMTMiIHgyPSI4IiB5Mj0iMTMiLz4KICAgICAgICAgICAgPGxpbmUgeDE9IjE2IiB5MT0iMTciIHgyPSI4IiB5Mj0iMTciLz4KICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgTm90ZSAmIFNlcXVlbmNlIE9DUgogICAgICAgIDwvYnV0dG9uPgoKICAgICAgICA8YnV0dG9uIGNsYXNzPSJ0YWItYnRuIiBpZD0idGFiQ2FudmFzQnRuIiBkYXRhLXRhcmdldD0iY2FudmFzTW9kZSIgcm9sZT0idGFiIiBhcmlhLXNlbGVjdGVkPSJmYWxzZSI+CiAgICAgICAgICA8c3ZnIHdpZHRoPSIxNyIgaGVpZ2h0PSIxNyIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPgogICAgICAgICAgICA8cGF0aCBkPSJNMjEuMTc0IDYuODEyYTEgMSAwIDAgMC0zLjk4Ni0zLjk4N0wzLjg0MiAxNi4xNzRhMiAyIDAgMCAwLS41LjgzbC0xLjMyMSA0LjM1MmEuNS41IDAgMCAwIC42MjMuNjIybDQuMzUzLTEuMzJhMiAyIDAgMCAwIC44My0uNDk3eiIvPgogICAgICAgICAgPC9zdmc+CiAgICAgICAgICBMaXZlIERyYXdpbmcgUGFkCiAgICAgICAgPC9idXR0b24+CgogICAgICAgIDxidXR0b24gY2xhc3M9InRhYi1idG4gdGFiLW93bmVyLWhpZ2hsaWdodCIgaWQ9InRhYk93bmVyQnRuIiBkYXRhLXRhcmdldD0ib3duZXJTZWN0aW9uIj4KICAgICAgICAgIDxzdmcgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMiAybDMuMDkgNi4yNkwyMiA5LjI3bC01IDQuODcgMS4xOCA2Ljg4TDEyIDE3Ljc3bC02LjE4IDMuMjVMNyAxNC4xNCAyIDkuMjdsNi45MS0xLjAxTDEyIDJ6Ii8+CiAgICAgICAgICA8L3N2Zz4KICAgICAgICAgIE93bmVyIENvbnRyb2xzCiAgICAgICAgPC9idXR0b24+CiAgICAgIDwvbmF2PgogICAgPC9kaXY+CgogICAgPCEtLSBNT0RFIDE6IE5vdGUgJiBTZXF1ZW5jZSBSZWFkZXIgLS0+CiAgICA8c2VjdGlvbiBpZD0ibm90ZU1vZGUiIGNsYXNzPSJ0YWItdmlldyBhY3RpdmUiPgogICAgICA8ZGl2IGNsYXNzPSJ2aWV3LWdyaWQiPgogICAgICAgIDwhLS0gVXBsb2FkICYgT3B0aW9ucyBDYXJkIC0tPgogICAgICAgIDxkaXYgY2xhc3M9ImNhcmQgY2xlYW4tY2FyZCBhbmltYXRlLWNhcmQiPgogICAgICAgICAgPGRpdiBjbGFzcz0iY2FyZC1oZWFkZXIiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJjYXJkLXRpdGxlLXJvdyI+CiAgICAgICAgICAgICAgPGgyIGNsYXNzPSJjYXJkLXRpdGxlIj5VcGxvYWQgSGFuZHdyaXR0ZW4gTm90ZTwvaDI+CiAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InBpbGwtYmFkZ2UgcGlsbC1ibHVlIj5BSSBWaXNpb248L3NwYW4+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8cCBjbGFzcz0iY2FyZC1kZXNjIj5VcGxvYWQgYSBwaG90bywgc25hcHNob3QsIG9yIGRvY3VtZW50IHNjYW4gb2YgaGFuZHdyaXR0ZW4gd29yZHMgb3IgbnVtYmVycy48L3A+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8IS0tIERyb3B6b25lIC0tPgogICAgICAgICAgPGRpdiBpZD0iZHJvcHpvbmUiIGNsYXNzPSJkcm9wem9uZSI+CiAgICAgICAgICAgIDxpbnB1dCB0eXBlPSJmaWxlIiBpZD0ibm90ZUZpbGVJbnB1dCIgYWNjZXB0PSJpbWFnZS9wbmcsaW1hZ2UvanBlZyxpbWFnZS9qcGcsaW1hZ2Uvd2VicCIgY2xhc3M9ImZpbGUtaW5wdXQiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJkcm9wem9uZS1jb250ZW50Ij4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJkcm9wem9uZS1pY29uIHB1bHNlLXNvZnQiPgogICAgICAgICAgICAgICAgPHN2ZyB3aWR0aD0iMzYiIGhlaWdodD0iMzYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPgogICAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMjEgMTV2NGEyIDIgMCAwIDEtMiAySDVhMiAyIDAgMCAxLTItMnYtNCIvPgogICAgICAgICAgICAgICAgICA8cG9seWxpbmUgcG9pbnRzPSIxNyA4IDEyIDMgNyA4Ii8+CiAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIxMiIgeTE9IjMiIHgyPSIxMiIgeTI9IjE1Ii8+CiAgICAgICAgICAgICAgICA8L3N2Zz4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8cCBjbGFzcz0iZHJvcHpvbmUtdGl0bGUiPkNsaWNrIHRvIHVwbG9hZCBvciBkcmFnICYgZHJvcDwvcD4KICAgICAgICAgICAgICA8cCBjbGFzcz0iZHJvcHpvbmUtc3ViIj5TdXBwb3J0cyBjYW1lcmEgcGhvdG9zIChQTkcsIEpQRywgSlBFRywgV2ViUCk8L3A+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgPC9kaXY+CgogICAgICAgICAgPCEtLSBJbWFnZSBQcmV2aWV3IC0tPgogICAgICAgICAgPGRpdiBpZD0ibm90ZVByZXZpZXdDb250YWluZXIiIGNsYXNzPSJwcmV2aWV3LWNvbnRhaW5lciBoaWRkZW4iPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJwcmV2aWV3LWJveCI+CiAgICAgICAgICAgICAgPGltZyBpZD0ibm90ZVByZXZpZXdJbWciIHNyYz0iIiBhbHQ9IlVwbG9hZGVkIG5vdGUgcHJldmlldyIgY2xhc3M9InByZXZpZXctaW1nIj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDxidXR0b24gaWQ9ImNsZWFyTm90ZUJ0biIgY2xhc3M9ImJ0biBidG4tc2Vjb25kYXJ5IGJ0bi1zbSBjbGVhci1idG4iIHRpdGxlPSJSZW1vdmUgaW1hZ2UiPgogICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjE0IiBoZWlnaHQ9IjE0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjE4IiB5MT0iNiIgeDI9IjYiIHkyPSIxOCI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjYiIHkxPSI2IiB4Mj0iMTgiIHkyPSIxOCI+PC9saW5lPgogICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICAgIFJlbW92ZSBJbWFnZQogICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgIDwvZGl2PgoKICAgICAgICAgIDwhLS0gUmVjb2duaXRpb24gTW9kZSBDb25zdHJhaW50IEZpbHRlciAtLT4KICAgICAgICAgIDxkaXYgY2xhc3M9Im1vZGUtZmlsdGVyLXNlY3Rpb24iPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJtb2RlLWxhYmVsLXJvdyI+CiAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InNlY3Rpb24tbGFiZWwiPlJlY29nbml0aW9uIE1vZGU6PC9zcGFuPgogICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdWJ0bGUtY2hpcCIgaWQ9ImFjdGl2ZU1vZGVMYWJlbCI+QXV0byBDb250ZXh0IEFjdGl2ZTwvc3Bhbj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDxkaXYgY2xhc3M9Im1vZGUtcGlsbHMiIGlkPSJtb2RlUGlsbHNDb250YWluZXIiPgogICAgICAgICAgICAgIDxidXR0b24gdHlwZT0iYnV0dG9uIiBjbGFzcz0ibW9kZS1waWxsLWJ0biBhY3RpdmUiIGRhdGEtbW9kZT0iYXV0byI+CiAgICAgICAgICAgICAgICA8c3Bhbj7wn46vIEF1dG8gQ29udGV4dDwvc3Bhbj4KICAgICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgICAgICA8YnV0dG9uIHR5cGU9ImJ1dHRvbiIgY2xhc3M9Im1vZGUtcGlsbC1idG4iIGRhdGEtbW9kZT0iZGlnaXRzIj4KICAgICAgICAgICAgICAgIDxzcGFuPvCflKIgRGlnaXRzIE9ubHkgKDAtOSk8L3NwYW4+CiAgICAgICAgICAgICAgPC9idXR0b24+CiAgICAgICAgICAgICAgPGJ1dHRvbiB0eXBlPSJidXR0b24iIGNsYXNzPSJtb2RlLXBpbGwtYnRuIiBkYXRhLW1vZGU9ImxldHRlcnMiPgogICAgICAgICAgICAgICAgPHNwYW4+8J+UpCBMZXR0ZXJzIE9ubHkgKEEtWik8L3NwYW4+CiAgICAgICAgICAgICAgPC9idXR0b24+CiAgICAgICAgICAgICAgPGJ1dHRvbiB0eXBlPSJidXR0b24iIGNsYXNzPSJtb2RlLXBpbGwtYnRuIiBkYXRhLW1vZGU9ImFscGhhbnVtZXJpYyI+CiAgICAgICAgICAgICAgICA8c3Bhbj7wn5SgIE1peGVkICgwLTksIEEtWik8L3NwYW4+CiAgICAgICAgICAgICAgPC9idXR0b24+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8cCBjbGFzcz0ibW9kZS1oaW50IiBpZD0ibW9kZUhpbnRUZXh0Ij5TbWFydCBDb250ZXh0OiBBdXRvLWRldGVjdHMgd2hldGhlciB5b3VyIHRleHQgaXMgbnVtZXJpYyBvciB3b3Jkcy48L3A+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8IS0tIFF1aWNrIFRlc3QgU2FtcGxlcyAtLT4KICAgICAgICAgIDxkaXYgY2xhc3M9InF1aWNrLXNhbXBsZXMiPgogICAgICAgICAgICA8c3BhbiBjbGFzcz0ic2VjdGlvbi1sYWJlbCI+UXVpY2sgVGVzdCBTYW1wbGVzOjwvc3Bhbj4KICAgICAgICAgICAgPGRpdiBjbGFzcz0ic2FtcGxlcy1idXR0b25zIj4KICAgICAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzYW1wbGUtcGlsbC1idG4iIGRhdGEtc2FtcGxlPSI5OTU0NjEiPkRpZ2l0czogIjk5NTQ2MSI8L2J1dHRvbj4KICAgICAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzYW1wbGUtcGlsbC1idG4iIGRhdGEtc2FtcGxlPSJISSBCUk8iPk5vdGU6ICJISSBCUk8iPC9idXR0b24+CiAgICAgICAgICAgICAgPGJ1dHRvbiBjbGFzcz0ic2FtcGxlLXBpbGwtYnRuIiBkYXRhLXNhbXBsZT0iSEVMTE8iPk5vdGU6ICJIRUxMTyI8L2J1dHRvbj4KICAgICAgICAgICAgICA8YnV0dG9uIGNsYXNzPSJzYW1wbGUtcGlsbC1idG4iIGRhdGEtc2FtcGxlPSJDT0RFIj5Ob3RlOiAiQ09ERSI8L2J1dHRvbj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8ZGl2IGNsYXNzPSJjYXJkLWZvb3RlciI+CiAgICAgICAgICAgIDxidXR0b24gaWQ9InByb2Nlc3NOb3RlQnRuIiBjbGFzcz0iYnRuIGJ0bi1wcmltYXJ5IGJ0bi1mdWxsIHJpcHBsZS1lZmZlY3QiIGRpc2FibGVkPgogICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjE4IiBoZWlnaHQ9IjE4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBwb2ludHM9IjUgMyAxOSAxMiA1IDIxIDUgMyIvPgogICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICAgIDxzcGFuPlJlY29nbml6ZSBIYW5kd3JpdHRlbiBOb3RlPC9zcGFuPgogICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgIDwvZGl2PgogICAgICAgIDwvZGl2PgoKICAgICAgICA8IS0tIFRyYW5zY3JpcHRpb24gUmVzdWx0cyBDYXJkIC0tPgogICAgICAgIDxkaXYgY2xhc3M9ImNhcmQgY2xlYW4tY2FyZCBhbmltYXRlLWNhcmQiPgogICAgICAgICAgPGRpdiBjbGFzcz0iY2FyZC1oZWFkZXIiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJjYXJkLXRpdGxlLXJvdyI+CiAgICAgICAgICAgICAgPGgyIGNsYXNzPSJjYXJkLXRpdGxlIj5UcmFuc2NyaXB0aW9uIFJlc3VsdHM8L2gyPgogICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJwaWxsLWJhZGdlIHBpbGwtZ3JlZW4iIGlkPSJyZXN1bHRTdGF0dXNCYWRnZSI+RW5naW5lIElkbGU8L3NwYW4+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8cCBjbGFzcz0iY2FyZC1kZXNjIj5Bc3NlbWJsZWQgdGV4dCwgY29uZmlkZW5jZSBkaXN0cmlidXRpb24sIGFuZCB2aXN1YWwgY2hhcmFjdGVyIHNlZ21lbnRhdGlvbi48L3A+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8IS0tIEVtcHR5IFN0YXRlIC0tPgogICAgICAgICAgPGRpdiBpZD0ic2VxdWVuY2VFbXB0eVN0YXRlIiBjbGFzcz0iZW1wdHktc3RhdGUiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJlbXB0eS1pY29uIGZsb2F0LXNvZnQiPgogICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjQ4IiBoZWlnaHQ9IjQ4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTQgMkg2YTIgMiAwIDAgMC0yIDJ2MTZhMiAyIDAgMCAwIDIgMmgxMmEyIDIgMCAwIDAgMi0yVjh6Ii8+CiAgICAgICAgICAgICAgICA8cG9seWxpbmUgcG9pbnRzPSIxNCAyIDE0IDggMjAgOCIvPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjEyIiB5MT0iMTIiIHgyPSIxMiIgeTI9IjE2Ii8+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iMTAiIHkxPSIxNCIgeDI9IjE0IiB5Mj0iMTQiLz4KICAgICAgICAgICAgICA8L3N2Zz4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDxwIGNsYXNzPSJlbXB0eS10aXRsZSI+UmVhZHkgZm9yIEhhbmR3cml0aW5nIElucHV0PC9wPgogICAgICAgICAgICA8cCBjbGFzcz0iZW1wdHktc3VidGl0bGUiPlVwbG9hZCBhIHBob3RvIG9yIHNlbGVjdCBhIHF1aWNrIHNhbXBsZSB0byB0cmFuc2NyaWJlIGhhbmR3cml0dGVuIGNoYXJhY3RlcnMuPC9wPgogICAgICAgICAgPC9kaXY+CgogICAgICAgICAgPCEtLSBMb2FkaW5nIFN0YXRlIC0tPgogICAgICAgICAgPGRpdiBpZD0ic2VxdWVuY2VMb2FkaW5nIiBjbGFzcz0ibG9hZGluZy1zdGF0ZSBoaWRkZW4iPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJsb2FkaW5nLXZpc3VhbCI+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0ic3Bpbm5lci1yaW5nIj48L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJwdWxzZS1jb3JlIj48L2Rpdj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDxwIGNsYXNzPSJsb2FkaW5nLXRpdGxlIj5FeHRyYWN0aW5nICYgU2VnbWVudGluZyBOb3RlLi4uPC9wPgogICAgICAgICAgICA8cCBjbGFzcz0ibG9hZGluZy1zdWJ0aXRsZSI+QXBwbHlpbmcgbW9ycGhvbG9naWNhbCBkaXZpc2lvbiwgY29udG91ciBhbmFseXNpcyAmIHBlcnNvbmFsaXplZCBDTk48L3A+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8IS0tIFBvcHVsYXRlZCBSZXN1bHRzIC0tPgogICAgICAgICAgPGRpdiBpZD0ic2VxdWVuY2VSZXN1bHRzIiBjbGFzcz0icmVzdWx0cy1jb250YWluZXIgaGlkZGVuIj4KICAgICAgICAgICAgPCEtLSBUZXh0IEJveCAtLT4KICAgICAgICAgICAgPGRpdiBjbGFzcz0idHJhbnNjcmlwdGlvbi1ib3ggaGlnaGxpZ2h0LXNoZWVuIj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJ0cmFuc2NyaXB0aW9uLWhlYWRlciI+CiAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJsYWJlbC13aXRoLWljb24iPgogICAgICAgICAgICAgICAgICA8c3ZnIHdpZHRoPSIxNSIgaGVpZ2h0PSIxNSIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj4KICAgICAgICAgICAgICAgICAgICA8cG9seWxpbmUgcG9pbnRzPSI0IDcgNCA0IDIwIDQgMjAgNyIvPgogICAgICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI5IiB5MT0iMjAiIHgyPSIxNSIgeTI9IjIwIi8+CiAgICAgICAgICAgICAgICAgICAgPGxpbmUgeDE9IjEyIiB5MT0iNCIgeDI9IjEyIiB5Mj0iMjAiLz4KICAgICAgICAgICAgICAgICAgPC9zdmc+CiAgICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJ0cmFuc2NyaXB0aW9uLWxhYmVsIj5SZWNvZ25pemVkIE91dHB1dDwvc3Bhbj4KICAgICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgICAgPGRpdiBjbGFzcz0iYWN0aW9uLWJ1dHRvbnMiPgogICAgICAgICAgICAgICAgICA8YnV0dG9uIGlkPSJjb3B5VGV4dEJ0biIgY2xhc3M9ImJ0bi1pY29uIHRvb2x0aXAtYW5jaG9yIiB0aXRsZT0iQ29weSB0ZXh0IHRvIGNsaXBib2FyZCI+CiAgICAgICAgICAgICAgICAgICAgPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+CiAgICAgICAgICAgICAgICAgICAgICA8cmVjdCB4PSI5IiB5PSI5IiB3aWR0aD0iMTMiIGhlaWdodD0iMTMiIHJ4PSIyIiByeT0iMiI+PC9yZWN0PgogICAgICAgICAgICAgICAgICAgICAgPHBhdGggZD0iTTUgMTVINGEyIDIgMCAwIDEtMi0yVjRhMiAyIDAgMCAxIDItMmg5YTIgMiAwIDAgMSAyIDJ2MSI+PC9wYXRoPgogICAgICAgICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICAgICAgICAgICAgPGJ1dHRvbiBpZD0iZG93bmxvYWRUZXh0QnRuIiBjbGFzcz0iYnRuLWljb24gdG9vbHRpcC1hbmNob3IiIHRpdGxlPSJEb3dubG9hZCB0cmFuc2NyaWJlZCB0ZXh0Ij4KICAgICAgICAgICAgICAgICAgICA8c3ZnIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj4KICAgICAgICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMSAxNXY0YTIgMiAwIDAgMS0yIDJINWEyIDIgMCAwIDEtMi0ydi00Ij48L3BhdGg+CiAgICAgICAgICAgICAgICAgICAgICA8cG9seWxpbmUgcG9pbnRzPSI3IDEwIDEyIDE1IDE3IDEwIj48L3BvbHlsaW5lPgogICAgICAgICAgICAgICAgICAgICAgPGxpbmUgeDE9IjEyIiB5MT0iMTUiIHgyPSIxMiIgeTI9IjMiPjwvbGluZT4KICAgICAgICAgICAgICAgICAgICA8L3N2Zz4KICAgICAgICAgICAgICAgICAgPC9idXR0b24+CiAgICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8dGV4dGFyZWEgaWQ9InJlY29nbml6ZWRPdXRwdXRUZXh0IiBjbGFzcz0idHJhbnNjcmlwdGlvbi10ZXh0YXJlYSIgcm93cz0iMiIgcmVhZG9ubHk+PC90ZXh0YXJlYT4KICAgICAgICAgICAgPC9kaXY+CgogICAgICAgICAgICA8IS0tIFN0YXRzIFJvdyAtLT4KICAgICAgICAgICAgPGRpdiBjbGFzcz0ic3RhdHMtcm93Ij4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJzdGF0LXBpbGwiPgogICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InN0YXQtbnVtIGNvdW50ZXItYW5pbSIgaWQ9InN0YXRUb3RhbENoYXJzIj4wPC9zcGFuPgogICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InN0YXQtbmFtZSI+Q2hhcmFjdGVycyBEZXRlY3RlZDwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJzdGF0LXBpbGwiPgogICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InN0YXQtbnVtIGNvdW50ZXItYW5pbSIgaWQ9InN0YXRBdmdDb25maWRlbmNlIj4wJTwvc3Bhbj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdGF0LW5hbWUiPkF2ZXJhZ2UgQ29uZmlkZW5jZTwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPC9kaXY+CgogICAgICAgICAgICA8IS0tIENoYXJhY3RlciBQaWxscyBHcmlkIC0tPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJjaGFyYWN0ZXJzLWJyZWFrZG93biI+CiAgICAgICAgICAgICAgPGgzIGNsYXNzPSJicmVha2Rvd24tdGl0bGUiPkNoYXJhY3RlciBDb25maWRlbmNlIEJyZWFrZG93bjo8L2gzPgogICAgICAgICAgICAgIDxkaXYgaWQ9ImNoYXJQaWxsc0dyaWQiIGNsYXNzPSJjaGFyLXBpbGxzLWdyaWQiPjwvZGl2PgogICAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICAgIDwhLS0gQm91bmRpbmcgQm94IEltYWdlIFByZXZpZXcgLS0+CiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImFubm90YXRlZC1wcmV2aWV3LXdyYXBwZXIiPgogICAgICAgICAgICAgIDxkaXYgY2xhc3M9ImFubm90YXRlZC10aXRsZS1yb3ciPgogICAgICAgICAgICAgICAgPGgzIGNsYXNzPSJicmVha2Rvd24tdGl0bGUiPlNlZ21lbnRhdGlvbiAmIEJvdW5kaW5nIEJveCBPdmVybGF5OjwvaDM+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3VidGxlLWJhZGdlIj5PcGVuQ1YgRG9jdW1lbnQgUk9JPC9zcGFuPgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgIDxkaXYgY2xhc3M9ImFubm90YXRlZC1pbWctZnJhbWUiPgogICAgICAgICAgICAgICAgPGltZyBpZD0iYW5ub3RhdGVkSW1hZ2UiIHNyYz0iIiBhbHQ9IlNlZ21lbnRlZCBib3VuZGluZyBib3hlcyIgY2xhc3M9ImFubm90YXRlZC1pbWciPgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgIDwvZGl2PgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvc2VjdGlvbj4KCiAgICA8IS0tIE1PREUgMjogSW50ZXJhY3RpdmUgRHJhd2luZyBQYWQgLS0+CiAgICA8c2VjdGlvbiBpZD0iY2FudmFzTW9kZSIgY2xhc3M9InRhYi12aWV3IGhpZGRlbiI+CiAgICAgIDxkaXYgY2xhc3M9InZpZXctZ3JpZCI+CiAgICAgICAgPCEtLSBDYW52YXMgUGFkIC0tPgogICAgICAgIDxkaXYgY2xhc3M9ImNhcmQgY2xlYW4tY2FyZCBhbmltYXRlLWNhcmQiPgogICAgICAgICAgPGRpdiBjbGFzcz0iY2FyZC1oZWFkZXIiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJjYXJkLXRpdGxlLXJvdyI+CiAgICAgICAgICAgICAgPGgyIGNsYXNzPSJjYXJkLXRpdGxlIj5MaXZlIERyYXdpbmcgJiBDaGFyYWN0ZXIgUGFkPC9oMj4KICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0icGlsbC1iYWRnZSBwaWxsLXB1cnBsZSI+TGl2ZSBQYWQ8L3NwYW4+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8cCBjbGFzcz0iY2FyZC1kZXNjIj5Xcml0ZSBhbnkgbGV0dGVyLCBkaWdpdCwgb3IgaGFuZHdyaXR0ZW4gd29yZCAoZS5nLiAiSGkiLCAiQSIsICI0MiIsICJBSSIpIGluIHJlYWwgdGltZS48L3A+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8ZGl2IGNsYXNzPSJjYW52YXMtd3JhcHBlciI+CiAgICAgICAgICAgIDxjYW52YXMgaWQ9ImRyYXdpbmdDYW52YXMiIHdpZHRoPSIyODAiIGhlaWdodD0iMjgwIj48L2NhbnZhcz4KICAgICAgICAgICAgPGRpdiBjbGFzcz0iY2FudmFzLWhpbnQiPkRyYXcgc2luZ2xlIGxldHRlcnMgb3Igd29yZHMgaW5zaWRlIHRoZSBjYW52YXM8L2Rpdj4KICAgICAgICAgIDwvZGl2PgoKICAgICAgICAgIDxkaXYgY2xhc3M9ImNhbnZhcy10b29sYmFyIj4KICAgICAgICAgICAgPGRpdiBjbGFzcz0idG9vbGJhci1ncm91cCI+CiAgICAgICAgICAgICAgPGxhYmVsIGZvcj0icGVuU2l6ZVNsaWRlciIgY2xhc3M9InRvb2xiYXItbGFiZWwiPlBlbiBTdHJva2U6PC9sYWJlbD4KICAgICAgICAgICAgICA8aW5wdXQgdHlwZT0icmFuZ2UiIGlkPSJwZW5TaXplU2xpZGVyIiBtaW49IjEyIiBtYXg9IjMyIiB2YWx1ZT0iMjAiIGNsYXNzPSJyYW5nZS1zbGlkZXIiPgogICAgICAgICAgICAgIDxzcGFuIGlkPSJwZW5TaXplRGlzcGxheSIgY2xhc3M9InJhbmdlLXZhbCI+MjBweDwvc3Bhbj4KICAgICAgICAgICAgPC9kaXY+CgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJ0b29sYmFyLWJ1dHRvbnMiPgogICAgICAgICAgICAgIDxidXR0b24gaWQ9ImNsZWFyQ2FudmFzQnRuIiBjbGFzcz0iYnRuIGJ0bi1zZWNvbmRhcnkgYnRuLXNtIj4KICAgICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjE0IiBoZWlnaHQ9IjE0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPgogICAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMyA2aDE4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xOSA2djE0YzAgMS0xIDItMiAySDdjLTEgMC0yLTEtMi0yVjYiPjwvcGF0aD4KICAgICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICAgICAgQ2xlYXIgUGFkCiAgICAgICAgICAgICAgPC9idXR0b24+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgPC9kaXY+CgogICAgICAgICAgPCEtLSBRdWljayBUZXN0IFN1Z2dlc3Rpb25zIC0tPgogICAgICAgICAgPGRpdiBjbGFzcz0icXVpY2stc2FtcGxlcyIgc3R5bGU9Im1hcmdpbi10b3A6IDZweDsgbWFyZ2luLWJvdHRvbTogMTJweDsiPgogICAgICAgICAgICA8c3BhbiBjbGFzcz0ic2FtcGxlcy10aXRsZSIgc3R5bGU9ImZvbnQtc2l6ZTogMC43NnJlbTsgY29sb3I6IHZhcigtLXRleHQtbXV0ZWQpOyBmb250LXdlaWdodDogNjAwOyI+RFJBV0lORyBFWEFNUExFUzo8L3NwYW4+CiAgICAgICAgICAgIDxkaXYgY2xhc3M9InNhbXBsZXMtYnV0dG9ucyI+CiAgICAgICAgICAgICAgPGJ1dHRvbiB0eXBlPSJidXR0b24iIGNsYXNzPSJzYW1wbGUtcGlsbC1idG4gY2FudmFzLXF1aWNrLWJ0biIgZGF0YS10ZXh0PSJIaSI+V29yZDogIkhpIjwvYnV0dG9uPgogICAgICAgICAgICAgIDxidXR0b24gdHlwZT0iYnV0dG9uIiBjbGFzcz0ic2FtcGxlLXBpbGwtYnRuIGNhbnZhcy1xdWljay1idG4iIGRhdGEtdGV4dD0iQUkiPldvcmQ6ICJBSSI8L2J1dHRvbj4KICAgICAgICAgICAgICA8YnV0dG9uIHR5cGU9ImJ1dHRvbiIgY2xhc3M9InNhbXBsZS1waWxsLWJ0biBjYW52YXMtcXVpY2stYnRuIiBkYXRhLXRleHQ9IkEiPkxldHRlcjogIkEiPC9idXR0b24+CiAgICAgICAgICAgICAgPGJ1dHRvbiB0eXBlPSJidXR0b24iIGNsYXNzPSJzYW1wbGUtcGlsbC1idG4gY2FudmFzLXF1aWNrLWJ0biIgZGF0YS10ZXh0PSI0MiI+TnVtYmVyOiAiNDIiPC9idXR0b24+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgPC9kaXY+CgogICAgICAgICAgPGRpdiBjbGFzcz0iY2FyZC1mb290ZXIiIHN0eWxlPSJtYXJnaW4tdG9wOiA4cHg7Ij4KICAgICAgICAgICAgPGJ1dHRvbiBpZD0icHJlZGljdENhbnZhc0J0biIgY2xhc3M9ImJ0biBidG4tcHJpbWFyeSBidG4tZnVsbCByaXBwbGUtZWZmZWN0Ij4KICAgICAgICAgICAgICA8c3ZnIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj4KICAgICAgICAgICAgICAgIDxwb2x5bGluZSBwb2ludHM9IjIyIDEyIDE4IDEyIDE1IDIxIDkgMyA2IDEyIDIgMTIiLz4KICAgICAgICAgICAgICA8L3N2Zz4KICAgICAgICAgICAgICA8c3Bhbj5SZWNvZ25pemUgRHJhd2luZzwvc3Bhbj4KICAgICAgICAgICAgPC9idXR0b24+CiAgICAgICAgICA8L2Rpdj4KICAgICAgICA8L2Rpdj4KCiAgICAgICAgPCEtLSBDYW52YXMgUHJlZGljdGlvbnMgQ2FyZCAtLT4KICAgICAgICA8ZGl2IGNsYXNzPSJjYXJkIGNsZWFuLWNhcmQgYW5pbWF0ZS1jYXJkIj4KICAgICAgICAgIDxkaXYgY2xhc3M9ImNhcmQtaGVhZGVyIj4KICAgICAgICAgICAgPGRpdiBjbGFzcz0iY2FyZC10aXRsZS1yb3ciPgogICAgICAgICAgICAgIDxoMiBjbGFzcz0iY2FyZC10aXRsZSI+UmVjb2duaXRpb24gUmVzdWx0czwvaDI+CiAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InBpbGwtYmFkZ2UgcGlsbC1pbmRpZ28iPlNjcmF3bCBFbmdpbmU8L3NwYW4+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8cCBjbGFzcz0iY2FyZC1kZXNjIj5EZWVwIENOTiBwcmVkaWN0aW9uIHdpdGggbXVsdGktY2hhcmFjdGVyIHNlZ21lbnRhdGlvbiAmIGNvbmZpZGVuY2UgYnJlYWtkb3duLjwvcD4KICAgICAgICAgIDwvZGl2PgoKICAgICAgICAgIDwhLS0gRW1wdHkgU3RhdGUgLS0+CiAgICAgICAgICA8ZGl2IGlkPSJjYW52YXNFbXB0eVN0YXRlIiBjbGFzcz0iZW1wdHktc3RhdGUiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJlbXB0eS1pY29uIGZsb2F0LXNvZnQiPgogICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjQ0IiBoZWlnaHQ9IjQ0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjEuNSI+CiAgICAgICAgICAgICAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCI+PC9jaXJjbGU+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNOS4wOSA5YTMgMyAwIDAgMSA1LjgzIDFjMCAyLTMgMy0zIDMiPjwvcGF0aD4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIxMiIgeTE9IjE3IiB4Mj0iMTIuMDEiIHkyPSIxNyI+PC9saW5lPgogICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPHAgY2xhc3M9ImVtcHR5LXRpdGxlIj5DYW52YXMgUmVhZHk8L3A+CiAgICAgICAgICAgIDxwIGNsYXNzPSJlbXB0eS1zdWJ0aXRsZSI+RHJhdyB0ZXh0LCBjaGFyYWN0ZXJzLCBvciBudW1iZXJzIG9uIHRoZSBwYWQgYW5kIGNsaWNrIFJlY29nbml6ZSB0byBzZWUgdGhlIG1vZGVsIHByZWRpY3QuPC9wPgogICAgICAgICAgPC9kaXY+CgogICAgICAgICAgPCEtLSBSZXN1bHRzIC0tPgogICAgICAgICAgPGRpdiBpZD0iY2FudmFzUmVzdWx0cyIgY2xhc3M9InJlc3VsdHMtY29udGFpbmVyIGhpZGRlbiI+CiAgICAgICAgICAgIDxkaXYgY2xhc3M9Imhlcm8tY2hhcmFjdGVyLWJveCBoaWdobGlnaHQtc2hlZW4iPgogICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJoZXJvLWxhYmVsIj5SZWNvZ25pemVkIERyYXdpbmc8L3NwYW4+CiAgICAgICAgICAgICAgPGRpdiBpZD0iaGVyb0NoYXJhY3RlciIgY2xhc3M9Imhlcm8tY2hhcmFjdGVyIHNjYWxlLXBvcCI+QTwvZGl2PgogICAgICAgICAgICAgIDxkaXYgY2xhc3M9Imhlcm8tY29uZmlkZW5jZSI+CiAgICAgICAgICAgICAgICA8c3BhbiBpZD0iaGVyb0NvbmZpZGVuY2UiPjk5LjglPC9zcGFuPiBjb25maWRlbmNlCiAgICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDwvZGl2PgoKICAgICAgICAgICAgPCEtLSBNdWx0aS1jaGFyYWN0ZXIgY29uZmlkZW5jZSBicmVha2Rvd24gLS0+CiAgICAgICAgICAgIDxkaXYgaWQ9ImNhbnZhc0NoYXJCcmVha2Rvd24iIGNsYXNzPSJjaGFyYWN0ZXJzLWJyZWFrZG93biBoaWRkZW4iIHN0eWxlPSJtYXJnaW4tdG9wOiAxNHB4OyI+CiAgICAgICAgICAgICAgPGgzIGNsYXNzPSJicmVha2Rvd24tdGl0bGUiPkNoYXJhY3RlciBDb25maWRlbmNlIEJyZWFrZG93bjo8L2gzPgogICAgICAgICAgICAgIDxkaXYgaWQ9ImNhbnZhc0NoYXJQaWxscyIgY2xhc3M9ImNoYXItcGlsbHMtZ3JpZCI+PC9kaXY+CiAgICAgICAgICAgIDwvZGl2PgoKICAgICAgICAgICAgPCEtLSBTaW5nbGUtY2hhcmFjdGVyIHRvcCBjYW5kaWRhdGUgZGlzdHJpYnV0aW9uIC0tPgogICAgICAgICAgICA8ZGl2IGlkPSJjYW52YXNDYW5kaWRhdGVzQ29udGFpbmVyIiBjbGFzcz0icHJvYmFiaWxpdHktZGlzdHJpYnV0aW9uIiBzdHlsZT0ibWFyZ2luLXRvcDogMTRweDsiPgogICAgICAgICAgICAgIDxoMyBjbGFzcz0iYnJlYWtkb3duLXRpdGxlIj5Ub3AgQ2FuZGlkYXRlczo8L2gzPgogICAgICAgICAgICAgIDxkaXYgaWQ9ImNhbmRpZGF0ZXNMaXN0IiBjbGFzcz0iY2FuZGlkYXRlcy1saXN0Ij48L2Rpdj4KICAgICAgICAgICAgPC9kaXY+CgogICAgICAgICAgICA8IS0tIExpdmUgU2VnbWVudGF0aW9uIEJvdW5kaW5nIEJveCBQcmV2aWV3IC0tPgogICAgICAgICAgICA8ZGl2IGlkPSJjYW52YXNBbm5vdGF0ZWRXcmFwcGVyIiBjbGFzcz0iYW5ub3RhdGVkLXByZXZpZXctd3JhcHBlciBoaWRkZW4iIHN0eWxlPSJtYXJnaW4tdG9wOiAxNHB4OyI+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0iYW5ub3RhdGVkLXRpdGxlLXJvdyI+CiAgICAgICAgICAgICAgICA8aDMgY2xhc3M9ImJyZWFrZG93bi10aXRsZSI+U2VnbWVudGF0aW9uICYgQ2hhcmFjdGVyIEJveGVzOjwvaDM+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3VidGxlLWJhZGdlIj5MaXZlIFNlZ21lbnRhdGlvbjwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJhbm5vdGF0ZWQtaW1nLWZyYW1lIj4KICAgICAgICAgICAgICAgIDxpbWcgaWQ9ImNhbnZhc0Fubm90YXRlZEltYWdlIiBzcmM9IiIgYWx0PSJDYW52YXMgc2VnbWVudGF0aW9uIiBjbGFzcz0iYW5ub3RhdGVkLWltZyIgc3R5bGU9Im1heC1oZWlnaHQ6IDE4MHB4OyI+CiAgICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9zZWN0aW9uPgoKICAgIDwhLS0gTU9ERSAzOiBPd25lcnNoaXAgQ29udHJvbHMgJiBEZXZlbG9wZXIgUHJvZmlsZSAtLT4KICAgIDxzZWN0aW9uIGlkPSJvd25lclNlY3Rpb24iIGNsYXNzPSJ0YWItdmlldyBoaWRkZW4iPgogICAgICA8ZGl2IGNsYXNzPSJvd25lci1wYW5lbC1jb250YWluZXIgYW5pbWF0ZS1jYXJkIj4KICAgICAgICA8ZGl2IGNsYXNzPSJvd25lci1wYW5lbC1oZWFkZXIiPgogICAgICAgICAgPGRpdiBjbGFzcz0ib3duZXItYXZhdGFyLWxhcmdlIj5UUDwvZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0ib3duZXItaGVhZGVyLWluZm8iPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJvd25lci1uYW1lLXJvdyI+CiAgICAgICAgICAgICAgPGgyIGNsYXNzPSJvd25lci1mdWxsLW5hbWUiPlRyaWd1biBQYXJ2YXRoYXB1cmFtPC9oMj4KICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0idmVyaWZpZWQtcGlsbCI+CiAgICAgICAgICAgICAgICA8c3ZnIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJjdXJyZW50Q29sb3IiPgogICAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTIgMkM2LjQ4IDIgMiA2LjQ4IDIgMTJzNC40OCAxMCAxMCAxMCAxMC00LjQ4IDEwLTEwUzE3LjUyIDIgMTIgMnptLTIgMTVsLTUtNSAxLjQxLTEuNDFMMTAgMTQuMTdsNy41OS03LjU5TDE5IDhsLTkgOXoiLz4KICAgICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICAgICAgUHJvamVjdCBPd25lciAmIE1vZGVsIEFyY2hpdGVjdAogICAgICAgICAgICAgIDwvc3Bhbj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDxwIGNsYXNzPSJvd25lci1iaW8iPkxlYWQgRGV2ZWxvcGVyIOKAoiBDb2RlQWxwaGEgTWFjaGluZSBMZWFybmluZyBUcmFjayDigKIgVGFzayAzOiBIYW5kd3JpdHRlbiBDaGFyYWN0ZXIgUmVjb2duaXRpb248L3A+CiAgICAgICAgICA8L2Rpdj4KICAgICAgICA8L2Rpdj4KCiAgICAgICAgPGRpdiBjbGFzcz0ib3duZXItZ3JpZCI+CiAgICAgICAgICA8IS0tIENvbnRyb2wgMTogTW9kZWwgUGVyc29uYWxpemF0aW9uIFN0YXR1cyAtLT4KICAgICAgICAgIDxkaXYgY2xhc3M9Im93bmVyLWNhcmQiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJvd25lci1jYXJkLXRpdGxlLXJvdyI+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0ib3duZXItaWNvbi1ib3giPgogICAgICAgICAgICAgICAgPHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+CiAgICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMCAyMXYtMmE0IDQgMCAwIDAtNC00SDhhNCA0IDAgMCAwLTQgNHYyIj48L3BhdGg+CiAgICAgICAgICAgICAgICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iNyIgcj0iNCI+PC9jaXJjbGU+CiAgICAgICAgICAgICAgICA8L3N2Zz4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8aDMgY2xhc3M9Im93bmVyLWNhcmQtdGl0bGUiPkhhbmR3cml0aW5nIFBlcnNvbmFsaXphdGlvbjwvaDM+CiAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InRvZ2dsZS1zdGF0dXMgYWN0aXZlIiBpZD0icGVyc29uYWxpemF0aW9uU3RhdHVzVGV4dCI+QWN0aXZlPC9zcGFuPgogICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPHAgY2xhc3M9Im93bmVyLWNhcmQtZGVzYyI+VGhlIENOTiBtb2RlbCBoYXMgYmVlbiBmaW5lLXR1bmVkIGRpcmVjdGx5IG9uIFRyaWd1biBQYXJ2YXRoYXB1cmFtJ3MgcGVyc29uYWwgaGFuZHdyaXRpbmcgc2FtcGxlIHNoZWV0IChB4oCTWiwgYeKAk3osIDDigJM5KS48L3A+CiAgICAgICAgICAgIDxkaXYgY2xhc3M9Im93bmVyLW1ldHJpYy1ib3giPgogICAgICAgICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pdGVtIj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJtZXRyaWMtbGFiZWwiPlZhbGlkYXRpb24gQWNjdXJhY3k8L3NwYW4+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ibWV0cmljLXZhbCB0ZXh0LXN1Y2Nlc3MiPjk2LjMyJTwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJtZXRyaWMtaXRlbSI+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ibWV0cmljLWxhYmVsIj5Vc2VyIEF1Z21lbnRlZCBTYW1wbGVzPC9zcGFuPgogICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9Im1ldHJpYy12YWwiPjUsMzMyPC9zcGFuPgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgIDxkaXYgY2xhc3M9Im1ldHJpYy1pdGVtIj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJtZXRyaWMtbGFiZWwiPkNsYXNzZXMgQ292ZXJlZDwvc3Bhbj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJtZXRyaWMtdmFsIj4zNiAoMOKAkzksIEHigJNaKTwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8IS0tIENvbnRyb2wgMjogT3duZXJzaGlwIENvbnRyb2xzICYgU2lnbmF0dXJlIFdhdGVybWFya2luZyAtLT4KICAgICAgICAgIDxkaXYgY2xhc3M9Im93bmVyLWNhcmQiPgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJvd25lci1jYXJkLXRpdGxlLXJvdyI+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0ib3duZXItaWNvbi1ib3giPgogICAgICAgICAgICAgICAgPHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiI+CiAgICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMiAyMnM4LTQgOC0xMFY1bC04LTMtOCAzdjdjMCA2IDggMTAgOCAxMHoiPjwvcGF0aD4KICAgICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgIDxoMyBjbGFzcz0ib3duZXItY2FyZC10aXRsZSI+T3duZXJzaGlwIEF0dHJpYnV0aW9uPC9oMz4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgIDxwIGNsYXNzPSJvd25lci1jYXJkLWRlc2MiPkNvbmZpZ3VyZSBvd25lcnNoaXAgY29udHJvbHMgYW5kIHdhdGVybWFyayBleHBvcnQgb3B0aW9ucy48L3A+CgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJvd25lci10b2dnbGVzLWxpc3QiPgogICAgICAgICAgICAgIDxsYWJlbCBjbGFzcz0idG9nZ2xlLXJvdyI+CiAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJ0b2dnbGUtaW5mbyI+CiAgICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJ0b2dnbGUtbmFtZSI+U2lnbmF0dXJlIFdhdGVybWFyayBvbiBFeHBvcnRzPC9zcGFuPgogICAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0idG9nZ2xlLXN1YiI+QXBwZW5kcyAiVHJhbnNjcmliZWQgd2l0aCBTY3Jhd2wgYnkgVHJpZ3VuIFBhcnZhdGhhcHVyYW0iPC9zcGFuPgogICAgICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgICAgICA8aW5wdXQgdHlwZT0iY2hlY2tib3giIGlkPSJ0b2dnbGVXYXRlcm1hcmsiIGNsYXNzPSJjdXN0b20tdG9nZ2xlIiBjaGVja2VkPgogICAgICAgICAgICAgIDwvbGFiZWw+CgogICAgICAgICAgICAgIDxsYWJlbCBjbGFzcz0idG9nZ2xlLXJvdyI+CiAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJ0b2dnbGUtaW5mbyI+CiAgICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJ0b2dnbGUtbmFtZSI+T3duZXIgU2lnbmF0dXJlIENhcmQ8L3NwYW4+CiAgICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJ0b2dnbGUtc3ViIj5EaXNwbGF5IFRyaWd1bidzIHZlcmlmaWVkIGJhZGdlIG9uIHJlc3VsdHM8L3NwYW4+CiAgICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICAgIDxpbnB1dCB0eXBlPSJjaGVja2JveCIgaWQ9InRvZ2dsZU93bmVyQmFkZ2UiIGNsYXNzPSJjdXN0b20tdG9nZ2xlIiBjaGVja2VkPgogICAgICAgICAgICAgIDwvbGFiZWw+CgogICAgICAgICAgICAgIDxsYWJlbCBjbGFzcz0idG9nZ2xlLXJvdyI+CiAgICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJ0b2dnbGUtaW5mbyI+CiAgICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJ0b2dnbGUtbmFtZSI+TW9ycGhvbG9naWNhbCBJbGx1bWluYXRpb24gRmxhdHRlbmluZzwvc3Bhbj4KICAgICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InRvZ2dsZS1zdWIiPk5vcm1hbGl6ZXMgc2hhZG93cyBhbmQgcGFwZXIgYmFja2dyb3VuZCB2YXJpYXRpb25zPC9zcGFuPgogICAgICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICAgICAgICA8aW5wdXQgdHlwZT0iY2hlY2tib3giIGlkPSJ0b2dnbGVNb3JwaCIgY2xhc3M9ImN1c3RvbS10b2dnbGUiIGNoZWNrZWQgZGlzYWJsZWQ+CiAgICAgICAgICAgICAgPC9sYWJlbD4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICA8L2Rpdj4KCiAgICAgICAgICA8IS0tIENvbnRyb2wgMzogU3lzdGVtIEFyY2hpdGVjdHVyZSAtLT4KICAgICAgICAgIDxkaXYgY2xhc3M9Im93bmVyLWNhcmQgb3duZXItY2FyZC13aWRlIj4KICAgICAgICAgICAgPGRpdiBjbGFzcz0ib3duZXItY2FyZC10aXRsZS1yb3ciPgogICAgICAgICAgICAgIDxkaXYgY2xhc3M9Im93bmVyLWljb24tYm94Ij4KICAgICAgICAgICAgICAgIDxzdmcgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiPgogICAgICAgICAgICAgICAgICA8cmVjdCB4PSIyIiB5PSIzIiB3aWR0aD0iMjAiIGhlaWdodD0iMTQiIHJ4PSIyIiByeT0iMiI+PC9yZWN0PgogICAgICAgICAgICAgICAgICA8bGluZSB4MT0iOCIgeTE9IjIxIiB4Mj0iMTYiIHkyPSIyMSI+PC9saW5lPgogICAgICAgICAgICAgICAgICA8bGluZSB4MT0iMTIiIHkxPSIxNyIgeDI9IjEyIiB5Mj0iMjEiPjwvbGluZT4KICAgICAgICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgIDxoMyBjbGFzcz0ib3duZXItY2FyZC10aXRsZSI+VGVjaG5pY2FsIFBpcGVsaW5lIEFyY2hpdGVjdHVyZTwvaDM+CiAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICA8ZGl2IGNsYXNzPSJwaXBlbGluZS1mbG93Ij4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJwaXBlbGluZS1zdGVwIj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdGVwLW51bSI+MTwvc3Bhbj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdGVwLXRpdGxlIj5Eb2N1bWVudCBST0k8L3NwYW4+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC1zdWIiPlBhcGVyIGNvbnRvdXIgaXNvbGF0aW9uPC9zcGFuPgogICAgICAgICAgICAgIDwvZGl2PgogICAgICAgICAgICAgIDxkaXYgY2xhc3M9InBpcGVsaW5lLWFycm93Ij7ihpI8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJwaXBlbGluZS1zdGVwIj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdGVwLW51bSI+Mjwvc3Bhbj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdGVwLXRpdGxlIj5Nb3JwaG9sb2dpY2FsIERpdmlzaW9uPC9zcGFuPgogICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InN0ZXAtc3ViIj5JbGx1bWluYXRpb24gZmxhdHRlbmluZzwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJwaXBlbGluZS1hcnJvdyI+4oaSPC9kaXY+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0icGlwZWxpbmUtc3RlcCI+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC1udW0iPjM8L3NwYW4+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC10aXRsZSI+U3Ryb2tlIENsdXN0ZXJpbmc8L3NwYW4+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC1zdWIiPkxpbmUgJiB3b3JkIHNlZ21lbnRhdGlvbjwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJwaXBlbGluZS1hcnJvdyI+4oaSPC9kaXY+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0icGlwZWxpbmUtc3RlcCI+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC1udW0iPjQ8L3NwYW4+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC10aXRsZSI+VHJpZ3VuIEZpbmUtVHVuZWQgQ05OPC9zcGFuPgogICAgICAgICAgICAgICAgPHNwYW4gY2xhc3M9InN0ZXAtc3ViIj4zNi1DbGFzcyBpbmZlcmVuY2UgKDk2LjMlKTwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgICA8ZGl2IGNsYXNzPSJwaXBlbGluZS1hcnJvdyI+4oaSPC9kaXY+CiAgICAgICAgICAgICAgPGRpdiBjbGFzcz0icGlwZWxpbmUtc3RlcCI+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC1udW0iPjU8L3NwYW4+CiAgICAgICAgICAgICAgICA8c3BhbiBjbGFzcz0ic3RlcC10aXRsZSI+Q29udGV4dCBEaXNhbWJpZ3VhdGlvbjwvc3Bhbj4KICAgICAgICAgICAgICAgIDxzcGFuIGNsYXNzPSJzdGVwLXN1YiI+TnVtZXJpYyB2cyB3b3JkIGZpbHRlcmluZzwvc3Bhbj4KICAgICAgICAgICAgICA8L2Rpdj4KICAgICAgICAgICAgPC9kaXY+CiAgICAgICAgICA8L2Rpdj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L3NlY3Rpb24+CiAgPC9tYWluPgoKICA8IS0tIE93bmVyc2hpcCBNb2RhbCAoYWNjZXNzaWJsZSBmcm9tIGhlYWRlciBidXR0b24pIC0tPgogIDxkaXYgaWQ9Im93bmVyTW9kYWwiIGNsYXNzPSJtb2RhbC1iYWNrZHJvcCBoaWRkZW4iPgogICAgPGRpdiBjbGFzcz0ibW9kYWwtY2FyZCI+CiAgICAgIDxidXR0b24gaWQ9ImNsb3NlT3duZXJNb2RhbEJ0biIgY2xhc3M9Im1vZGFsLWNsb3NlLWJ0biIgdGl0bGU9IkNsb3NlIj4KICAgICAgICA8c3ZnIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIj4KICAgICAgICAgIDxsaW5lIHgxPSIxOCIgeTE9IjYiIHgyPSI2IiB5Mj0iMTgiPjwvbGluZT4KICAgICAgICAgIDxsaW5lIHgxPSI2IiB5MT0iNiIgeDI9IjE4IiB5Mj0iMTgiPjwvbGluZT4KICAgICAgICA8L3N2Zz4KICAgICAgPC9idXR0b24+CgogICAgICA8ZGl2IGNsYXNzPSJtb2RhbC1oZWFkZXIiPgogICAgICAgIDxkaXYgY2xhc3M9Im93bmVyLWF2YXRhci1tb2RhbCI+VFA8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtb2RhbC10aXRsZS13cmFwIj4KICAgICAgICAgIDxoMyBjbGFzcz0ibW9kYWwtdGl0bGUiPlRyaWd1biBQYXJ2YXRoYXB1cmFtPC9oMz4KICAgICAgICAgIDxwIGNsYXNzPSJtb2RhbC1zdWJ0aXRsZSI+UHJvamVjdCBPd25lciAmIE1hY2hpbmUgTGVhcm5pbmcgRW5naW5lZXI8L3A+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgoKICAgICAgPGRpdiBjbGFzcz0ibW9kYWwtYm9keSI+CiAgICAgICAgPGRpdiBjbGFzcz0ibW9kYWwtaW5mby1yb3ciPgogICAgICAgICAgPHNwYW4gY2xhc3M9ImluZm8tdGFnIj5Qcm9qZWN0PC9zcGFuPgogICAgICAgICAgPHNwYW4gY2xhc3M9ImluZm8tdmFsIj48c3Ryb25nPlNjcmF3bDwvc3Ryb25nPiDigJQgTmV1cmFsIEhhbmR3cml0aW5nIE9DUjwvc3Bhbj4KICAgICAgICA8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtb2RhbC1pbmZvLXJvdyI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0iaW5mby10YWciPlRyYWNrPC9zcGFuPgogICAgICAgICAgPHNwYW4gY2xhc3M9ImluZm8tdmFsIj5Db2RlQWxwaGEgTUwgSW50ZXJuc2hpcCDigJQgVGFzayAzPC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1vZGFsLWluZm8tcm93Ij4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJpbmZvLXRhZyI+TW9kZWw8L3NwYW4+CiAgICAgICAgICA8c3BhbiBjbGFzcz0iaW5mby12YWwiPlBlcnNvbmFsaXplZCBDTk4gRmluZS1UdW5lZCBvbiBUcmlndW4ncyBIYW5kd3JpdGluZzwvc3Bhbj4KICAgICAgICA8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJtb2RhbC1pbmZvLXJvdyI+CiAgICAgICAgICA8c3BhbiBjbGFzcz0iaW5mby10YWciPk1vZGVsIEFjY3VyYWN5PC9zcGFuPgogICAgICAgICAgPHNwYW4gY2xhc3M9ImluZm8tdmFsIHRleHQtc3VjY2VzcyBmb250LWJvbGQiPjk2LjMyJSBWYWxpZGF0aW9uIEFjY3VyYWN5PC9zcGFuPgogICAgICAgIDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9Im1vZGFsLWluZm8tcm93Ij4KICAgICAgICAgIDxzcGFuIGNsYXNzPSJpbmZvLXRhZyI+VmVyaWZpZWQgSUQ8L3NwYW4+CiAgICAgICAgICA8c3BhbiBjbGFzcz0iaW5mby12YWwgZm9udC1tb25vIj5TQ1JBV0wtT1dORVItVFAtMjAyNi1PSzwvc3Bhbj4KICAgICAgICA8L2Rpdj4KCiAgICAgICAgPGRpdiBjbGFzcz0ibW9kYWwtYWN0aW9uLWJhciI+CiAgICAgICAgICA8YnV0dG9uIGlkPSJtb2RhbFN3aXRjaFRvT3duZXJUYWJCdG4iIGNsYXNzPSJidG4gYnRuLXByaW1hcnkgYnRuLWZ1bGwiPgogICAgICAgICAgICBPcGVuIEFkdmFuY2VkIE93bmVyIENvbnRyb2xzCiAgICAgICAgICA8L2J1dHRvbj4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICA8L2Rpdj4KCiAgPCEtLSBGb290ZXIgLS0+CiAgPGZvb3RlciBjbGFzcz0iYXBwLWZvb3RlciI+CiAgICA8ZGl2IGNsYXNzPSJmb290ZXItY29udGFpbmVyIj4KICAgICAgPGRpdiBjbGFzcz0iZm9vdGVyLWxlZnQiPgogICAgICAgIDxzcGFuIGNsYXNzPSJicmFuZC1mb290ZXIiPlNjcmF3bDxzcGFuIGNsYXNzPSJicmFuZC1kb3QiPi48L3NwYW4+PC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJmb290ZXItc2VwIj7igKI8L3NwYW4+CiAgICAgICAgPHNwYW4+RW5naW5lZXJlZCBieSA8c3Ryb25nPlRyaWd1biBQYXJ2YXRoYXB1cmFtPC9zdHJvbmc+PC9zcGFuPgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZm9vdGVyLXJpZ2h0Ij4KICAgICAgICA8c3BhbiBjbGFzcz0ib3duZXItdmVyaWZpZWQtcGlsbCIgdGl0bGU9IlZlcmlmaWVkIENyZWF0b3IgJiBNb2RlbCBBcmNoaXRlY3QiPgogICAgICAgICAgPHN2ZyB3aWR0aD0iMTMiIGhlaWdodD0iMTMiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMi41Ij4KICAgICAgICAgICAgPHBvbHlsaW5lIHBvaW50cz0iMjAgNiA5IDE3IDQgMTIiLz4KICAgICAgICAgIDwvc3ZnPgogICAgICAgICAgT3duZXI6IFRyaWd1biBQYXJ2YXRoYXB1cmFtCiAgICAgICAgPC9zcGFuPgogICAgICAgIDxzcGFuIGNsYXNzPSJjb3B5cmlnaHQtdGV4dCI+wqkgMjAyNiBUcmlndW4gUGFydmF0aGFwdXJhbS4gQWxsIHJpZ2h0cyByZXNlcnZlZC48L3NwYW4+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9mb290ZXI+CgogIDxzY3JpcHQgc3JjPSIvc3RhdGljL2FwcC5qcyI+PC9zY3JpcHQ+CjwvYm9keT4KPC9odG1sPg=="
EMBEDDED_HTML = base64.b64decode(EMBEDDED_HTML_B64).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=EMBEDDED_HTML)

import time
import subprocess
import threading
import webbrowser

def open_chrome_browser(url: str):
    time.sleep(1.2)
    try:
        subprocess.run(["open", "-a", "Google Chrome", url], check=False)
    except Exception:
        webbrowser.open(url)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    url = f"http://{host}:{port}"
    print("=========================================================")
    print("    TextReader AI — Standalone All-In-One Application    ")
    print("=========================================================")
    print(f"  Web App:       {url}")
    print(f"  API Docs:      {url}/docs")
    print(f"  Health Check:  {url}/api/health")
    print("  [*] Opening Google Chrome automatically...")
    print("=========================================================")
    threading.Thread(target=open_chrome_browser, args=(url,), daemon=True).start()
    uvicorn.run(app, host=host, port=port)
