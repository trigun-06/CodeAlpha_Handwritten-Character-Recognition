import os
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.schemas import (
    SinglePredictionResponse,
    SequencePredictionResponse,
    HealthResponse
)
from backend.inference import get_engine, NUM_CLASSES

app = FastAPI(
    title="Scrawl — Neural Handwritten OCR & Intelligent Transcription",
    description="Engineered by Trigun Parvathapuram (CodeAlpha ML Track Task 3). Fullstack Deep Learning API powered by a fine-tuned CNN on user handwriting with document illumination flattening and context disambiguation.",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


class Base64ImagePayload(BaseModel):
    image: str
    top_k: Optional[int] = 3
    mode: Optional[str] = "auto"


def extract_bytes_from_base64(data_url: str) -> bytes:
    """Parses raw bytes from a data URL or raw base64 string."""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        return base64.b64decode(data_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")


@app.get("/api/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    engine = get_engine()
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded(),
        "model_version": "2.0.0-scrawl-trigun-emnist",
        "num_classes": NUM_CLASSES
    }


@app.get("/api/ownership", tags=["Ownership"])
def ownership_details():
    return {
        "project": "Scrawl",
        "version": "2.0.0",
        "developer": "Trigun Parvathapuram",
        "track": "CodeAlpha Machine Learning Internship — Task 3",
        "architecture": "Convolutional Neural Network (CNN) with Morphological Background Normalization",
        "validation_accuracy": "96.32%",
        "personalization": "Fine-tuned on Trigun's handwriting dataset (A–Z, a–z, 0–9)",
        "copyright": "© 2026 Trigun Parvathapuram. All rights reserved."
    }


@app.post("/api/predict/character", response_model=SinglePredictionResponse, tags=["Inference"])
async def predict_single(
    file: Optional[UploadFile] = File(None),
    top_k: int = Form(3),
    mode: str = Form("auto")
):
    if file is None:
        raise HTTPException(status_code=400, detail="No image file provided.")

    try:
        content = await file.read()
        engine = get_engine()
        result = engine.predict_single_character(content, top_k=top_k, mode=mode)
        return {**result, "message": "Successfully predicted character."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/character/base64", response_model=SinglePredictionResponse, tags=["Inference"])
async def predict_single_base64(payload: Base64ImagePayload):
    try:
        image_bytes = extract_bytes_from_base64(payload.image)
        engine = get_engine()
        result = engine.predict_single_character(image_bytes, top_k=payload.top_k or 3, mode=payload.mode or "auto")
        return {**result, "message": "Successfully predicted character."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/sequence", response_model=SequencePredictionResponse, tags=["Inference"])
async def predict_sequence(
    file: Optional[UploadFile] = File(None),
    mode: str = Form("auto")
):
    if file is None:
        raise HTTPException(status_code=400, detail="No image file provided.")

    try:
        content = await file.read()
        engine = get_engine()
        result = engine.predict_note_sequence(content, annotate=True, mode=mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence recognition error: {str(e)}")


@app.post("/api/predict/sequence/base64", response_model=SequencePredictionResponse, tags=["Inference"])
async def predict_sequence_base64(payload: Base64ImagePayload):
    try:
        image_bytes = extract_bytes_from_base64(payload.image)
        engine = get_engine()
        result = engine.predict_note_sequence(image_bytes, annotate=True, mode=payload.mode or "auto")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence recognition error: {str(e)}")


# Mount frontend static directory if present
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"message": "Frontend index.html not found. API is running."})
