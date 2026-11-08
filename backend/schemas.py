from typing import List, Optional
from pydantic import BaseModel, Field

class CharacterPrediction(BaseModel):
    character: str = Field(..., description="Predicted alphanumeric character (0-9, A-Z)")
    confidence: float = Field(..., description="Confidence score percentage (0-100)")
    box: Optional[List[int]] = Field(None, description="Bounding box [x, y, width, height]")

class TopPrediction(BaseModel):
    character: str
    confidence: float

class SinglePredictionResponse(BaseModel):
    character: str
    confidence: float
    top_predictions: List[TopPrediction]
    characters: Optional[List[CharacterPrediction]] = None
    annotated_image: Optional[str] = None
    message: str = "Success"

class SequencePredictionResponse(BaseModel):
    recognized_text: str
    total_characters: int
    characters: List[CharacterPrediction]
    annotated_image: Optional[str] = Field(None, description="Base64 encoded image with bounding boxes drawn")
    message: str = "Success"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    num_classes: int
