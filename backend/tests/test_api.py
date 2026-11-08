import os
import sys
from fastapi.testclient import TestClient
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 36


def test_ownership_endpoint():
    response = client.get("/api/ownership")
    assert response.status_code == 200
    data = response.json()
    assert data["developer"] == "Trigun Parvathapuram"
    assert data["project"] == "Scrawl"
    assert "96.32%" in data["validation_accuracy"]


def test_predict_character_multipart():
    # Create synthetic test image of digit '7'
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(img, "7", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 4)
    _, img_bytes = cv2.imencode(".png", img)

    response = client.post(
        "/api/predict/character",
        files={"file": ("test_7.png", img_bytes.tobytes(), "image/png")},
        data={"top_k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert "character" in data
    assert "confidence" in data
    assert len(data["top_predictions"]) <= 3


def test_predict_sequence_multipart():
    # Create synthetic image with word 'CODE'
    img = np.zeros((100, 300), dtype=np.uint8)
    cv2.putText(img, "CODE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 4)
    _, img_bytes = cv2.imencode(".png", img)

    response = client.post(
        "/api/predict/sequence",
        files={"file": ("test_code.png", img_bytes.tobytes(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "recognized_text" in data
    assert data["total_characters"] > 0
    assert "annotated_image" in data
    assert data["annotated_image"].startswith("data:image/")


def test_predict_canvas_base64_drawing():
    # Test base64 canvas drawing endpoint with user canvas "Hi"
    canvas_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "user_canvas.png")
    if os.path.exists(canvas_file):
        import base64
        with open(canvas_file, "rb") as f:
            b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
        response = client.post(
            "/api/predict/character/base64",
            json={"image": b64, "top_k": 3, "mode": "auto"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["character"] == "HI"
        assert data["confidence"] > 90.0
        assert len(data["characters"]) == 2
        assert data["annotated_image"] is not None


if __name__ == "__main__":
    print("Running integration tests...")
    test_health_check()
    print("Health check passed.")
    test_ownership_endpoint()
    print("Ownership endpoint passed.")
    test_predict_character_multipart()
    print("Character endpoint passed.")
    test_predict_sequence_multipart()
    print("Sequence endpoint passed.")
    test_predict_canvas_base64_drawing()
    print("Canvas drawing base64 endpoint passed.")
    print("All tests passed successfully!")
