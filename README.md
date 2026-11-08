# ✍️ Scrawl — Neural Handwritten OCR & Intelligent Transcription

> **Created by: Trigun Parvathapuram**  
> **Task 3: Handwritten Character & Note Recognition (CodeAlpha Machine Learning Track)**  
> Production-ready fullstack deep learning web application powered by a custom 36-class Convolutional Neural Network (CNN) fine-tuned directly on user handwriting (**96.32% Validation Accuracy**).

---

## 🌟 Key Features

* **👑 Creator & Ownership Controls (Trigun Parvathapuram)**:
  * Personalized model weighting trained on Trigun's handwriting dataset (A–Z, a–z, 0–9).
  * Ownership Attribution panel & modal with verified creator credentials.
  * Optional signature watermark on text copy and export: `"Transcribed with Scrawl by Trigun Parvathapuram"`.
  * Technical pipeline diagnostics and personal accuracy metrics.

* **✨ Calming, Aesthetic Light Theme & Animations**:
  * Soft ambient background glow effects and responsive cards.
  * Micro-interactions on buttons, mode pills, and drawing canvas.
  * Number counter animations and pop-in character confidence breakdown badges.
  * Smooth transitions between note OCR, live canvas pad, and owner controls.

* **📝 Handwritten Note & Sequence Transcription**:
  * Upload photos or scans of handwritten notes, words, or alphanumeric codes.
  * Morphological background division (`cv2.morphologyEx` + `cv2.divide`) to neutralize shadows and camera lighting.
  * Document ROI polygon detection to filter out desk or bedsheet patterns.
  * Context-aware disambiguation (`Auto`, `Digits Only`, `Letters Only`, `Mixed Alphanumeric`).
  * Visual segmentation overlay displaying color-coded bounding boxes and per-character confidence labels.
  * One-click copy to clipboard and text file export with signature watermarking.

* **🎨 Interactive Live Drawing & Character Pad**:
  * Smooth HTML5 canvas with touch and mouse support for drawing characters, digits, or handwritten words (e.g., `"Hi"`, `"AI"`, `"42"`, `"A"`).
  * **Intelligent Multi-Character Segmentation**: Automatically separates drawn letters from left to right while merging diacritics (dots on `i` and `j` linked to their vertical stems).
  * **Dual Display Mode**: Displays large hero text with dynamic scaling, per-character confidence breakdown badges, and top candidate distribution bars.
  * **Live Segmentation Overlay**: Renders color-coded bounding boxes and prediction tags directly on your drawn strokes.
  * Quick-draw sample buttons (`"Hi"`, `"AI"`, `"A"`, `"42"`) for immediate testing, plus adjustable pen stroke thickness (12px–32px).

* **⚡ Production-Ready REST API**:
  * Built with **FastAPI** for high performance and asynchronous throughput.
  * Auto-generated interactive Swagger UI documentation at `/docs` and ReDoc at `/redoc`.
  * Support for both standard `multipart/form-data` file uploads and `base64` image payloads.

* **🐳 Ready to Deploy**:
  * Optimized multi-stage `Dockerfile` (CPU-optimized, non-root user, healthcheck).
  * `docker-compose.yml` for single-command container startup.
  * Pre-configured `render.yaml` and `Procfile` for one-click deployment on Render, Railway, or Fly.io.

---

## 🏗️ Architecture

```
                                  [Web Browser Client]
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
          [Interactive Drawing Pad]                    [Handwritten Note Upload]
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │ HTTP REST / JSON
                                           ▼
                                [FastAPI Backend Server]
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
       [Single Char Pipeline]                              [Note Sequence Pipeline]
        - Aspect Ratio Square Pad                           - Adaptive Otsu Threshold
        - 28x28 Normalization                               - Contour Extraction
        - Inversion Detection                               - Multi-Line & Space Clustering
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                      [36-Class Alphanumeric Deep CNN]
                               (442,884 Params)
                                   Validation Accuracy: 96.32% (Trigun Personalized)
```

---

## 📂 Project Directory Structure

```
textreader/
├── backend/
│   ├── app.py                     # FastAPI application & route definitions
│   ├── inference.py               # Image processing & CNN inference engine
│   ├── schemas.py                 # Pydantic data validation schemas
│   └── tests/
│       └── test_api.py            # Automated API integration tests
├── frontend/
│   ├── index.html                 # Responsive UI with creator ownership panel & modal
│   ├── style.css                  # Calming light-theme styling, glow effects & animations
│   └── app.js                     # Canvas controller, drag-drop & API integration
├── emnist_alphanumeric_cnn.keras  # Trained 36-class Keras model weights (5.4MB)
├── model.py                       # Original training and evaluation script
├── Dockerfile                     # Production container definition
├── docker-compose.yml             # Local container orchestration
├── render.yaml                    # Render Blueprint configuration
├── Procfile                       # Cloud platform process runner
├── requirements.txt               # Deployment dependencies
├── run_app.sh                     # 1-click local launch script
└── README.md
```

---

## 🚀 Quick Start (Local Setup)

### Option 1: VS Code 1-Click Execution (Recommended)
1. Open the project folder in VS Code:
   ```bash
   code /Users/trigun06/Downloads/textreader
   ```
2. Press **F5** (or select **"🚀 Run Scrawl App (app.py)"** from the Run & Debug pane).
3. The server starts at `http://127.0.0.1:8000` and automatically opens Google Chrome.

### Option 2: Using the 1-Click Shell Script
```bash
cd /Users/trigun06/Downloads/textreader
./run_app.sh
```

### Option 3: All-in-One Standalone Runner
Run everything directly from a single self-contained file (embedded frontend):
```bash
python textreader_standalone.py
```

### Option 4: Manual Terminal Startup
1. **Activate the virtual environment**:
   ```bash
   cd /Users/trigun06/Downloads/textreader
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the FastAPI server**:
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```
4. **Open in your browser**:
   * **Fullstack Web UI**: [http://localhost:8000](http://localhost:8000)
   * **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Running Automated Tests

Run the backend integration test suite to verify all endpoints and model loading:
```bash
.venv/bin/python backend/tests/test_api.py
```

Expected output:
```
Health check passed.
Ownership endpoint passed.
Character endpoint passed.
Sequence endpoint passed.
Canvas drawing base64 endpoint passed.
All tests passed successfully!
```

---

## 🚢 Production Deployment Guides

### 1. Deploy with Docker
```bash
# Build and run with Docker Compose
docker compose up --build -d

# Or build manually:
docker build -t textreader-ai .
docker run -p 8000:8000 textreader-ai
```
The application will be live at `http://localhost:8000`.

---

### 2. Deploy to Render (Recommended)
1. Push this project to a GitHub repository (ensure `.dockerignore` prevents pushing large raw dataset CSVs).
2. Go to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** -> **Web Service** and connect your GitHub repo.
4. Render automatically detects the `render.yaml` or you can manually configure:
   * **Environment**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**. Your app is live with a free HTTPS URL!

---

### 3. Deploy to Railway
1. Install Railway CLI or connect via GitHub on [railway.app](https://railway.app).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Railway automatically detects the included `Dockerfile` and builds the optimized production container.
4. Add a custom domain or use the generated Railway URL.

---

## 📡 REST API Reference

| Method | Endpoint | Description | Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status & model metadata | None |
| `POST` | `/api/predict/sequence` | Transcribes note or word image with bounding boxes | Multipart form file (`file`) |
| `POST` | `/api/predict/sequence/base64` | Transcribes note or sequence from base64 string | JSON `{ "image": "data:image/png;base64,..." }` |
| `POST` | `/api/predict/character` | Predicts single character with Top-K candidates | Multipart form file (`file`, `top_k`) |
| `POST` | `/api/predict/character/base64` | Predicts single character from canvas base64 | JSON `{ "image": "...", "top_k": 3 }` |
