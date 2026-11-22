# Who Is in the Photo? — Face Recognition App

Offline face detection and recognition platform designed for group photographs.

---

## Current Milestone: Step 1 — Face Detection with Bounding Boxes

### What is Built
- **Minimal Single-Page Web App:** Pure HTML5, CSS3, and Vanilla JavaScript (zero bundler, no npm install, no framework dependencies).
- **Photo Viewer & Canvas Overlay:** Upload any group photo to view it. The canvas dynamically synchronizes to image dimensions and scales bounding boxes accurately.
- **Offline Face Detection:** Powered by InsightFace SCRFD (`buffalo_l` pack) with CPUExecutionProvider via ONNX Runtime.
- **Ordered Face Numbering:** Faces are sorted left-to-right (`x1` coordinate) and labeled with styled badges (`Face 1`, `Face 2`, etc.) in `#3B82F6` blue.
- **Summary & Status Bar:** Clear feedback on detection counts (e.g., "Detected 11 faces") and loading indicators.

---

## Tech Stack
- **Backend:** Python 3.10+ (tested on Python 3.11), FastAPI, Uvicorn, InsightFace (`buffalo_l` / SCRFD), OpenCV, NumPy
- **Frontend:** Pure HTML5, CSS3, Vanilla ES6+ JavaScript
- **Execution:** 100% offline — local ONNX model weights in `backend/models/`

---

## Quick Start

### 1. Using Existing Virtual Environment (`ml_env`)
Activate the existing virtual environment located at `/Users/vishalsharma/Documents/Prashant/venvs/ml_env`:
```bash
source /Users/vishalsharma/Documents/Prashant/venvs/ml_env/bin/activate
```

Or run directly using the environment's Python:
```bash
/Users/vishalsharma/Documents/Prashant/venvs/ml_env/bin/python Face_Recognition_dev/backend/main.py
```

### 2. Standard Setup (Alternative)
```bash
cd Face_Recognition_dev/backend
pip install -r requirements.txt
python main.py
```

### 3. Open in Browser
Visit:
```
http://127.0.0.1:8000
```

---

## How to Test Step 1
1. Click **📷 Upload Photo** (or click inside the empty dashed viewer area).
2. Select a group photo (e.g. from `Face_Regognition/test_data/group_photos/group_easy.jpg`).
3. Click **🔍 Detect Faces**.
4. The button displays a spinner (`Detecting...`) while the local model processes the image.
5. All detected faces appear framed in blue bounding boxes with `Face 1`, `Face 2`, ... badges, and the status bar updates with the count.
