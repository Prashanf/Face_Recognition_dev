# Who Is in the Photo? — Face Recognition App

Offline face detection and recognition platform designed for group photographs.

---

## Current Milestones

### Feature 1: Face Detection with Bounding Boxes
- **Minimal Single-Page Web App:** Pure HTML5, CSS3, and Vanilla JavaScript (zero bundler, no npm install, no framework dependencies).
- **Photo Viewer & Canvas Overlay:** Upload any group photo to view it. The canvas dynamically synchronizes to image dimensions and scales bounding boxes accurately.
- **Offline Face Detection:** Powered by InsightFace SCRFD (`buffalo_l` pack) with CPUExecutionProvider via ONNX Runtime.
- **Ordered Face Numbering:** Faces are sorted left-to-right (`x1` coordinate) and labeled with styled badges (`Face 1`, `Face 2`, etc.) in `#3B82F6` blue.

### Feature 2: Reference Folder Upload + Face Recognition + Results UI
- **Reference Folder Upload (`webkitdirectory`):** Ingests an entire folder containing ~50 individual photos (filenames define identity, e.g. `Priya Sharma.jpg`), extracts 512-dimensional ArcFace embeddings, and stores them in memory.
- **Greedy One-to-One Matching:**
  - Pairwise cosine similarity matrix between detected group faces and all reference embeddings.
  - Threshold filtering (default `0.40`, configurable via slider between `0.20` and `0.70`).
  - Greedy one-to-one assignment preventing any person from being matched to multiple faces.
  - Non-matching or below-threshold individuals are clearly labeled as **"Unknown"**.
- **Interactive Results Panel & UI:**
  - Bounding boxes dynamically recolor on canvas: **Green (`#10B981`)** for matched, **Amber (`#F59E0B`)** for unknown.
  - Bounding box labels display the person's name or "Unknown".
  - Two-way interactive hover: hovering a card in the results panel glows/pulses the bounding box on the canvas; hovering a box on the canvas highlights the card.
  - Crisp 48×48px cropped thumbnails extracted from full-resolution photos.
  - Proportion bar and grouped list (`── Identified (N) ──` and `── Unknown (N) ──`).
- **Instant Re-Matching:** Moving the threshold slider triggers `/api/re-match` on cached embeddings, updating canvas and results in real-time (<5ms).

---

## Tech Stack
- **Backend:** Python 3.10+ (tested on Python 3.11), FastAPI, Uvicorn, InsightFace (`buffalo_l` SCRFD + ArcFace), ONNX Runtime, OpenCV, NumPy
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

## How to Test Step 2
1. Click **📁 Select Folder** under Step 1 and choose a folder with individual reference photos (e.g. `Face_Regognition/test_data/reference_folder`).
2. The button will show processing progress, then confirm `✅ 50/50 faces registered` (with expandable warnings if any).
3. Click **📷 Upload Photo** (or drag and drop) and select a group photo (e.g. `Face_Regognition/test_data/group_photos/group_easy.jpg`).
4. Click **🔍 Recognize Faces**.
5. Bounding boxes on the canvas update to Green (matched) and Amber (unknown).
6. The Results Panel opens on the right displaying individual face thumbnails, names, confidence scores, and summary breakdown.
7. Adjust the **Similarity Threshold slider** to test real-time strict vs. lenient matching without re-uploading!
