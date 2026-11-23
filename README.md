# 🔍 Who Is in the Photo?

An offline face recognition platform that identifies people in group photos by matching them against a folder of individual reference photos.

Built for the **Take-home Task A** assessment.

---

## ✨ Features

- **Folder Upload** — Select a reference folder of individual photos (each filename = person's name, e.g. `Priya Sharma.jpg`).
- **Group Photo Recognition** — Upload any group photo to locate, frame, and identify everyone in it.
- **Smart Matching** — State-of-the-art ArcFace (512-d) embeddings with cosine similarity and greedy one-to-one assignment.
- **Unknown Detection** — Individuals present in the photo who are not in the reference folder are reliably marked as **"Unknown"**.
- **Adjustable Threshold Slider** — Fine-tune matching sensitivity in real time ($0.20$ to $0.70$, default $0.40$) without re-detecting.
- **Side-by-Side Comparison** — Click any matched face to open a side-by-side modal comparing the face crop directly against the original reference photo.
- **Spotlight Mode & Canvas Overlay** — Interactive bounding boxes, two-way list-to-canvas highlighting, and high-DPI retina face thumbnails.
- **100% Offline at Runtime** — Zero external API calls, zero cloud dependencies. Everything runs locally on CPU via ONNX Runtime.

---

## 🛠 Requirements

- **Python 3.10+** (tested on Python 3.11)
- `pip`
- A modern web browser (Google Chrome, Microsoft Edge, or Mozilla Firefox recommended)

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Prashanf/Face_Recognition_dev.git
cd Face_Recognition_dev
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

*(Note: If using an existing virtual environment, simply activate it before running).*

### 3. Run the application
```bash
cd backend
python main.py
```

### 4. Open in your browser
Navigate to:
```
http://127.0.0.1:8000
```

---

### First Run & Offline Verification

On the first ever run, InsightFace checks `backend/models/`. The model pack (`buffalo_l`, ~300MB containing SCRFD detector `det_10g.onnx` and ArcFace recognizer `w600k_r50.onnx`) is bundled in `backend/models/models/buffalo_l/`. 

To verify true offline execution:
1. Turn off your Wi-Fi / disconnect your network.
2. Start the app: `python backend/main.py`.
3. Ingest reference photos and group photos — the entire pipeline executes locally with zero network calls.

---

## 📖 Usage Guide

1. **Select Reference Folder (Step 1)**:
   - Click **Select reference folder** and choose a folder with individual photos.
   - Each filename defines the person's identity (e.g. `Priya Sharma.jpg`, `Rahul_Verma.png`).
   - The system extracts 512-d ArcFace embeddings, automatically handles multiple faces (picks largest), skips corrupted files, and displays a summary with expandable warnings.
2. **Upload Group Photo (Step 2)**:
   - Click **Upload photo** or drag-and-drop a group photo into the viewing stage.
   - The system immediately locates and frames all detected faces.
3. **Recognize Faces (Step 3)**:
   - Click **Recognize faces**.
   - Matched people are boxed in **Green (`#10B981`)** with their name badges.
   - Unrecognized individuals are boxed in **Amber (`#F59E0B`)** as "Unknown".
4. **Adjust Threshold & Compare**:
   - Drag the **Similarity Threshold** slider to tighten or loosen matching in real time.
   - Click any person in the photo or sidebar list to focus them, or click **Compare** to view a high-resolution side-by-side comparison against their reference photo.
5. **Reset All**:
   - Click **Reset All** in the top navigation bar at any time to clear in-memory embeddings and reset the workspace.

---

## 🏗 Tech Stack

| Component | Technology | Role / Justification |
|---|---|---|
| **Backend Framework** | Python 3.10+, FastAPI, Uvicorn | High-performance asynchronous REST API serving both endpoints and static assets |
| **Face Detection** | InsightFace (SCRFD / `det_10g.onnx`) | High recall on small, angled, and partially occluded faces in group photos |
| **Face Recognition** | InsightFace (ArcFace / `w600k_r50.onnx`) | 512-d discriminative embeddings; state-of-the-art one-shot verification accuracy |
| **Matching Engine** | NumPy, SciPy (Cosine similarity) | Greedy one-to-one bipartite matching preventing duplicate person assignments |
| **Runtime Engine** | ONNX Runtime (`CPUExecutionProvider`) | Lightweight, self-contained CPU inference without GPU/CUDA or heavy PyTorch overhead |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (ES6+) | Zero build step, no npm/node dependency, fast loading, custom canvas overlay |

---

## 📁 Project Structure

```
Face_Recognition_dev/
├── backend/
│   ├── main.py              # FastAPI server, endpoints, and error handling
│   ├── face_engine.py       # Face detection, embedding, matching & edge-case logic
│   ├── requirements.txt     # Pinned Python dependencies
│   └── models/              # Local InsightFace model weights (buffalo_l)
├── frontend/
│   ├── index.html           # Single-page interface layout
│   ├── css/
│   │   └── styles.css       # Clean, modern design system & animations
│   └── js/
│       └── app.js           # Client-side workflows, canvas rendering & compare modal
├── test_data/
│   ├── people_and_group_photos/ # Practice reference photos and multi-person group shots
│   ├── reference_folder/        # 50 practice reference photos (LFW)
│   └── group_photos/            # Practice synthetic group photos
├── tools/
│   ├── make_test_data.py    # LFW test data generator
│   └── evaluate.py          # Benchmark evaluation script across threshold sweeps
├── .gitignore               # Excludes models, venvs, and development prompts
├── README.md                # Quickstart & installation documentation
├── RESEARCH.md              # Comparative research, model benchmarks & citations
├── AI_COLLAB.md             # Key decisions, AI steering evidence & chat excerpts
└── ARCHITECTURE.md          # Architecture diagram, data flows & trade-off analysis
```

---

## ⚙ Configuration

| Parameter | Default | Range | Description |
|---|---|---|---|
| `Similarity Threshold` | `0.40` | `0.20 – 0.70` | Minimum cosine similarity required to accept a match (live adjustable via UI slider) |
| `Detection Size` | `(640, 640)` | Fixed | Input resolution for SCRFD face detector |
| `Max Image Dimension` | `4000px` | Internal | Images exceeding 4000px are downsampled for inference to prevent memory spikes |
| `Server Port` | `8000` | Configurable | Local FastAPI server port (`http://127.0.0.1:8000`) |

---

## 📝 License

This project was developed as an assessment submission for Take-home Task A. The face recognition models (InsightFace/ArcFace) are licensed under the MIT License for academic and non-commercial evaluation.
