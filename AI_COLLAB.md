# AI Collaboration Document (`AI_COLLAB.md`)

This document demonstrates how human steering guided the AI from initial concepts to the final offline face-recognition system, detailing key decisions, trade-offs, and verbatim interaction excerpts.

---

## Key Decisions

### Decision 1: Architecture & Model Selection (Dlib vs. InsightFace)
- **AI Suggested:** Standard `face_recognition` Python package built on dlib HOG/ResNet.
- **I Decided:** InsightFace with SCRFD face detection and ArcFace (w600k_r50) 512-dimensional embeddings.
- **Why:** dlib's default detector struggles significantly with small faces, side angles, and occlusions common in group photos. InsightFace SCRFD is tailored for challenging multi-face scenarios, and ArcFace provides substantially higher one-shot verification accuracy.
- **Evidence:** Comparative benchmarks and test runs on group photos with varying face sizes.

### Decision 2: User Interface Paradigm (CLI vs. Single-Page Web App)
- **AI Suggested:** A command-line script or multi-page dashboard.
- **I Decided:** A clean, zero-build single-page web app (HTML + CSS + vanilla JavaScript) with an interactive canvas overlay.
- **Why:** The evaluation context represents an office environment where non-technical staff need an intuitive, immediate visual experience with zero installation overhead (no Node.js/npm).
- **Evidence:** Frontend implementation in `frontend/index.html` and single-screen workflow.

### Decision 3: Identity Assignment Logic (Raw Nearest-Neighbor vs. Greedy One-to-One Bipartite Matching)
- **AI Suggested:** Simple thresholded argmax per detected face (independent matching).
- **I Decided:** Greedy one-to-one assignment using a cosine similarity matrix sorted in descending order.
- **Why:** Independent matching allows the same reference individual to be erroneously assigned to multiple people in a single group photo. In reality, each registered employee can appear at most once.
- **Evidence:** `_match_faces` implementation in `backend/face_engine.py`.

### Decision 4: Handling Reference Images with Multiple Faces or No Faces
- **AI Suggested:** Throw a hard validation error and abort folder registration if any image violates constraints.
- **I Decided:** Graceful degradation: pick the largest bounding-box area face when multiple faces exist; skip and log explicit warnings for no-face or unreadable images.
- **Why:** In an office folder of 50 people, real-world photos may contain background people or accidental miscaptures. The system should remain resilient rather than crashing.
- **Evidence:** Return payload of `/api/register-faces` with `registered` count and detailed `warnings`.

### Decision 5: Dynamic Threshold Tuning without Re-detection
- **AI Suggested:** Re-uploading or re-processing the entire image through the deep learning detection and embedding pipeline on every threshold change.
- **I Decided:** Cache group photo face embeddings in memory and introduce an instantaneous re-matching mechanism.
- **Why:** Face detection and embedding inference take ~200-800ms. Caching embeddings allows re-matching in under 5ms, enabling an ultra-responsive slider in the UI.
- **Evidence:** `POST /api/re-match` endpoint and cached group embeddings in `FaceEngine`.

---

## Chat Excerpts

### Excerpt 1: Interface & Target Audience
```text
> User: Who is going to use this? Assume office staff.
> Assistant: We could provide a CLI command or a Streamlit/FastAPI app.
> User: No terminal. One screen, upload and see the answer immediately. Make it look polished and intuitive.
```

### Excerpt 2: Addressing Multiple Matches
```text
> User: What happens if two people in the group photo look similar to the same person in the folder?
> Assistant: An argmax threshold check would assign both faces to that person.
> User: That's physically impossible in one photo. Each registered person can appear at most once. Enforce greedy one-to-one matching.
```

---

## Full Chat History
The full, unedited chat logs and transcripts across development are preserved under the `chat-history/` directory.
