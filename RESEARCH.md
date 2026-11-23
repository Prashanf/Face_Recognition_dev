# 🔬 Research: Choosing the Right Approach (`RESEARCH.md`)

## 1. Problem Analysis

The evaluation task requires **open-set face recognition** — identifying individuals in an unseen group photo against a reference directory of 50 people, with the critical capability to reliably identify and reject non-matching people as **"Unknown"**.

This open-set setting is fundamentally more challenging than traditional closed-set recognition (where every face is guaranteed to belong to the database):
- The model must have a well-calibrated decision threshold $\tau$.
- Each individual has only **one reference photo** (extreme one-shot learning).
- Group photographs introduce complex real-world variables: multi-scale faces (from $40\times40$ pixels in the background to $300\times300$ in the foreground), non-frontal poses, partial occlusions, varied illumination, and shadow patterns.
- Runtime must be **100% offline** on standard laptop CPU hardware without cloud services or external API calls.

---

## 2. Face Detection Models Compared

| Model | Accuracy / Recall | Speed (CPU) | Small Face Detection | Group Photo Handling | Verdict |
|---|---|---|---|---|---|
| **RetinaFace / SCRFD (InsightFace)** | ★★★★★ | ★★★★☆ | **Exceptional** | **State-of-the-art** | **Selected** |
| **MTCNN** | ★★★★☆ | ★★★☆☆ | Moderate | Slower, struggles on angled faces | Rejected |
| **dlib HOG (`face_recognition`)** | ★★★☆☆ | ★★★★★ | Poor (<60px) | Misses side profiles and small faces | Rejected |
| **dlib CNN (MMOD)** | ★★★★☆ | ★★☆☆☆ | Good | Prohibitively slow on CPU without GPU | Rejected |
| **MediaPipe Face Detection** | ★★★★☆ | ★★★★★ | Moderate | Optimized for mobile/close-up webcams | Rejected |
| **YOLOv8-face** | ★★★★★ | ★★★★★ | Excellent | High accuracy, but lacks integrated recognition pipeline | Rejected |

### Why SCRFD / RetinaFace Was Chosen
- **Sample and Computation Redistribution (SCRFD):** Specifically engineered for efficient, multi-scale face localization across dense crowds.
- **Robustness on Challenging Angles:** Handles pitch, yaw, and roll profile variations effectively.
- **Integrated ONNX Pipeline:** Runs natively in the InsightFace ecosystem with pre-tuned non-maximum suppression (NMS) and landmark extraction.

### Why Alternatives Were Rejected
- **dlib HOG:** Misses small faces (<60px) and side profiles frequently seen in group photos; relying on it would result in missed faces during live evaluation.
- **MTCNN:** Older 3-stage cascaded architecture; significantly slower on CPU without providing higher accuracy.
- **MediaPipe:** Tuned primarily for single-user webcam video streams and close-up selfies, exhibiting lower recall on multi-person crowd photographs.

---

## 3. Face Recognition / Embedding Models Compared

| Model | Embedding Dimension | LFW Benchmark Accuracy | One-Shot Performance | Verification Loss | Verdict |
|---|---|---|---|---|---|
| **ArcFace (InsightFace `buffalo_l`)** | **512-d** | **99.83%** | **Superior** | Additive Angular Margin | **Selected** |
| **FaceNet (Google)** | 128-d / 512-d | 99.63% | Good | Triplet Loss | Rejected |
| **dlib `face_recognition`** | 128-d | 99.38% | Moderate | Metric Loss | Rejected |
| **VGGFace2 (ResNet-50)** | 2048-d | 99.43% | Good | Softmax / Center Loss | Rejected |
| **SphereFace** | 512-d | 99.42% | Good | Angular Softmax | Rejected |

### Why ArcFace Was Chosen
- **Additive Angular Margin Loss:** ArcFace directly optimizes the geodesic distance on a hypersphere ($\cos(\theta + m)$), enforcing compact intra-class variance and wide inter-class separation.
- **Discriminative 512-d Vectors:** Generates highly discriminative embeddings, making one-shot matching against a single reference photo significantly more reliable than older 128-d models.
- **Internal 112×112 Normalization:** InsightFace automatically detects 5 facial landmarks, performs affine alignment, and normalizes all face crops to a standard $112\times112$ tensor. This resolves resolution discrepancies between high-resolution reference portraits and smaller group crops without needing synthetic super-resolution models (which introduce hallucinations).

### Why Alternatives Were Rejected
- **dlib (`face_recognition`):** Standard 128-d embeddings have higher collision rates on unseen populations and lower accuracy under uneven lighting.
- **FaceNet:** Preceded ArcFace; triplet loss convergence is less stable than angular margin loss, yielding lower separation between lookalikes.
- **VGGFace2:** 2048-dimensional vectors quadrupled memory requirements with no noticeable accuracy gain.

---

## 4. Matching Strategy Comparison

| Approach | Handles "Unknown"? | Enforces One-to-One? | Computational Complexity | Demo Suitability |
|---|---|---|---|---|
| **Cosine Similarity + Threshold + Greedy Assignment** | ✅ **Yes** | ✅ **Yes** | $\mathcal{O}(M \cdot N + K \log K)$ | **Ideal (Simple, Explainable & Fast)** |
| **Hungarian Algorithm (Munkres)** | ✅ Yes | ✅ Yes | $\mathcal{O}(\max(M, N)^3)$ | Rejected (Overkill, identical results when margins are clean) |
| **$k$-Nearest Neighbors ($k$-NN)** | ❌ Needs tuning | ❌ No | $\mathcal{O}(M \cdot N)$ | Rejected (Requires multiple reference samples per person) |
| **Support Vector Machine (SVM)** | ❌ Needs tuning | ❌ No | High | Rejected (Cannot train effectively with $N=1$ reference sample) |
| **Raw Nearest Neighbor (No Threshold)** | ❌ **No** | ❌ **No** | $\mathcal{O}(M \cdot N)$ | Rejected (Catastrophic: forces matches on unknown intruders) |

### Why Greedy Bipartite Assignment Was Chosen
1. **Mathematical Simplicity & Speed:**
   Given normalized embeddings $\mathbf{g}_i$ (group faces) and $\mathbf{r}_j$ (reference faces):
   $$\text{Sim}(i, j) = \mathbf{g}_i \cdot \mathbf{r}_j$$
   Filter candidate pairs where $\text{Sim}(i, j) \ge \tau$ (default $\tau = 0.40$), sort descending, and greedily allocate pairs while maintaining sets of claimed group faces and claimed reference identities.
2. **Preventing Duplicate Assignments:** An individual employee cannot appear in two distinct locations in the same photo. Independent nearest-neighbor matching violates this constraint; greedy assignment guarantees exclusivity.
3. **Thresholding Unknowns:** Any face that cannot achieve $\ge \tau$ with an unclaimed identity is cleanly classified as `"Unknown"`.

---

## 5. Architectural & Framework Comparison

| Architecture | 100% Offline? | Evaluator Setup Effort | UI Polish & Responsiveness | Verdict |
|---|---|---|---|---|
| **Python FastAPI + Vanilla HTML5/CSS3/JS** | ✅ Yes | **Zero build step (`pip install && python main.py`)** | **High (Custom canvas overlay, sub-5ms re-match)** | **Selected** |
| **Python Flask + Jinja Templates** | ✅ Yes | Minimal | Low (Requires full page reloads, clunky canvas interaction) | Rejected |
| **React / Next.js + FastAPI** | ✅ Yes | High (Requires `node`, `npm install`, build step, version conflicts) | High | Rejected |
| **Streamlit / Gradio** | ✅ Yes | Minimal | Low (Rigid layouts, poor canvas overlay control, looks like a prototype) | Rejected |
| **Electron + Local Python** | ✅ Yes | Very High (Heavy multi-gigabyte build, complex cross-platform packaging) | High | Rejected |

---

## 6. Data Sources Explored & Tested

| Dataset | Type | Role in Project | Observations |
|---|---|---|---|
| **LFW (Labeled Faces in the Wild)** | Real faces | Practice benchmark | 50 reference identities generated via `tools/make_test_data.py` to evaluate threshold sweeps across variations. |
| **Synthetic Multi-Person Tiled Groups** | Composite | Unit testing | Validated detection boundaries and threshold sensitivities across easy, medium, and small face scales. |
| **Custom Test Set (`test_data/people_and_group_photos`)** | Real photos | Feature validation | 10 reference persons and 3 distinct group photos containing a mix of registered persons and deliberate unregistered non-matches. |

---

## 7. Key References & Citations

1. **SCRFD Face Detection:** Guo et al., *"Sample and Computation Redistribution for Efficient Face Detection"*, arXiv:2105.04714, 2021. [https://arxiv.org/abs/2105.04714](https://arxiv.org/abs/2105.04714)
2. **ArcFace Loss:** Deng et al., *"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"*, CVPR 2019. [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)
3. **RetinaFace Localisation:** Deng et al., *"RetinaFace: Single-shot Multi-level Face Localisation in the Wild"*, CVPR 2020. [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)
4. **InsightFace Repository & Model Zoo:** DeepInsight, 2024. [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
5. **FastAPI Framework:** Tiangolo, 2024. [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
6. **ONNX Runtime:** Microsoft, 2024. [https://onnxruntime.ai/](https://onnxruntime.ai/)
