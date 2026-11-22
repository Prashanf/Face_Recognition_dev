# Research & Evaluation (`RESEARCH.md`)

This document reviews the technical research, architectural alternatives, models, and datasets evaluated for building the offline face-recognition system.

---

## 1. Problem Landscape & Constraints

The application must operate **100% offline at runtime** on standard laptop hardware (CPU-only support required), identifying members of an office group photo (~10 individuals) against an unseen 50-person reference folder.

Key challenges include:
- Varied face scales (faces in group photos can be as small as 40×40 pixels).
- Non-frontal poses, partial occlusions, varied lighting, and shadows.
- Distinguishing registered individuals from non-registered people ("Unknowns").
- Zero runtime network dependencies.

---

## 2. Face Detection & Recognition Model Comparison

We evaluated four primary technology stacks:

| Framework | Detection Backbone | Embedding Model | Accuracy / Robustness on Group Photos | Inference Speed (CPU) | Verdict |
|---|---|---|---|---|---|
| **dlib / `face_recognition`** | HOG / CNN (MMOD) | dlib ResNet (128-d) | Poor on small faces (<60px), fails on side-profiles | Slow (CNN) / Fast (HOG) | **Rejected**: Poor group photo recall |
| **MediaPipe Face Mesh / Detection** | BlazeFace | None (detection only) | High detection speed, but lacks robust recognition embeddings | Very Fast | **Rejected**: Incomplete solution for recognition |
| **DeepFace (VGG-Face / Facenet)** | OpenCV / MTCNN | FaceNet512 / VGG | Moderate; heavy dependencies (TensorFlow/Keras) | Heavy, slow cold start | **Rejected**: Bulky runtime footprint |
| **InsightFace (buffalo_l)** | SCRFD | ArcFace (w600k_r50, 512-d) | **State-of-the-Art** on small/angled faces and one-shot recognition | Fast via ONNX Runtime | **Selected** |

### Why InsightFace (SCRFD + ArcFace)?
1. **SCRFD (Sample and Computation Redistribution for Face Detection):**
   - Specifically engineered for multi-scale and small face detection.
   - Preserves high recall even when people are standing at different distances from the camera.
   - [Paper: SCRFD (arXiv:2105.04714)](https://arxiv.org/abs/2105.04714)
2. **ArcFace (Additive Angular Margin Loss):**
   - Imposes a geodesic distance margin on a hypersphere, producing highly discriminative 512-dimensional embeddings.
   - Exceptional one-shot verification performance on LFW and MegaFace benchmarks.
   - [Paper: ArcFace (arXiv:1801.07698)](https://arxiv.org/abs/1801.07698)
3. **Self-Contained ONNX Runtime:**
   - InsightFace model packs (`buffalo_l`) are executed via `onnxruntime`, eliminating large framework bloat (e.g. PyTorch/TensorFlow) and running efficiently on standard multi-core laptop CPUs.

---

## 3. Matching Strategy: Bipartite One-to-One Matching

### Rejected: Independent Argmax Matching
- A standard naive approach computes cosine similarities and picks $\arg\max_j \text{sim}(g_i, r_j)$ if $\ge \tau$.
- **Failure Mode:** If two group faces $g_1$ and $g_2$ both resemble registered person $r_k$, both are labeled as $r_k$. In reality, an individual cannot occupy two distinct locations simultaneously in a single photograph.

### Selected: Greedy One-to-One Assignment
1. Compute full pairwise cosine similarity matrix:
   $$\mathbf{S} \in \mathbb{R}^{M \times N}, \quad S_{ij} = \mathbf{e}_{g_i} \cdot \mathbf{e}_{r_j}$$
   *(Since embeddings are L2-normalized, dot product equals cosine similarity).*
2. Extract all candidate pairs with $S_{ij} \ge \tau$.
3. Sort all pairs descending by similarity.
4. Greedily match pairs while maintaining exclusivity sets:
   $$\text{assigned\_faces} \cap \{i\} = \emptyset \quad \text{and} \quad \text{assigned\_identities} \cap \{j\} = \emptyset$$
5. All unassigned faces default to `"Unknown"`.

---

## 4. Test Data Strategy

To rigorously validate the pipeline without manual photography, we utilized the public **Labeled Faces in the Wild (LFW)** dataset:
- 50 individuals selected for the reference folder.
- Group photos synthesized by compositing held-out images of registered individuals alongside external, unregistered individuals ("distractors") across various scales and grid positions.
- [LFW Dataset Link](http://vis-www.cs.umass.edu/lfw/)

---

## 5. References & Links
- [InsightFace Repository](https://github.com/deepinsight/insightface)
- [FastAPI Framework](https://fastapi.tiangolo.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019)](https://arxiv.org/abs/1801.07698)
- [SCRFD: Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714)
