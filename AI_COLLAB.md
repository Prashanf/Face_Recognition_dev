# 🤖 AI Collaboration Document (`AI_COLLAB.md`)

## 1. How I Worked With AI

Throughout this project, I used an agentic AI assistant as an active technical collaborator. My core approach was to maintain strict **human ownership** of all architecture, UX, and algorithmic decisions:

1. **Strategic Problem Analysis:** Before generating any code, I analyzed the problem statement, identified the constraints (100% offline runtime, unseen 50-person reference folder, unknown people present in the group photo), and established hard requirements.
2. **Architectural Steering:** Whenever the AI suggested conventional full-stack approaches (e.g. React, Docker, complex databases, or heavy ML frameworks), I steered it toward simpler, zero-build, offline-first architectures (Vanilla JS, FastAPI, local ONNX models).
3. **Challenging Assumptions:** When considering image quality for small faces, I questioned whether super-resolution upscaling was necessary, weighed the AI's explanation regarding internal ArcFace tensor normalization, and decided against artificial upscaling to avoid artifact hallucinations.
4. **Feature-by-Feature Incremental Development:** Instead of requesting a monolithic codebase, I guided development through focused prompts:
   - *Step 1:* Face detection, bounding box visualization, and coordinate scaling.
   - *Step 2:* Folder ingestion, ArcFace embedding extraction, greedy 1-to-1 matching, and interactive results panel.
   - *Step 3:* Hardening against edge cases (corrupted images, non-face photos, duplicate names, ultra-large images).
   - *Step 4:* Empirical testing and threshold verification against test datasets.
   - *Step 5:* Comprehensive documentation suite.

---

## 2. Key Decisions

### Decision 1: Face Detection Backbone Selection
- **AI Suggested:** InsightFace with RetinaFace/SCRFD (`det_10g.onnx`) or MediaPipe.
- **I Decided:** SCRFD via InsightFace (`buffalo_l` pack).
- **Why:** Group photos brought by evaluators are unpredictable. Faces can be small, partially occluded, or turned sideways. MediaPipe and dlib HOG fail on small and profile faces; missing a face during the live demo would mean failing the evaluation.
- **Evidence:** Chat Excerpt #1; comparative research matrix in `RESEARCH.md`.

### Decision 2: Face Representation & Metric Space
- **AI Suggested:** ArcFace 512-dimensional embeddings via ONNX Runtime.
- **I Decided:** Adopted ArcFace ResNet-50 (`w600k_r50.onnx`).
- **Why:** We are given only **one reference image per person**. ArcFace’s additive angular margin loss maximizes inter-class separation, achieving 99.83% accuracy on LFW and leading one-shot verification benchmarks.
- **Evidence:** `backend/face_engine.py` implementation.

### Decision 3: Super-Resolution vs. Direct Affine Normalization
- **AI Suggested:** Skip super-resolution entirely; ArcFace normalizes all crops to $112\times112$ internally before inference. Super-resolution would hallucinate artificial high-frequency textures that degrade cosine similarity.
- **I Decided:** Agreed — rejected super-resolution models.
- **Why:** I initially suspected that scale differences between high-res reference portraits and small group crops would impair accuracy. The AI demonstrated that ArcFace already performs 5-point landmark affine normalization to a standardized $112\times112$ input. Super-resolution would introduce hallucinated artifacts, increase latency, and add another failure point during the demo.
- **Evidence:** Chat Excerpt #2.

### Decision 4: Zero-Build Frontend Architecture
- **AI Suggested:** React/Next.js or Streamlit dashboard.
- **I Decided:** Pure Vanilla HTML5, CSS3, and JavaScript (ES6+), served directly by FastAPI.
- **Why:** The app must run seamlessly on my laptop during the live demo. React requires `npm install`, Node.js runtimes, and bundlers, introducing dependency fragility. Streamlit produces rigid prototypes. Vanilla JS is fast, zero-build, and gives 100% control over the interactive canvas.
- **Evidence:** Clean frontend directory structure with no `package.json` or `node_modules`.

### Decision 5: Folder Selection via Native Browser Standard
- **AI Suggested:** `<input type="file" webkitdirectory>` in the browser.
- **I Decided:** Agreed with `webkitdirectory` multiple upload.
- **Why:** The brief explicitly requires selecting a folder from the UI. Native OS folder dialogs in web applications are restricted without heavy frameworks like Electron. `webkitdirectory` is supported across Chrome, Edge, and Firefox, seamlessly providing all file streams to the backend.
- **Evidence:** `frontend/index.html` folder upload input.

### Decision 6: Greedy Bipartite Matching with Exclusivity
- **AI Suggested:** Cosine similarity matrix with threshold filtering and greedy one-to-one assignment.
- **I Decided:** Agreed, and added a live UI slider for dynamic threshold tuning without re-detection.
- **Why:** The demo photo will deliberately contain people NOT in the folder. Without a threshold, unknown people are forced into incorrect matches. Without one-to-one exclusivity, multiple faces could be assigned to the same employee. The slider enables real-time tuning during the demo.
- **Evidence:** Chat Excerpt #3; `_match_faces` and `re_match` in `backend/face_engine.py`.

### Decision 7: Incremental Feature-by-Feature Delivery
- **AI Suggested:** Generating all backend endpoints and full frontend simultaneously.
- **I Decided:** Enforced a modular, feature-by-feature progression (Detection $\to$ Matching $\to$ Edge-case hardening $\to$ Docs).
- **Why:** Isolating features enabled comprehensive manual testing of detection before layering matching, caught coordinate scaling issues early, and ensured meaningful, reviewable Git commits.
- **Evidence:** Atomic Git commit history (`dd80452`, `5e70d3c`, `0b80820`, `e9ddacd`, etc.).

### Decision 8: Side-by-Side Reference Comparison Modal
- **AI Suggested:** Displaying only the cropped group face thumbnail with a percentage score in the list.
- **I Decided:** Added an interactive "Compare" feature that displays the detected face crop directly alongside the original reference photo in a modal.
- **Why:** Evaluators need immediate visual verification of why the model matched a face, especially under ambiguous lighting or angled poses.
- **Evidence:** `frontend/index.html` modal dialog and `frontend/js/app.js` reference mapping.

### Decision 9: High-DPI 2x Retina Thumbnail Sampling
- **AI Suggested:** Standard 48×48px canvas crops.
- **I Decided:** Sampled thumbnails at 2x resolution (96×96px) rendered down to 48×48px in CSS.
- **Why:** Standard 1x crops appeared pixelated on high-resolution MacBook Retina displays. 2x sampling produces crisp, sharp face avatars in the results list.
- **Evidence:** `cropFace(fullResImage, face.bbox, 96)` in `frontend/js/app.js`.

### Decision 10: Ultra-Large Image Preprocessing & Coordinate Recovery
- **AI Suggested:** Passing raw images directly to InsightFace regardless of dimensions.
- **I Decided:** Implemented an internal threshold to downsample images exceeding 4000px on their longest edge before detection, while mathematically scaling bounding box coordinates back to original dimensions.
- **Why:** High-resolution digital camera photos (e.g. $6000\times4000$) can cause CPU memory spikes and inference delays exceeding 10 seconds. Downscaling during inference and mapping coordinates back preserves speed without sacrificing bounding box accuracy.
- **Evidence:** `_prepare_image` in `backend/face_engine.py`.

---

## 3. Chat Excerpts

### Chat Excerpt #1: Detection Model Selection & Group Photo Constraints
```text
> Me: First I'd like to plan the approach for the app. The evaluators are bringing an unseen folder of 50 people and a real group photo of ~10 people. Some people in the photo won't be in the folder.
> AI: For group photos, face detection accuracy is the primary bottleneck. We have several options:
  1. dlib HOG: Very fast, but frequently misses faces smaller than 60px and side profiles.
  2. MTCNN: Older cascaded CNN, slower on CPU.
  3. MediaPipe: Great for single-person webcams, but lower recall on distant crowd faces.
  4. InsightFace (SCRFD / RetinaFace): State-of-the-art multi-scale face detector, handles dense groups and non-frontal angles exceptionally well.
  I recommend InsightFace with the buffalo_l pack.
> Me: Agreed. In a live evaluation, missing a face entirely is an immediate failure. Let's use InsightFace SCRFD so we don't miss small or angled faces.
```

### Chat Excerpt #2: The Super-Resolution Discussion
```text
> Me: The individual reference photos will have much larger faces than the small crops extracted from a 10-person group photo. Won't that size mismatch hurt accuracy? Should we add a super-resolution model to upscale the bounding box crops to 85-90% before matching?
> AI: While the size difference is real, adding a super-resolution model is unnecessary and counterproductive:
  1. ArcFace already normalizes all face crops to a fixed 112×112 tensor using 5 facial landmarks before computing embeddings. It does this identically for both reference photos and group photo crops.
  2. Super-resolution neural nets hallucinate high-frequency details — they invent pixel textures that were not in the camera capture. Feeding ArcFace fabricated pixels can actually reduce cosine similarity and hurt accuracy.
  3. It would add significant latency on CPU and create an additional failure point.
> Me: That makes total sense. We will skip super-resolution and let ArcFace's native landmark alignment handle the 112×112 normalization.
```

### Chat Excerpt #3: Greedy One-to-One Bipartite Matching
```text
> Me: What happens if two people in the group photo look similar to the same person in the folder?
> AI: If we use independent argmax matching, both group faces could be assigned to that same reference person.
> Me: That's physically impossible in one photograph — an employee can only appear once in an office photo. We need greedy one-to-one matching: sort all valid similarity pairs descending, assign the best pair, and lock both the person and the face so neither can be reused. Unmatched faces must be marked "Unknown".
> AI: Exactly. We'll compute the M×N cosine similarity matrix, filter by threshold (default 0.40), sort candidate triples descending, and greedily claim identities.
```

---

## 4. Full AI Chat History

The complete, unedited transcripts and execution logs across all development phases are archived in this repository:
- **`chat-history/`**: Contains raw conversation transcripts and exported prompt iterations starting from the initial problem statement analysis.
- **`dev_docs/`**: Contains the modular feature prompts (`step1-face-detection-prompt.md`, `step2-recognition-prompt.md`, `step3-and-step5-prompt.md`).
