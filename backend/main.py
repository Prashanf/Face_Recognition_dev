"""FastAPI backend server for Face Detection and Recognition.

Supports:
- /api/register-faces : Ingest reference folder with ~50 individual photos
- /api/detect         : Detect faces in group photo (Feature 1)
- /api/recognize      : Run end-to-end recognition & greedy 1-to-1 matching
- /api/re-match       : Instant re-matching on cached group embeddings with new threshold
"""

from pathlib import Path
import re
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from face_engine import FaceEngine

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="Who Is in the Photo? - Face Recognition")
engine = FaceEngine()


def clean_person_name(filename: str) -> str:
    """Extract clean person name from file name (e.g. 'Priya Sharma.jpg' -> 'Priya Sharma')."""
    stem = Path(filename.replace("\\", "/")).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = stem.replace("_", " ")
    return re.sub(r"\s+", " ", stem).strip()


class RematchRequest(BaseModel):
    threshold: float = Field(default=0.40, ge=0.10, le=0.90)


@app.post("/api/register-faces")
async def register_faces(files: list[UploadFile] = File(...)):
    """Register individual reference photos for face recognition.

    Extracts ArcFace 512-d embeddings and clears any previous registrations.
    """
    engine.clear_registrations()

    warnings = []
    registered_names = []
    registered = 0

    for f in files:
        original_name = f.filename or "unknown.jpg"
        clean_name = clean_person_name(original_name)
        file_suffix = Path(original_name).suffix.lower()

        if file_suffix not in IMAGE_EXTENSIONS:
            warnings.append({
                "file": original_name,
                "issue": f"Unsupported file extension '{file_suffix}'",
            })
            continue

        contents = await f.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            warnings.append({
                "file": original_name,
                "issue": "Could not decode image file",
            })
            continue

        result = engine.register_face(image, clean_name)
        if result["success"]:
            registered += 1
            registered_names.append(clean_name)
            if result["issue"]:
                warnings.append({
                    "file": original_name,
                    "issue": result["issue"],
                })
        else:
            warnings.append({
                "file": original_name,
                "issue": result.get("issue", "No face detected"),
            })

    return {
        "registered": registered,
        "total_files": len(files),
        "warnings": warnings,
        "registered_names": registered_names,
    }


@app.post("/api/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in an uploaded group image (Feature 1)."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not read image", "faces": [], "total": 0}

    h, w = image.shape[:2]
    faces = engine.detect_faces(image)

    return {
        "faces": faces,
        "total": len(faces),
        "image_width": w,
        "image_height": h,
    }


@app.post("/api/recognize")
async def recognize_faces(
    file: UploadFile | None = File(None),
    threshold: float = Form(0.40),
):
    """Recognize faces in the group photo and match against registered reference faces."""
    if file is not None:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not read uploaded image")

        h, w = image.shape[:2]
        faces = engine.recognize_group(image, threshold)
    elif engine.group_faces and engine.group_image_shape:
        # Re-use already detected/uploaded group photo
        h, w = engine.group_image_shape
        faces = engine.re_match(threshold)
    else:
        raise HTTPException(
            status_code=400,
            detail="No group photo provided or previously uploaded",
        )

    matched = sum(1 for f in faces if not f["is_unknown"])
    unknown = sum(1 for f in faces if f["is_unknown"])

    return {
        "faces": faces,
        "total_detected": len(faces),
        "matched": matched,
        "unknown": unknown,
        "image_width": w,
        "image_height": h,
    }


@app.post("/api/re-match")
async def re_match_faces(req: RematchRequest):
    """Re-run matching with a new threshold on already extracted embeddings."""
    if not engine.group_faces:
        raise HTTPException(
            status_code=400,
            detail="No group photo faces have been processed yet",
        )

    faces = engine.re_match(req.threshold)
    h, w = engine.group_image_shape if engine.group_image_shape else (0, 0)
    matched = sum(1 for f in faces if not f["is_unknown"])
    unknown = sum(1 for f in faces if f["is_unknown"])

    return {
        "faces": faces,
        "total_detected": len(faces),
        "matched": matched,
        "unknown": unknown,
        "image_width": w,
        "image_height": h,
    }


# Serve frontend static assets (HTML, CSS, JS)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
