"""FastAPI backend server for Face Detection and Recognition.

Hardened against all production edge cases (Step 3):
- /api/register-faces : Ingest reference folder with error suppression, duplicate detection & name cleanup
- /api/detect         : Detect faces in group photo with non-image guards & friendly empty states
- /api/recognize      : Run end-to-end recognition & greedy 1-to-1 matching with state guards
- /api/re-match       : Instant re-matching on cached group embeddings with new threshold
- /api/reset          : Reset all server-side session data and embeddings
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

# Supported image file extensions for reference and group photos
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")
# Ignored non-image system / text files
IGNORED_FILES = {".ds_store", "thumbs.db", "desktop.ini"}

app = FastAPI(title="Who Is in the Photo? - Face Recognition")
engine = FaceEngine()


def clean_person_name(filename: str) -> str:
    """Extract clean person name from file name (Edge Case 6).

    Examples:
    - 'Priya_Sharma.jpg' -> 'Priya Sharma'
    - 'Priya Sharma (1).jpg' -> 'Priya Sharma'
    - '  rahul_verma.PNG ' -> 'rahul verma'
    - '.jpg' -> ''
    """
    path = Path(filename.replace("\\", "/"))
    name = path.name.strip()

    # Strip supported extension case-insensitively
    for ext in IMAGE_EXTENSIONS:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    # Strip duplicate/copy number patterns like (1), (2), etc.
    name = re.sub(r"\s*\(\d+\)$", "", name)
    # Replace underscores with spaces
    name = name.replace("_", " ")
    # Collapse multiple whitespaces and strip leading/trailing spaces
    cleaned = re.sub(r"\s+", " ", name).strip()
    return cleaned


class RematchRequest(BaseModel):
    threshold: float = Field(default=0.40, ge=0.10, le=0.90)


@app.post("/api/register-faces")
async def register_faces(files: list[UploadFile] = File(...)):
    """Register individual reference photos for face recognition.

    Edge Cases Handled:
    1. No face detected in photo -> skipped with warning.
    2. Multiple faces in photo -> largest used with warning.
    3. Corrupted / unreadable image -> skipped with warning.
    4. Non-image files (.txt, .DS_Store, etc.) -> silently skipped.
    5. Duplicate names -> latest replaces earlier with warning.
    6. Filename sanitization -> extensions stripped, underscores replaced with spaces.
    7. Empty folder -> returns clear error response.
    8. Large folders -> efficient processing without memory blowup.
    """
    engine.clear_registrations()

    warnings: list[dict] = []
    registered_names: list[str] = []
    registered = 0
    valid_images_count = 0

    for f in files:
        original_name = f.filename or "unknown"
        base_name = Path(original_name.replace("\\", "/")).name

        # Silently skip operating system junk files
        if base_name.lower() in IGNORED_FILES:
            continue

        file_suffix = Path(base_name).suffix.lower()

        # Edge Case 4: Skip non-image files
        if not any(base_name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue

        valid_images_count += 1
        clean_name = clean_person_name(base_name)

        # Edge Case 6: Empty name after cleanup
        if not clean_name:
            warnings.append({
                "file": base_name,
                "issue": f"Could not derive a valid person name from '{base_name}' — skipped",
            })
            continue

        contents = await f.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Edge Case 3: Corrupted or unreadable image
        if image is None:
            warnings.append({
                "file": base_name,
                "issue": f"Could not read '{base_name}' — skipped",
            })
            continue

        # Edge Case 5: Duplicate person name
        is_duplicate = clean_name in engine.registered_faces

        result = engine.register_face(image, clean_name, base_name)
        if result["success"]:
            registered += 1
            if clean_name not in registered_names:
                registered_names.append(clean_name)

            if is_duplicate:
                warnings.append({
                    "file": base_name,
                    "issue": f"Duplicate name '{clean_name}' — used latest",
                })
            elif result["issue"]:
                warnings.append({
                    "file": base_name,
                    "issue": result["issue"],
                })
        else:
            # Edge Case 1: No face detected
            warnings.append({
                "file": base_name,
                "issue": result.get(
                    "issue", f"No face detected in '{base_name}' — skipped"
                ),
            })

    # Edge Case 7: Empty folder or no valid images
    if registered == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid images found in the folder",
        )

    return {
        "registered": registered,
        "total_files": valid_images_count,
        "warnings": warnings,
        "registered_names": registered_names,
    }


@app.post("/api/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in an uploaded group image (Feature 1).

    Edge cases handled:
    - Non-photo / unreadable files (Edge Case 11)
    - Zero faces detected (Edge Case 9)
    - Ultra-large images (Edge Case 10)
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Edge Case 11: Unreadable file
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read this file. Please upload a JPG, PNG, or WebP image.",
        )

    h, w = image.shape[:2]
    faces = engine.detect_faces(image)

    # Edge Case 9: Zero faces detected
    message = None
    if len(faces) == 0:
        message = (
            "No faces detected in this photo. Try a clearer photo or one with"
            " people facing the camera."
        )

    return {
        "faces": faces,
        "total": len(faces),
        "image_width": w,
        "image_height": h,
        "message": message,
    }


@app.post("/api/recognize")
async def recognize_faces(
    file: UploadFile | None = File(None),
    threshold: float = Form(0.40),
):
    """Recognize faces in the group photo and match against registered reference faces.

    Edge cases handled:
    - Attempted before group photo uploaded (Edge Case 13)
    - Attempted before folder uploaded (Edge Case 12) -> returns all as 'Unknown'
    - Non-photo / unreadable group image (Edge Case 11)
    - Zero faces detected (Edge Case 9)
    """
    if file is not None:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Edge Case 11: Non-image file uploaded as group photo
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not read this file. Please upload a JPG, PNG, or WebP image.",
            )

        h, w = image.shape[:2]
        faces = engine.recognize_group(image, threshold)
    elif engine.group_faces and engine.group_image_shape:
        # Re-use already detected/uploaded group photo
        h, w = engine.group_image_shape
        faces = engine.re_match(threshold)
    else:
        # Edge Case 13: Attempted recognition before group photo is provided
        raise HTTPException(
            status_code=400,
            detail="No group photo provided or previously uploaded",
        )

    matched = sum(1 for f in faces if not f["is_unknown"])
    unknown = sum(1 for f in faces if f["is_unknown"])

    # Edge Case 9 & 12 messages
    message = None
    if len(faces) == 0:
        message = (
            "No faces detected in this photo. Try a clearer photo or one with"
            " people facing the camera."
        )
    elif len(engine.registered_faces) == 0:
        message = (
            "No reference folder uploaded — all detected faces marked as"
            " Unknown."
        )

    return {
        "faces": faces,
        "total_detected": len(faces),
        "matched": matched,
        "unknown": unknown,
        "image_width": w,
        "image_height": h,
        "message": message,
    }


@app.post("/api/re-match")
async def re_match_faces(req: RematchRequest):
    """Re-run matching with a new threshold on already extracted embeddings.

    Edge Case 14: Handles re-match call when no group photo exists gracefully.
    """
    if not engine.group_faces:
        return {
            "faces": [],
            "total_detected": 0,
            "matched": 0,
            "unknown": 0,
            "image_width": 0,
            "image_height": 0,
            "message": "No group photo has been processed yet.",
        }

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


@app.post("/api/reset")
async def reset():
    """Reset all session data, embeddings and photo caches (Edge Case 21 & 25)."""
    engine.reset_all()
    return {
        "status": "reset",
        "message": "All server state has been reset successfully",
    }


# Serve frontend static assets (HTML, CSS, JS)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
