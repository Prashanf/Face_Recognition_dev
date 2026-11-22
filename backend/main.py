"""FastAPI backend server for Face Detection in group photos."""

from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from face_engine import FaceEngine

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="Who Is in the Photo? - Face Detection")
engine = FaceEngine()


@app.post("/api/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in an uploaded image and return bounding boxes."""
    contents = await file.read()

    # Decode image from buffer
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not read image", "faces": [], "total": 0}

    # Get image dimensions for the frontend coordinate scaling
    h, w = image.shape[:2]

    # Detect faces
    faces = engine.detect_faces(image)

    return {
        "faces": faces,
        "total": len(faces),
        "image_width": w,
        "image_height": h,
    }


# Serve frontend static assets (HTML, CSS, JS)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
