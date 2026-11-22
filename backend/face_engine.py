"""Face detection engine using InsightFace (buffalo_l / SCRFD).

Detects faces in images and returns bounding boxes and detection confidence scores.
"""

from pathlib import Path
import numpy as np
from insightface.app import FaceAnalysis

MODELS_DIR = Path(__file__).resolve().parent / "models"


class FaceEngine:
    def __init__(self):
        """Initialize InsightFace with buffalo_l model pack using CPUExecutionProvider."""
        MODELS_DIR.mkdir(exist_ok=True)
        # Using local models directory under backend/models to guarantee offline execution
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR),
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect_faces(self, image_bgr: np.ndarray) -> list[dict]:
        """Detect faces in an image.

        Args:
            image_bgr: OpenCV image in BGR format.

        Returns:
            List of dicts, each with:
            - "id": int (1-based, ordered left-to-right)
            - "bbox": [x1, y1, x2, y2] as integers (pixel coordinates)
            - "confidence": float (detection confidence score)
        """
        faces = self.app.get(image_bgr)

        results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            results.append({
                "id": i + 1,
                "bbox": bbox,
                "confidence": round(float(face.det_score), 4),
            })

        # Sort faces left-to-right by x1 coordinate for consistent numbering
        results.sort(key=lambda f: f["bbox"][0])

        # Re-number after sorting
        for i, face in enumerate(results):
            face["id"] = i + 1

        return results
