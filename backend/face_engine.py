"""Face detection, embedding extraction, and one-to-one matching engine.

Uses InsightFace buffalo_l (SCRFD detector + ArcFace 512-d embeddings) with CPUExecutionProvider.
"""

from pathlib import Path
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis

MODELS_DIR = Path(__file__).resolve().parent / "models"


class FaceEngine:
    def __init__(self):
        """Initialize InsightFace with detection and recognition modules."""
        MODELS_DIR.mkdir(exist_ok=True)
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        # In-memory storage for reference and group faces
        self.registered_faces: dict[str, np.ndarray] = {}  # {person_name: embedding_512}
        self.group_faces: list[dict] = []  # list of {"bbox", "embedding", "det_score"}
        self.group_image_shape: tuple[int, int] | None = None  # (height, width)

    def clear_registrations(self):
        """Clear all registered reference embeddings."""
        self.registered_faces = {}

    def register_face(self, image_bgr: np.ndarray, person_name: str) -> dict:
        """Register a single person's face from their individual photo.

        Args:
            image_bgr: BGR numpy image of the reference photo.
            person_name: Name of the person derived from filename.

        Returns:
            dict with 'success' (bool) and 'issue' (str or None).
        """
        faces = self.app.get(image_bgr)

        if len(faces) == 0:
            return {"success": False, "issue": "No face detected"}

        if len(faces) > 1:
            # Use largest face by bounding box area
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            self.registered_faces[person_name] = faces[0].embedding
            return {"success": True, "issue": "Multiple faces found — used largest"}

        self.registered_faces[person_name] = faces[0].embedding
        return {"success": True, "issue": None}

    def detect_faces(self, image_bgr: np.ndarray) -> list[dict]:
        """Detect faces and return bounding boxes (Feature 1 compatibility).

        Also extracts and stores embeddings so recognition can run without re-detection.
        """
        h, w = image_bgr.shape[:2]
        self.group_image_shape = (h, w)

        faces = self.app.get(image_bgr)

        self.group_faces = []
        for face in faces:
            self.group_faces.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": face.embedding,
                "det_score": float(face.det_score),
            })

        # Sort faces left-to-right by x1 coordinate for consistent ordering
        self.group_faces.sort(key=lambda f: f["bbox"][0])

        results = []
        for i, face in enumerate(self.group_faces):
            results.append({
                "id": i + 1,
                "bbox": face["bbox"],
                "confidence": round(face["det_score"], 4),
            })

        return results

    def recognize_group(self, image_bgr: np.ndarray, threshold: float = 0.4) -> list[dict]:
        """Detect faces in group photo and match against registered faces."""
        h, w = image_bgr.shape[:2]
        self.group_image_shape = (h, w)

        faces = self.app.get(image_bgr)

        self.group_faces = []
        for face in faces:
            self.group_faces.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": face.embedding,
                "det_score": float(face.det_score),
            })

        # Sort left-to-right
        self.group_faces.sort(key=lambda f: f["bbox"][0])

        return self._match_faces(threshold)

    def re_match(self, threshold: float) -> list[dict]:
        """Re-run matching with a new threshold on already-extracted embeddings."""
        if not self.group_faces:
            return []
        return self._match_faces(threshold)

    def _match_faces(self, threshold: float) -> list[dict]:
        """Core matching logic:

        1. Compute cosine similarity matrix (group faces x registered faces)
        2. Apply threshold filter (candidates >= threshold)
        3. Greedy one-to-one assignment by descending similarity
        4. Mark unassigned group faces as 'Unknown'
        """
        if not self.registered_faces:
            # No registered faces in reference memory — everything is Unknown
            results = []
            for i, gf in enumerate(self.group_faces):
                results.append({
                    "id": i + 1,
                    "name": "Unknown",
                    "confidence": 0.0,
                    "bbox": gf["bbox"],
                    "is_unknown": True,
                })
            return results

        ref_names = list(self.registered_faces.keys())
        ref_embeddings = np.array([self.registered_faces[n] for n in ref_names])

        # Build candidate similarity list
        candidates = []
        for gi, gf in enumerate(self.group_faces):
            g_emb = gf["embedding"]
            g_norm = norm(g_emb)
            if g_norm == 0:
                continue
            for ri, r_emb in enumerate(ref_embeddings):
                r_norm = norm(r_emb)
                if r_norm == 0:
                    continue
                sim = float(np.dot(g_emb, r_emb) / (g_norm * r_norm))
                if sim >= threshold:
                    candidates.append((gi, ri, sim))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Greedy one-to-one assignment:
        # Each group face gets at most one match, each reference person gets at most one face
        assigned_group = set()
        assigned_ref = set()
        matches = {}  # gi -> (name, similarity)

        for gi, ri, sim in candidates:
            if gi not in assigned_group and ri not in assigned_ref:
                matches[gi] = (ref_names[ri], sim)
                assigned_group.add(gi)
                assigned_ref.add(ri)

        # Build final structured response
        results = []
        for i, gf in enumerate(self.group_faces):
            if i in matches:
                name, sim = matches[i]
                results.append({
                    "id": i + 1,
                    "name": name,
                    "confidence": round(sim, 3),
                    "bbox": gf["bbox"],
                    "is_unknown": False,
                })
            else:
                results.append({
                    "id": i + 1,
                    "name": "Unknown",
                    "confidence": 0.0,
                    "bbox": gf["bbox"],
                    "is_unknown": True,
                })

        return results
