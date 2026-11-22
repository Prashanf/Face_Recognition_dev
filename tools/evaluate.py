"""Score recognition on the generated test data (see make_test_data.py).

A detected face is scored against the ground-truth tile that contains its centre.
Faces outside every tile (bystanders inside LFW crops) are ignored.

    python tools/evaluate.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from face_engine import FaceEngine, name_from_filename  # noqa: E402

THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]


def tile_for(face, truth):
    cx, cy = (face["bbox"][0] + face["bbox"][2]) / 2, (face["bbox"][1] + face["bbox"][3]) / 2
    for t in truth:
        x1, y1, x2, y2 = t["tile_bbox"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return t
    return None


def main():
    engine = FaceEngine()
    for p in sorted((ROOT / "test_data/reference_folder").glob("*.jpg")):
        engine.register_face(p.read_bytes(), name_from_filename(p.name), p.name)
    print(f"registered {len(engine.registered_faces)} people\n")

    groups = sorted((ROOT / "test_data/group_photos").glob("*.jpg"))
    print(f"{'threshold':>9} | correct  wrong-name  missed  false-match | (over tile faces)")
    for th in THRESHOLDS:
        tot = dict(correct=0, wrong=0, missed=0, false=0, tiles=0)
        for g in groups:
            truth = json.loads(g.with_suffix(".json").read_text())
            faces = engine.recognize_group(g.read_bytes(), th)
            in_tile = {id(t): [] for t in truth}
            for f in faces:
                t = tile_for(f, truth)
                if t is not None:
                    in_tile[id(t)].append(f)
            # A tile can hold bystander faces too, so judge the tile by all faces inside it.
            for t in truth:
                fs = in_tile[id(t)]
                named = [f["name"] for f in fs if not f["is_unknown"]]
                tot["tiles"] += 1
                if t["name"] == "Unknown":
                    tot["false" if named else "correct"] += 1
                elif t["name"] in named:
                    tot["correct"] += 1
                elif named:
                    tot["wrong"] += 1
                else:
                    tot["missed"] += 1
        print(f"{th:>9} | {tot['correct']:>7}  {tot['wrong']:>10}  {tot['missed']:>6}  {tot['false']:>11} | of {tot['tiles']}")


if __name__ == "__main__":
    main()
