"""Build practice data from the public LFW dataset (Labeled Faces in the Wild).

Creates, under test_data/:
  reference_folder/      50 people, one photo each, named "First Last.jpg"
  group_photos/          synthetic group photos (~10 people) built by tiling
                         *different* LFW photos of the same people, plus people
                         who are NOT in the folder, with ground truth in a .json

LFW is downloaded once by scikit-learn (setup-time only; the app itself never
touches the network). Run from the repo root:

    python tools/make_test_data.py
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_data"
SEED = 7
N_REFERENCE = 50
# (file stem, canvas W, canvas H, columns, tile px, in-folder count, unknown count)
GROUPS = [
    ("group_easy", 1600, 900, 5, 280, 8, 2),
    ("group_medium", 2000, 1100, 5, 190, 7, 3),
    ("group_small_faces", 2400, 1300, 5, 130, 7, 3),
]


def to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    if img.max() <= 1.0:
        img = img * 255.0
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)


def make_group(canvas_w, canvas_h, cols, tile, people, rng):
    """people: list of (label_or_None, bgr_image). Returns (canvas, truth)."""
    y, x = np.mgrid[0:canvas_h, 0:canvas_w]
    base = 110 + 50 * (y / canvas_h)[..., None] + 15 * np.sin(x / 90.0)[..., None]
    canvas = np.repeat(base, 3, axis=2).astype(np.float32)
    canvas += rng.normal(0, 4, canvas.shape)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    rows = -(-len(people) // cols)
    cell_w, cell_h = canvas_w // cols, canvas_h // rows
    truth = []
    for i, (name, img) in enumerate(people):
        r, c = divmod(i, cols)
        size = int(tile * rng.uniform(0.85, 1.15))
        face = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        x0 = c * cell_w + int((cell_w - size) / 2 + rng.uniform(-0.15, 0.15) * cell_w)
        y0 = r * cell_h + int((cell_h - size) / 2 + rng.uniform(-0.1, 0.1) * cell_h)
        x0, y0 = max(0, min(x0, canvas_w - size)), max(0, min(y0, canvas_h - size))
        canvas[y0:y0 + size, x0:x0 + size] = face
        truth.append({"name": name or "Unknown", "tile_bbox": [x0, y0, x0 + size, y0 + size]})
    return canvas, truth


def main():
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    print("Fetching LFW dataset for offline test data generation...")
    data = fetch_lfw_people(
        min_faces_per_person=3,
        resize=1.0,
        color=True,
        slice_=(slice(0, 250), slice(0, 250)),
    )
    by_person: dict[int, list[np.ndarray]] = {}
    for img, t in zip(data.images, data.target):
        by_person.setdefault(int(t), []).append(img)
    names = {t: n.replace("_", " ") for t, n in enumerate(data.target_names)}

    people = list(by_person)
    rng.shuffle(people)
    in_folder, outside = people[:N_REFERENCE], people[N_REFERENCE:]

    ref_dir, grp_dir = OUT / "reference_folder", OUT / "group_photos"
    for d in (ref_dir, grp_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            if old.name != ".gitkeep":
                old.unlink()

    # Reference folder: first photo of each person. Group photos use the others.
    for t in in_folder:
        cv2.imwrite(str(ref_dir / f"{names[t]}.jpg"), to_uint8_bgr(by_person[t][0]))
    print(f"Generated {len(in_folder)} reference photos in {ref_dir}")

    for stem, w, h, cols, tile, n_in, n_out in GROUPS:
        chosen_in = rng.sample(in_folder, n_in)
        chosen_out = rng.sample(outside, n_out)
        entries = [(names[t], to_uint8_bgr(rng.choice(by_person[t][1:]))) for t in chosen_in]
        entries += [(None, to_uint8_bgr(rng.choice(by_person[t]))) for t in chosen_out]
        rng.shuffle(entries)
        canvas, truth = make_group(w, h, cols, tile, entries, nrng)
        cv2.imwrite(str(grp_dir / f"{stem}.jpg"), canvas)
        (grp_dir / f"{stem}.json").write_text(json.dumps(truth, indent=2))
        print(f"Generated group image & truth: {stem}")


if __name__ == "__main__":
    main()
