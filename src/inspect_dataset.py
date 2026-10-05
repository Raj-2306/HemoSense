"""
inspect_dataset.py  --  HemoSense (educational prototype, NOT for clinical use)

Step 1 of the pipeline: inspect a dataset BEFORE training anything.

What it does
  * finds images and their labels (class folders, or an optional CSV)
  * checks labels against the 8 expected ABO/Rh classes
  * counts classes and reports imbalance
  * checks formats, resolutions, colour modes
  * detects corrupt files
  * detects exact duplicates (MD5) and possible near-duplicates (average hash)
  * flags the same image appearing under DIFFERENT labels
  * saves a class-distribution chart and a sample grid for VISUAL review
  * prints a suitability checklist

Run:
    python src/inspect_dataset.py
    python src/inspect_dataset.py --data-dir data/raw --out-dir results/dataset_report
    python src/inspect_dataset.py --csv labels.csv   (columns: filename,label)

Requires: pillow numpy pandas matplotlib
"""

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# ----------------------------- CONFIG ---------------------------------------
DATA_DIR = Path("data/raw")
OUT_DIR = Path("results/dataset_report")
CSV_PATH = None                       # e.g. Path("data/raw/labels.csv")
CSV_FILE_COL = "filename"
CSV_LABEL_COL = "label"
EXPECTED_LABELS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SAMPLES_PER_CLASS = 4
IMBALANCE_WARN_RATIO = 2.0            # warn if largest/smallest class > this
MIN_IMAGES_PER_CLASS = 100            # rough minimum for a student project
HASH_SIZE = 8                         # average-hash size (8x8 = 64 bits)
# ----------------------------------------------------------------------------


def normalize_label(name: str) -> str:
    """Map folder names like 'A_positive', 'ab neg', 'O-' to 'A+', 'AB-', 'O-'."""
    s = str(name).upper()
    s = s.replace("POSITIVE", "+").replace("NEGATIVE", "-")
    s = s.replace("POS", "+").replace("NEG", "-")
    return re.sub(r"[\s_]+", "", s)


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(img: Image.Image) -> str:
    small = img.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def collect_records(data_dir: Path, csv_path):
    """Return list of (path, raw_label)."""
    records = []
    if csv_path:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            p = Path(csv_path).parent / str(row[CSV_FILE_COL])
            records.append((p, str(row[CSV_LABEL_COL])))
    else:
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                records.append((p, p.parent.name))   # label = immediate folder
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.data_dir, args.csv)
    if not records:
        print(f"[ERROR] No images found in {args.data_dir.resolve()}.")
        print("Put images in data/raw/<class_name>/ or pass --csv.")
        return

    rows, corrupt = [], []
    for path, raw_label in records:
        if not path.exists():
            corrupt.append((path, "missing file"))
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:          # reopen after verify()
                w, h = im.size
                mode, fmt = im.mode, im.format
                ahash = average_hash(im)
                arr = np.asarray(im.convert("RGB"), dtype=np.float32)
            # how colourful is the image? (0 = pure greyscale)
            colourfulness = float(np.abs(arr[..., 0] - arr[..., 1]).mean()
                                  + np.abs(arr[..., 1] - arr[..., 2]).mean())
            rows.append(dict(path=str(path), raw_label=raw_label,
                             label=normalize_label(raw_label),
                             width=w, height=h, mode=mode, format=fmt,
                             md5=md5_of_file(path), ahash=ahash,
                             colourfulness=round(colourfulness, 2)))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            corrupt.append((path, f"{type(e).__name__}: {e}"))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "image_inventory.csv", index=False)
    print("=" * 70)
    print("HEMOSENSE DATASET INSPECTION  (educational prototype)")
    print("=" * 70)
    print(f"Images found: {len(records)} | readable: {len(df)} | corrupt/missing: {len(corrupt)}")
    for p, why in corrupt[:10]:
        print(f"  [CORRUPT] {p}  ({why})")
    if df.empty:
        return

    # ---- labels -----------------------------------------------------------
    print("\n--- LABELS ---")
    found = sorted(df["label"].unique())
    print(f"Classes found ({len(found)}): {found}")
    missing = [l for l in EXPECTED_LABELS if l not in found]
    unexpected = [l for l in found if l not in EXPECTED_LABELS]
    print(f"Missing expected labels : {missing or 'none'}")
    print(f"Unexpected labels       : {unexpected or 'none'}")

    counts = df["label"].value_counts().sort_index()
    print("\nImages per class:")
    print(counts.to_string())
    ratio = counts.max() / max(counts.min(), 1)
    print(f"Imbalance ratio (max/min): {ratio:.2f}")

    # ---- formats / resolution --------------------------------------------
    print("\n--- FORMAT & RESOLUTION ---")
    print("Formats :", dict(Counter(df["format"])))
    print("Modes   :", dict(Counter(df["mode"])))
    print(f"Width   : min {df.width.min()}  median {int(df.width.median())}  max {df.width.max()}")
    print(f"Height  : min {df.height.min()}  median {int(df.height.median())}  max {df.height.max()}")
    print(f"Unique sizes: {df[['width','height']].drop_duplicates().shape[0]}")
    print(f"Mean colourfulness: {df.colourfulness.mean():.1f} "
          "(near 0 = greyscale-like images, e.g. scanned fingerprints)")

    # ---- duplicates / leakage --------------------------------------------
    print("\n--- DUPLICATES & LEAKAGE RISK ---")
    exact = df[df.duplicated("md5", keep=False)]
    n_exact_extra = len(df) - df["md5"].nunique()
    print(f"Exact duplicate files (extra copies): {n_exact_extra}")
    near = df[df.duplicated("ahash", keep=False)]
    n_near_extra = len(df) - df["ahash"].nunique()
    print(f"Possible near-duplicates by average hash (extra copies): {n_near_extra} "
          "(can include false positives for low-detail images)")

    conflicts = []
    for col in ("md5", "ahash"):
        g = df.groupby(col)["label"].nunique()
        conflicts += list(g[g > 1].index)
    print(f"Hash groups appearing under DIFFERENT labels: {len(conflicts)}")
    if n_exact_extra or n_near_extra:
        print("  -> Duplicates must stay in the SAME split, or be removed, "
              "otherwise test scores are inflated (data leakage).")
    pd.concat([exact, near]).drop_duplicates("path").to_csv(
        args.out_dir / "duplicate_candidates.csv", index=False)

    # ---- plots ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Class distribution (selected dataset)")
    ax.set_ylabel("Number of images")
    plt.tight_layout()
    fig.savefig(args.out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    labels_sorted = list(counts.index)
    fig, axes = plt.subplots(len(labels_sorted), SAMPLES_PER_CLASS,
                             figsize=(2.2 * SAMPLES_PER_CLASS, 2.2 * len(labels_sorted)),
                             squeeze=False)
    rng = np.random.default_rng(42)
    for r, lab in enumerate(labels_sorted):
        sub = df[df.label == lab]
        pick = sub.sample(min(SAMPLES_PER_CLASS, len(sub)), random_state=int(rng.integers(1e6)))
        for c in range(SAMPLES_PER_CLASS):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(pick):
                ax.imshow(Image.open(pick.iloc[c]["path"]).convert("RGB"))
                if c == 0:
                    ax.set_title(lab, fontsize=9, loc="left")
    plt.tight_layout()
    fig.savefig(args.out_dir / "sample_grid.png", dpi=120)
    plt.close(fig)

    # ---- suitability checklist -------------------------------------------
    print("\n--- SUITABILITY CHECKLIST ---")
    checks = [
        ("All 8 expected labels present", not missing),
        ("No unexpected labels", not unexpected),
        ("No corrupt images", not corrupt),
        (f"Each class has >= {MIN_IMAGES_PER_CLASS} images",
         bool((counts >= MIN_IMAGES_PER_CLASS).all())),
        (f"Imbalance ratio <= {IMBALANCE_WARN_RATIO}", ratio <= IMBALANCE_WARN_RATIO),
        ("No exact duplicates", n_exact_extra == 0),
        ("No cross-label duplicate hashes", not conflicts),
    ]
    for text, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {text}")

    print("\n  [MANUAL] Open sample_grid.png and decide what the images ACTUALLY show:")
    print("           blood-reaction zones (Anti-A/B/D)?  fingerprints?  blood smears?  other?")
    print("  [MANUAL] Read the dataset page: licence, source, how labels were produced.")
    print("\nIMPORTANT: If images are NOT blood-reaction images (e.g. fingerprints or smears),")
    print("they are not valid evidence for blood typing. Do not force the project onto them;")
    print("use simulated Anti-A/Anti-B/Anti-D reaction-card images instead.")
    print("Educational prototype -- not for clinical use.")
    print(f"\nReports saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
