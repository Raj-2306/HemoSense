"""
prepare_data.py  --  HemoSense (educational prototype, NOT for clinical use)

Step 2: turn data/unique/<class folders> into a leakage-safe split file.

What it does
  * reads the de-duplicated images and normalises labels ("A negative" -> "A-")
  * groups exact + near-duplicate images (average-hash Hamming distance) so a
    group is NEVER split across folds
  * builds stratified, grouped K-fold splits (tiny dataset -> cross-validation
    instead of one 70/15/15 split)
  * derives per-tube labels from the class label (Anti-A / Anti-B / Anti-D /
    control) using the standard ABO/Rh rule
  * saves everything to data/splits/folds.csv and prints a summary

Run:
    python src/prepare_data.py

Requires: pillow numpy pandas scikit-learn
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------- CONFIG ---------------------------------------
DATA_DIR = Path("data/unique")
SPLIT_DIR = Path("data/splits")
OUT_CSV = SPLIT_DIR / "folds.csv"
N_FOLDS = 4          # A- has only 4 unique images, so 4 is the practical maximum
SEED = 42
HASH_SIZE = 8        # average hash: 8x8 = 64 bits
MAX_HAMMING = 4      # hashes within this many differing bits = "same group"
EXPECTED_LABELS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# ----------------------------------------------------------------------------


def normalize_label(name: str) -> str:
    s = name.upper().replace("POSITIVE", "+").replace("NEGATIVE", "-")
    s = s.replace("POS", "+").replace("NEG", "-")
    return re.sub(r"[\s_]+", "", s)


def average_hash(path: Path) -> int:
    with Image.open(path) as im:
        small = im.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_groups(hashes, labels):
    """Union-find: images with the same label and near-identical hash share a group."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j] and hamming(hashes[i], hashes[j]) <= MAX_HAMMING:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]


def tube_labels(label: str) -> dict:
    """Standard forward-typing rule. Control (Anti-AB) reacts for A, B and AB, not O."""
    abo = label[:-1]
    return dict(
        anti_a=int(abo in ("A", "AB")),
        anti_b=int(abo in ("B", "AB")),
        anti_d=int(label.endswith("+")),
        control=int(abo != "O"),
    )


def main():
    files = sorted(p for p in DATA_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"[ERROR] No images in {DATA_DIR.resolve()}. Run dedupe_report.py first.")
        return

    labels = [normalize_label(p.parent.name) for p in files]
    bad = sorted(set(labels) - set(EXPECTED_LABELS))
    if bad:
        print(f"[ERROR] Unexpected labels: {bad}")
        return

    hashes = [average_hash(p) for p in files]
    groups = build_groups(hashes, labels)

    df = pd.DataFrame({"path": [str(p) for p in files], "label": labels, "group": groups})
    for key in ("anti_a", "anti_b", "anti_d", "control"):
        df[key] = [tube_labels(l)[key] for l in labels]

    print(f"Images: {len(df)} | groups after near-duplicate merging: {df.group.nunique()}")
    print("\nGroups per class (a class needs >= N_FOLDS groups to appear in every fold):")
    gpc = df.groupby("label").group.nunique().reindex(EXPECTED_LABELS)
    print(gpc.to_string())
    short = gpc[gpc < N_FOLDS]
    if len(short):
        print(f"\n[WARNING] Classes with fewer than {N_FOLDS} groups: {list(short.index)}."
              " Some folds will have none of these classes; interpret their results with great care.")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for fold, (_, test_idx) in enumerate(sgkf.split(df, df["label"], df["group"])):
        df.loc[test_idx, "fold"] = fold

    # leakage check: every group must live in exactly one fold
    assert (df.groupby("group").fold.nunique() == 1).all(), "A group spans several folds!"

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\nImages per class per fold (each fold is the TEST set once):")
    print(pd.crosstab(df.label, df.fold).reindex(EXPECTED_LABELS).to_string())
    print("\nPer-tube positives (derived from the class label):")
    print(df[["anti_a", "anti_b", "anti_d", "control"]].agg(["sum", "count"]).to_string())
    print(f"\nSaved: {OUT_CSV.resolve()}")
    print("Rule for later steps: never tune on the fold you are testing; fix the recipe first.")
    print("Educational prototype -- not for clinical use.")


if __name__ == "__main__":
    main()