"""
extract_tubes.py (v3)  --  HemoSense (educational prototype, NOT for clinical use)

Find the four reaction spots per image, crop each one, build a per-tube table.
Normal layout (left -> right): Anti-A, Anti-B, control (Anti-AB), Anti-D.

Changes in v3
  * Crops are cut from each tube's OWN region (never crossing into a neighbour)
    and letterboxed to a square with the background colour, so a crop of one
    tube no longer contains parts of the next tube.
  * Some slides are photographed reversed (rotated 180 deg). Reagents are
    colour-coded: the Anti-A tube has a cyan/blue tint. If the cyan tube is on
    the RIGHT, the image is treated as reversed (names assigned right->left and
    the crop rotated 180 deg). Untinted images cannot be checked this way, so
    use MANUAL_REVERSED / MANUAL_NORMAL after looking at the review sheets.
  * A second sheet (crops_sheet.jpg) shows the actual crops for review.

Run:  python src/extract_tubes.py
Requires: opencv-python numpy pandas
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------------
FOLDS_CSV = Path("data/splits/folds.csv")
OUT_DIR = Path("data/tubes")
OUT_CSV = Path("data/splits/tubes.csv")
DEBUG_DIR = Path("results/tube_debug")
TUBE_NAMES = ["anti_a", "anti_b", "control", "anti_d"]   # normal order, left -> right
TUBE_SIZE = 224
PAD_FRAC = 0.15                 # margin around each tube (share of its size)
MIN_COMP_AREA = 60
MIN_GROUP_AREA = 400
GAPS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
STRONG_MIN, WEAK_MIN, WEAK_FACTOR = 12.0, 5.0, 0.4
OPEN_STRONG, OPEN_WEAK = 3, 5
MAX_SPACING_RATIO = 2.5
MAX_Y_SPREAD = 0.30
# cyan (Anti-A reagent) detection, OpenCV HSV: H 0-179, S/V 0-255
CYAN_H = (75, 105)
CYAN_MIN_S, CYAN_MIN_V = 25, 80
CYAN_TUBE_FRAC = 0.10           # share of a tube's box that must be cyan
# manual overrides by image index (the "#number" shown on the review sheets)
MANUAL_REVERSED = {27, 28, 82, 86, 89}
EXCLUDE_IMAGES = {75, 81, 84}    # label conflicts with A+ look-alikes   # e.g. {27, 28}
MANUAL_NORMAL = set()           # force normal order even if cyan says reversed
SHEET_COLS, SHEET_PER, SHEET_W = 4, 16, 480
CROP_SHEET_ROWS, CROP_THUMB = 12, 128
# ----------------------------------------------------------------------------


def read_bgr(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def chroma_dist(img):
    w = img.shape[1]
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=max(1.0, w / 400))
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1], lab[..., 2]
    dist = np.sqrt((a - np.median(a)) ** 2 + (b - np.median(b)) ** 2)
    d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return dist, otsu / 4.0


def components(mask):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    return sorted((int(st[i, 0]), int(st[i, 0] + st[i, 2]), int(st[i, 1]),
                   int(st[i, 1] + st[i, 3]), int(st[i, 4]))
                  for i in range(1, n) if st[i, 4] >= MIN_COMP_AREA)


def merge_x(comps, gap_px):
    groups = []
    for x0, x1, y0, y1, a in comps:
        if groups and x0 - groups[-1]["x1"] <= gap_px:
            g = groups[-1]
            g["x1"] = max(g["x1"], x1)
            g["y0"], g["y1"] = min(g["y0"], y0), max(g["y1"], y1)
            g["area"] += a
        else:
            groups.append(dict(x0=x0, x1=x1, y0=y0, y1=y1, area=a))
    return groups


def sane(groups, h):
    cx = [(g["x0"] + g["x1"]) / 2 for g in groups]
    cy = [(g["y0"] + g["y1"]) / 2 for g in groups]
    d = np.diff(cx)
    if d.min() <= 0 or d.max() / d.min() > MAX_SPACING_RATIO:
        return False
    return (max(cy) - min(cy)) <= MAX_Y_SPREAD * h


def pick_four(comps, w, h):
    for frac in GAPS:
        groups = [g for g in merge_x(comps, int(w * frac)) if g["area"] >= MIN_GROUP_AREA]
        if len(groups) == 4 and sane(groups, h):
            return groups
    return None


def find_four(img):
    h, w = img.shape[:2]
    dist, otsu = chroma_dist(img)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=max(1.0, w / 400))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cyan = ((hsv[..., 0] >= CYAN_H[0]) & (hsv[..., 0] <= CYAN_H[1]) &
            (hsv[..., 1] >= CYAN_MIN_S) & (hsv[..., 2] >= CYAN_MIN_V)).astype(np.uint8) * 255
    counts = {}
    for mode, thr, k in (("strong", max(otsu, STRONG_MIN), OPEN_STRONG),
                         ("weak", max(WEAK_FACTOR * otsu, WEAK_MIN), OPEN_WEAK)):
        mask = (dist > thr).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, cyan)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        comps = components(mask)
        groups = pick_four(comps, w, h)
        if groups:
            return groups, mode, counts
        counts[mode] = len([g for g in merge_x(comps, int(w * 0.05))
                            if g["area"] >= MIN_GROUP_AREA])
    return None, None, counts

def cyan_fractions(img, groups):
    out = []
    for g in groups:
        crop = img[g["y0"]:g["y1"], g["x0"]:g["x1"]]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 0] >= CYAN_H[0]) & (hsv[..., 0] <= CYAN_H[1]) &
             (hsv[..., 1] >= CYAN_MIN_S) & (hsv[..., 2] >= CYAN_MIN_V))
        out.append(float(m.mean()) if m.size else 0.0)
    return out


def decide_reversed(idx, fracs):
    """Return (reversed?, reason)."""
    if idx in MANUAL_REVERSED:
        return True, "manual"
    if idx in MANUAL_NORMAL:
        return False, "manual"
    left, right = fracs[0] >= CYAN_TUBE_FRAC, fracs[3] >= CYAN_TUBE_FRAC
    if right and not left:
        return True, "cyan on right"
    if left and not right:
        return False, "cyan on left"
    return False, "no colour cue"


def background_colour(img):
    return tuple(int(v) for v in np.median(img.reshape(-1, 3), axis=0))


def crop_tube(img, g, left, right, bg):
    h, w = img.shape[:2]
    side = max(g["x1"] - g["x0"], g["y1"] - g["y0"])
    m = int(PAD_FRAC * side)
    x0, x1 = max(g["x0"] - m, left, 0), min(g["x1"] + m, right, w)
    y0, y1 = max(g["y0"] - m, 0), min(g["y1"] + m, h)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = max(ch, cw)
    canvas = np.full((s, s, 3), bg, np.uint8)
    oy, ox = (s - ch) // 2, (s - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return cv2.resize(canvas, (TUBE_SIZE, TUBE_SIZE), interpolation=cv2.INTER_AREA), (x0, y0, x1, y1)


def overlay(img, rects, names, title, ok):
    vis = img.copy()
    colour = (0, 160, 0) if ok else (0, 0, 255)
    for (x0, y0, x1, y1), name in zip(rects, names):
        cv2.rectangle(vis, (x0, y0), (x1, y1), colour, 4)
        cv2.putText(vis, name, (x0 + 4, max(30, y0 + 34)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)
    cv2.putText(vis, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
    sc = SHEET_W / vis.shape[1]
    return cv2.resize(vis, (SHEET_W, int(vis.shape[0] * sc)))


def make_sheet(tiles, cols):
    hmax = max(t.shape[0] for t in tiles)
    wmax = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        chunk = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, wmax - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for t in tiles[i:i + cols]]
        while len(chunk) < cols:
            chunk.append(np.full((hmax, wmax, 3), 255, np.uint8))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


def main():
    if not FOLDS_CSV.exists():
        print(f"[ERROR] {FOLDS_CSV} not found. Run prepare_data.py first.")
        return
    df = pd.read_csv(FOLDS_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    rows, failed, tiles, modes, reversed_list, crop_rows = [], [], [], [], [], []
    for idx, r in df.iterrows():
        path = Path(r["path"])
        tag = f"#{idx} {r['label']} {path.name[:8]}"
        if idx in EXCLUDE_IMAGES:
            continue
        img = read_bgr(path)
        if img is None:
            failed.append((path.name, "unreadable"))
            continue
        groups, mode, counts = find_four(img)
        if groups is None:
            failed.append((path.name, f"no valid 4-tube layout (groups found: {counts})"))
            tiles.append(overlay(img, [], [], "FAIL " + tag, False))
            continue
        modes.append(mode)
        fracs = cyan_fractions(img, groups)
        rev, why = decide_reversed(idx, fracs)
        if rev:
            reversed_list.append((idx, path.name, r["label"], why))
        names = TUBE_NAMES[::-1] if rev else TUBE_NAMES

        w = img.shape[1]
        bg = background_colour(img)
        lefts = [0] + [(groups[i - 1]["x1"] + groups[i]["x0"]) // 2 for i in range(1, 4)]
        rights = lefts[1:] + [w]
        rects, crops = [], []
        for i, g in enumerate(groups):
            crop, rect = crop_tube(img, g, lefts[i], rights[i], bg)
            if rev:
                crop = cv2.rotate(crop, cv2.ROTATE_180)
            name = names[i]
            tube_path = OUT_DIR / f"{path.stem}_{i + 1}_{name}.png"
            cv2.imwrite(str(tube_path), crop)
            rects.append(rect)
            crops.append((name, crop))
            rows.append(dict(tube_path=str(tube_path), source_path=str(path),
                             image_index=idx, position=i + 1, tube=name,
                             reacted=int(r[name]), label=r["label"],
                             group=r["group"], fold=r["fold"], mode=mode,
                             reversed=int(rev)))
        flag = " REV" if rev else ""
        tiles.append(overlay(img, rects, names, f"{mode}{flag} {tag}", True))
        crop_rows.append((tag, sorted(crops, key=lambda c: TUBE_NAMES.index(c[0]))))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    for old in DEBUG_DIR.glob("review_*.jpg"):
        old.unlink()
    for s in range(0, len(tiles), SHEET_PER):
        cv2.imwrite(str(DEBUG_DIR / f"review_{s // SHEET_PER + 1:02d}.jpg"),
                    make_sheet(tiles[s:s + SHEET_PER], SHEET_COLS))

    # crops sheet: random images, one row each, columns = anti_a, anti_b, control, anti_d
    if crop_rows:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(crop_rows), size=min(CROP_SHEET_ROWS, len(crop_rows)),
                          replace=False)
        lines = []
        for i in sorted(pick):
            tag, crops = crop_rows[i]
            ims = []
            for name, c in crops:
                t = cv2.resize(c, (CROP_THUMB, CROP_THUMB))
                cv2.putText(t, name, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                ims.append(t)
            row = np.hstack(ims)
            cv2.putText(row, tag, (3, CROP_THUMB - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1)
            lines.append(row)
        cv2.imwrite(str(DEBUG_DIR / "crops_sheet.jpg"), np.vstack(lines))

    print(f"Images: {len(df)} | tubes found: {len(df) - len(failed)} | failed: {len(failed)}")
    print(f"Detection mode: strong={modes.count('strong')}  weak={modes.count('weak')}")
    print(f"Treated as REVERSED: {len(reversed_list)}")
    for idx, name, label, why in reversed_list:
        print(f"  #{idx} {label} {name}  ({why})")
    for name, why in failed:
        print(f"  [FAIL] {name}: {why}")
    print(f"Review sheets: {DEBUG_DIR.resolve()}\\review_XX.jpg  and  crops_sheet.jpg")
    print("CHECK: (1) boxes no longer overlap and sit on one tube each; "
          "(2) the cyan-tinted tube is labelled anti_a; (3) crops_sheet columns are "
          "anti_a, anti_b, control, anti_d.")
    print("Educational prototype -- not for clinical use.")


if __name__ == "__main__":
    main()