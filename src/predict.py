"""
predict.py  --  HemoSense (educational prototype, NOT for clinical use)

Image of a 4-tube slide -> tube crops -> "reacted?" per tube -> blood group by rule.

Command line:
    python src/predict.py path/to/image.jpg
    python src/predict.py path/to/image.jpg --rotated      (slide photographed 180 deg)

High model confidence does NOT mean clinical correctness.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from extract_tubes import (TUBE_NAMES, background_colour, crop_tube, cyan_fractions,
                           decide_reversed, find_four)
from model_utils import (MODEL_VERSION, blood_group, build_model, class_probabilities,
                         control_consistent, eval_tf)

# ----------------------------- CONFIG ---------------------------------------
MODEL_PATH = Path("models/tube_model.pt")
THRESHOLD = 0.5
LOW_TUBE_CONF = 0.80       # a tube decision weaker than this triggers a warning
LOW_GROUP_CONF = 0.70      # combined confidence weaker than this triggers a warning
WORK_WIDTH = 1280          # training images were 1280 px wide
# ----------------------------------------------------------------------------

_cache = {}


def load_model(path=MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(path), device)
    if key not in _cache:
        ck = torch.load(path, map_location=device, weights_only=True)
        model = build_model(pretrained=False)
        model.load_state_dict(ck["state_dict"])
        _cache[key] = (model.to(device).eval(), ck, device)
    return _cache[key]


def predict_image(img_bgr, rotated=None):
    """rotated: None = auto (cyan tint cue), True/False = user override."""
    h0, w0 = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (WORK_WIDTH, int(h0 * WORK_WIDTH / w0)), interpolation=cv2.INTER_AREA)
    groups, mode, _ = find_four(img)
    if groups is None:
        return dict(ok=False, error="Could not find exactly four reaction spots. "
                                    "Image rejected (poor quality or not a 4-tube slide).")

    fracs = cyan_fractions(img, groups)
    rev, why = decide_reversed(-1, fracs)
    if rotated is not None:
        rev, why = bool(rotated), "user setting"
    names = TUBE_NAMES[::-1] if rev else TUBE_NAMES

    w = img.shape[1]
    bg = background_colour(img)
    lefts = [0] + [(groups[i - 1]["x1"] + groups[i]["x0"]) // 2 for i in range(1, 4)]
    rights = lefts[1:] + [w]
    crops, rects = [], []
    for i, g in enumerate(groups):
        crop, rect = crop_tube(img, g, lefts[i], rights[i], bg)
        if rev:
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        crops.append(crop)
        rects.append(rect)

    model, ck, device = load_model()
    batch = torch.stack([eval_tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                         for c in crops]).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch).squeeze(1)).cpu().numpy()

    tubes = {}
    for name, p, crop, rect in zip(names, probs, crops, rects):
        tubes[name] = dict(prob=float(p), reacted=bool(p >= THRESHOLD),
                           confidence=float(max(p, 1 - p)),
                           crop_rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), rect=rect)

    a, b, d, c = (tubes[k]["reacted"] for k in ("anti_a", "anti_b", "anti_d", "control"))
    group = blood_group(a, b, d)
    cp = class_probabilities(tubes["anti_a"]["prob"], tubes["anti_b"]["prob"],
                             tubes["anti_d"]["prob"])
    top3 = sorted(cp.items(), key=lambda kv: kv[1], reverse=True)[:3]

    warnings = []
    weak = [k for k, v in tubes.items() if v["confidence"] < LOW_TUBE_CONF]
    if weak:
        warnings.append(f"Low confidence on tube(s): {', '.join(weak)}.")
    if cp[group] < LOW_GROUP_CONF:
        warnings.append("Combined confidence is low -- result unreliable.")
    if not control_consistent(a, b, c):
        warnings.append("Control tube is inconsistent with Anti-A/Anti-B results "
                        "(invalid test, or a tube was misread).")
    if why == "no colour cue":
        warnings.append("Slide orientation could not be checked from reagent colour; "
                        "tick 'rotated 180 deg' if the slide was photographed upside down.")

    annotated = img.copy()
    for name, v in tubes.items():
        x0, y0, x1, y1 = v["rect"]
        col = (0, 0, 255) if v["reacted"] else (0, 160, 0)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), col, 4)
        cv2.putText(annotated, f"{name} {'+' if v['reacted'] else '-'} {v['prob']:.2f}",
                    (x0 + 4, max(30, y0 + 34)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)

    return dict(ok=True, group=group, confidence=float(cp[group]), top3=top3, tubes=tubes,
                warnings=warnings, rotated=rev, rotated_reason=why,
                annotated_rgb=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                model_version=ck.get("version", MODEL_VERSION), model_trained=ck.get("trained", "?"))


def to_jsonable(res, source):
    return dict(source=source, model_version=res["model_version"],
                predicted_group=res["group"], combined_confidence=round(res["confidence"], 4),
                top3=[[g, round(p, 4)] for g, p in res["top3"]], rotated=res["rotated"],
                tubes={k: dict(reacted=v["reacted"], prob=round(v["prob"], 4))
                       for k, v in res["tubes"].items()},
                warnings=res["warnings"],
                disclaimer="Educational prototype - not for clinical use.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--rotated", action="store_true", default=None)
    args = ap.parse_args()
    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    res = predict_image(img, rotated=args.rotated)
    if not res["ok"]:
        print("REJECTED:", res["error"])
    else:
        print(json.dumps(to_jsonable(res, args.image), indent=2))
    print("Educational prototype -- not for clinical use.")