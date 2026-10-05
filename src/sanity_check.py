import sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from predict import predict_image

folds = pd.read_csv("data/splits/folds.csv")
tubes = pd.read_csv("data/splits/tubes.csv")
rev = tubes.groupby("image_index").reversed.first().to_dict()

bad, n = [], 0
for i, r in folds.iterrows():
    img = cv2.imdecode(np.fromfile(r["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
    res = predict_image(img, rotated=bool(rev[i]) if i in rev else None)
    pred = res["group"] if res["ok"] else "REJECTED"
    n += 1
    if pred != r["label"]:
        bad.append((i, r["label"], pred))
print(f"{n} images | mismatches: {len(bad)}")
for i, lab, pred in bad:
    print(f"  #{i}  dataset label {lab}  ->  predicted {pred}")
print("NOTE: the model saw most of these images in training, so this is a sanity check, "
      "NOT an accuracy figure. Educational prototype -- not for clinical use.")