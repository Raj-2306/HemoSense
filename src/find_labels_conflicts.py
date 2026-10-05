import cv2
import numpy as np
import pandas as pd

df = pd.read_csv("data/splits/folds.csv")


def feat(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(cv2.resize(img, (48, 24), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    v = lab[..., 1:].flatten()          # a*, b* only: ignores lighting
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


X = np.stack([feat(p) for p in df.path])
S = X @ X.T
lab = df.label.values
print("Calibration (known conflicting pairs):",
      {f"#{a}-#{b}": round(float(S[a, b]), 3) for a, b in [(9, 81), (7, 84), (24, 75)]})

pairs = sorted(((float(S[i, j]), i, j) for i in range(len(df))
                for j in range(i + 1, len(df)) if lab[i] != lab[j]), reverse=True)
print("\nMost similar pairs with DIFFERENT labels:")
tiles = []
for s, i, j in pairs[:16]:
    print(f"  {s:.3f}  #{i} {lab[i]}  vs  #{j} {lab[j]}")
    row = []
    for k in (i, j):
        im = cv2.imdecode(np.fromfile(df.path[k], dtype=np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (600, int(im.shape[0] * 600 / im.shape[1])))
        cv2.putText(im, f"#{k} {lab[k]}", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
        row.append(im)
    h = max(r.shape[0] for r in row)
    row = [cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255)) for r in row]
    tiles.append(np.hstack(row))
cv2.imwrite("results/tube_debug/label_conflicts.jpg", np.vstack(tiles))
print("\nSaved results/tube_debug/label_conflicts.jpg")