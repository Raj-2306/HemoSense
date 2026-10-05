import sys
import cv2
import numpy as np
import pandas as pd

df = pd.read_csv("data/splits/folds.csv")
idxs = [int(a) for a in sys.argv[1:]] or [9, 81, 7, 84, 24, 75]
tiles = []
for i in idxs:
    img = cv2.imdecode(np.fromfile(df.loc[i, "path"], dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (900, int(img.shape[0] * 900 / img.shape[1])))
    cv2.putText(img, f"#{i} {df.loc[i, 'label']}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    tiles.append(img)
hmax = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                            value=(255, 255, 255)) for t in tiles]
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
cv2.imwrite("results/tube_debug/compare.jpg", np.vstack(rows))
print("Saved results/tube_debug/compare.jpg")