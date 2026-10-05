"""
train.py  --  HemoSense (educational prototype, NOT for clinical use)

Trains a small pretrained CNN (ResNet18) to decide, for ONE tube crop,
"reacted (agglutination) or not", then evaluates with grouped 4-fold
cross-validation and combines the tube results into a blood group by rule.

Steps
  1. for each fold: test = that fold; train/validation = the other folds
     (validation = 20% of the groups, used only for early stopping)
  2. collect out-of-fold predictions for every tube
  3. per-tube metrics + composite 8-class result + confusion matrix
  4. list tubes the model strongly disagrees with (possible label problems)
  5. train a final model on all data and save it to models/tube_model.pt

The test fold is never used for tuning. All numbers are
"Performance on the selected dataset (small sample)", NOT clinical accuracy.

Run:  python src/train.py
Requires: torch torchvision scikit-learn pandas matplotlib pillow
"""

import copy
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model_utils import (CLASSES, IMG_SIZE, MEAN, MODEL_VERSION, STD, blood_group,
                         build_model, control_consistent, eval_tf, freeze_backbone)

# ----------------------------- CONFIG ---------------------------------------
TUBES_CSV = Path("data/splits/tubes.csv")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")
EXCLUDE_IMAGES = {75, 81, 84}      # O+ images that look like A+ (label conflict)
SEED = 42
EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.2
THRESHOLD = 0.5                    # probability above which a tube counts as "reacted"
FLAG_MARGIN = 0.8                  # |prob - label| above this -> flagged for review
NUM_WORKERS = 0                    # keep 0 on Windows
# ----------------------------------------------------------------------------

train_tf = transforms.Compose([
    # single-tube crops have no left/right meaning, so flips/rotations are safe here
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15, fill=(230, 230, 230)),
    transforms.ColorJitter(brightness=0.15, contrast=0.1),   # no hue change: tint matters
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class TubeDS(Dataset):
    def __init__(self, df, train):
        self.imgs = [Image.open(p).convert("RGB") for p in df.tube_path]
        self.y = df.reacted.values.astype(np.float32)
        self.tf = train_tf if train else eval_tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), torch.tensor(self.y[i])


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def run_training(train_df, val_df, device, tag):
    model = build_model().to(device)
    freeze_backbone(model)
    pos = float(train_df.reacted.mean())
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WEIGHT_DECAY)
    tl = DataLoader(TubeDS(train_df, True), batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS)
    vl = DataLoader(TubeDS(val_df, False), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    best, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(1, EPOCHS + 1):
        model.train()
        bn_eval(model)
        tloss = tcorr = n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x).squeeze(1)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tloss += loss.item() * len(y)
            tcorr += ((out > 0).float() == y).sum().item()
            n += len(y)
        model.eval()
        vloss = vcorr = vn = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze(1)
                vloss += crit(out, y).item() * len(y)
                vcorr += ((out > 0).float() == y).sum().item()
                vn += len(y)
        row = dict(epoch=ep, train_loss=tloss / n, train_acc=tcorr / n,
                   val_loss=vloss / vn, val_acc=vcorr / vn)
        hist.append(row)
        print(f"  [{tag}] ep {ep:02d}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
              f" | val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f}")
        if row["val_loss"] < best - 1e-4:
            best, best_state, bad = row["val_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  [{tag}] early stopping")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


@torch.no_grad()
def predict_probs(model, df, device):
    model.eval()
    probs = []
    for x, _ in DataLoader(TubeDS(df, False), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(probs)


def binary_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return dict(n=len(y), positives=int(y.sum()), accuracy=accuracy_score(y, p),
                balanced_accuracy=balanced_accuracy_score(y, p),
                precision=pr, recall=rc, f1=f1)


def plot_confusion(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (dataset label)")
    ax.set_title(title, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if not TUBES_CSV.exists():
        print(f"[ERROR] {TUBES_CSV} not found. Run extract_tubes.py first.")
        return
    df = pd.read_csv(TUBES_CSV)
    df = df[~df.image_index.isin(EXCLUDE_IMAGES)].reset_index(drop=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | images: {df.image_index.nunique()} | tube crops: {len(df)}"
          f" | excluded images: {sorted(EXCLUDE_IMAGES)}")

    # ---------------- grouped cross-validation ----------------
    parts, hists = [], []
    for fold in sorted(df.fold.unique()):
        test = df[df.fold == fold].copy()
        rest = df[df.fold != fold].reset_index(drop=True)
        tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRAC, random_state=SEED)
                      .split(rest, groups=rest.group))
        print(f"\nFold {fold}: train {len(tr)} | val {len(va)} | test {len(test)} tubes")
        model, hist = run_training(rest.iloc[tr], rest.iloc[va], device, f"fold{fold}")
        hist["fold"] = fold
        hists.append(hist)
        test["prob"] = predict_probs(model, test, device)
        parts.append(test)
    cv = pd.concat(parts, ignore_index=True)
    cv["pred"] = (cv.prob >= THRESHOLD).astype(int)
    cv.to_csv(RESULTS_DIR / "cv_predictions.csv", index=False)
    pd.concat(hists).to_csv(RESULTS_DIR / "training_log.csv", index=False)

    # ---------------- per-tube metrics ----------------
    rows = [dict(tube=t, **binary_metrics(g.reacted.values, g.pred.values))
            for t, g in cv.groupby("tube")]
    rows.append(dict(tube="ALL", **binary_metrics(cv.reacted.values, cv.pred.values)))
    tube_df = pd.DataFrame(rows).round(3)
    tube_df.to_csv(RESULTS_DIR / "tube_metrics.csv", index=False)

    # ---------------- composite 8-class ----------------
    wide = cv.pivot_table(index="image_index", columns="tube", values="prob")
    meta = cv.groupby("image_index").agg(label=("label", "first"), fold=("fold", "first"),
                                         group=("group", "first"))
    comp = wide.join(meta)
    comp["pred"] = [blood_group(r.anti_a >= THRESHOLD, r.anti_b >= THRESHOLD,
                                r.anti_d >= THRESHOLD) for r in comp.itertuples()]
    comp["control_ok"] = [control_consistent(r.anti_a >= THRESHOLD, r.anti_b >= THRESHOLD,
                                             r.control >= THRESHOLD) for r in comp.itertuples()]
    comp.to_csv(RESULTS_DIR / "composite_predictions.csv")
    acc = accuracy_score(comp.label, comp.pred)
    bacc = balanced_accuracy_score(comp.label, comp.pred)
    cm = confusion_matrix(comp.label, comp.pred, labels=CLASSES)
    plot_confusion(cm, CLASSES, RESULTS_DIR / "confusion_matrix_8class.png",
                   "Performance on the selected dataset (cross-validated, small sample)")
    report_txt = classification_report(comp.label, comp.pred, labels=CLASSES, zero_division=0)

    # ---------------- suspicious tubes ----------------
    cv["disagreement"] = (cv.prob - cv.reacted).abs()
    flagged = cv[cv.disagreement > FLAG_MARGIN].sort_values("disagreement", ascending=False)
    flagged[["image_index", "label", "tube", "reacted", "prob", "tube_path"]].round(3).to_csv(
        RESULTS_DIR / "flagged_tubes.csv", index=False)

    # ---------------- written report ----------------
    n_groups = df.groupby("label").group.nunique()
    lines = [
        "# HemoSense results", "",
        "**Educational prototype -- not for clinical use.**",
        "All values below are *Performance on the selected dataset* (ABO-BTI, "
        f"{df.image_index.nunique()} unique images after de-duplication, grouped 4-fold "
        "cross-validation). They are not clinical accuracy, sensitivity or specificity.", "",
        f"Excluded images (label conflict): {sorted(EXCLUDE_IMAGES)}",
        f"Model: {MODEL_VERSION}", "",
        "## Per-tube (reacted vs not reacted)", "", "```", tube_df.to_string(index=False), "```", "",
        "## Combined 8-class blood group (from the three tube results by rule)", "",
        f"- Accuracy: {acc:.3f}", f"- Balanced accuracy: {bacc:.3f}",
        f"- Control tube consistent with Anti-A/Anti-B: {comp.control_ok.mean():.3f}", "",
        "```", report_txt, "```", "",
        "Independent photo groups per class (small = unreliable per-class numbers):", "",
        "```", n_groups.reindex(CLASSES).to_string(), "```", "",
        f"Flagged tubes (model strongly disagrees with the derived label): {len(flagged)} "
        "-> results/flagged_tubes.csv", "",
        "Accuracy alone can mislead when classes are imbalanced, so balanced accuracy, "
        "per-class recall and the confusion matrix are reported too.",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Performance on the selected dataset (NOT clinical accuracy)")
    print("=" * 60)
    print(tube_df.to_string(index=False))
    print(f"\nCombined 8-class: accuracy {acc:.3f} | balanced accuracy {bacc:.3f} "
          f"| control consistent {comp.control_ok.mean():.3f}")
    print(report_txt)
    print(f"Flagged tubes: {len(flagged)}  (see results/flagged_tubes.csv)")
    thin = [c for c in CLASSES if n_groups.get(c, 0) < 4]
    if thin:
        print(f"[WARNING] Very few independent photos for: {thin} -> do not over-read their scores.")

    # ---------------- final model on all data ----------------
    print("\nTraining final model on all data ...")
    tr, va = next(GroupShuffleSplit(1, test_size=VAL_FRAC, random_state=SEED)
                  .split(df, groups=df.group))
    model, _ = run_training(df.iloc[tr], df.iloc[va], device, "final")
    torch.save(dict(state_dict=model.state_dict(), version=MODEL_VERSION, img_size=IMG_SIZE,
                    n_images=int(df.image_index.nunique()),
                    trained=time.strftime("%Y-%m-%d %H:%M")), MODEL_DIR / "tube_model.pt")
    print(f"Saved: {(MODEL_DIR / 'tube_model.pt').resolve()}")
    print("Educational prototype -- not for clinical use.")


if __name__ == "__main__":
    main()