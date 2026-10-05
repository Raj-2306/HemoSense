"""
model_utils.py  --  HemoSense (educational prototype, NOT for clinical use)

Shared pieces used by train.py, predict.py and app.py:
  * the tube classifier (ResNet18, one output: "reacted / agglutinated?")
  * image normalisation
  * the ABO/Rh decision rule that turns 3 tube results into a blood group
"""

import torch.nn as nn
from torchvision import models, transforms

# ----------------------------- CONFIG ---------------------------------------
MODEL_VERSION = "hemosense-tube-resnet18-v1"
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
TRAINABLE_PARTS = ("layer4", "fc")      # everything else stays frozen (small dataset)
# ----------------------------------------------------------------------------

eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)      # one logit: tube reacted?
    return model


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in TRAINABLE_PARTS


def blood_group(anti_a: bool, anti_b: bool, anti_d: bool) -> str:
    abo = "AB" if (anti_a and anti_b) else "A" if anti_a else "B" if anti_b else "O"
    return abo + ("+" if anti_d else "-")


def class_probabilities(pa: float, pb: float, pd: float) -> dict:
    """Probability of each of the 8 groups, assuming the three tube decisions are independent."""
    out = {}
    for c in CLASSES:
        abo, rh = c[:-1], c[-1]
        a, b, d = abo in ("A", "AB"), abo in ("B", "AB"), rh == "+"
        out[c] = ((pa if a else 1 - pa) * (pb if b else 1 - pb) * (pd if d else 1 - pd))
    return out


def control_consistent(anti_a: bool, anti_b: bool, control: bool) -> bool:
    """The Anti-AB control should react for A, B and AB samples, and not for O."""
    return control == (anti_a or anti_b)