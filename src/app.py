"""
app.py  --  HemoSense (educational prototype, NOT for clinical use)

Run from the project folder:
    streamlit run src/app.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from predict import MODEL_PATH, predict_image, to_jsonable          # noqa: E402

SAVE_DIR = Path("results/predictions")
DISCLAIMER = "Educational prototype — not for clinical use."

st.set_page_config(page_title="HemoSense", page_icon="🩸", layout="wide")
st.title("HemoSense — blood-group slide reader")
st.error(DISCLAIMER)
st.caption("Reads a 4-tube agglutination slide photo (Anti-A, Anti-B, control, Anti-D) and "
           "applies the standard ABO/Rh rule. It must not be used for transfusion or "
           "medical decisions. Confidence is the model's own score, not clinical correctness.")

if not MODEL_PATH.exists():
    st.warning("No trained model found. Run `python src/train.py` first.")
    st.stop()

source = st.radio("Image source", ["Upload file", "Webcam"], horizontal=True)
file = (st.file_uploader("Slide image", type=["jpg", "jpeg", "png"])
        if source == "Upload file" else st.camera_input("Take a photo of the slide"))
rotated_opt = st.selectbox("Slide orientation", ["Auto", "Normal", "Rotated 180°"])
rotated = {"Auto": None, "Normal": False, "Rotated 180°": True}[rotated_opt]

if file is not None:
    img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read this file as an image.")
        st.stop()
    res = predict_image(img, rotated=rotated)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Input")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if not res["ok"]:
        with right:
            st.error(res["error"])
            st.info("No blood group is reported for images that fail the quality check.")
        st.stop()

    with left:
        st.subheader("Detected tubes")
        st.image(res["annotated_rgb"], use_container_width=True)
        cols = st.columns(4)
        for col, name in zip(cols, ["anti_a", "anti_b", "control", "anti_d"]):
            col.image(res["tubes"][name]["crop_rgb"], caption=name, use_container_width=True)

    with right:
        st.subheader("Predicted blood group")
        st.metric("Prediction", res["group"], f"{res['confidence'] * 100:.1f}% model confidence")
        for w in res["warnings"]:
            st.warning(w)
        st.write("**Top-3 predictions**")
        st.table(pd.DataFrame([(g, f"{p * 100:.1f}%") for g, p in res["top3"]],
                              columns=["Blood group", "Model probability"]))
        st.write("**Per-tube result**")
        st.table(pd.DataFrame(
            [(k, "reacted" if v["reacted"] else "no reaction", f"{v['prob']:.2f}",
              f"{v['confidence'] * 100:.0f}%") for k, v in res["tubes"].items()],
            columns=["Tube", "Result", "P(reacted)", "Confidence"]))
        st.caption(f"Orientation: {'rotated 180°' if res['rotated'] else 'normal'} "
                   f"({res['rotated_reason']})")
        st.caption(f"Model version: {res['model_version']} · trained {res['model_trained']}")

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        out = SAVE_DIR / f"prediction_{time.strftime('%Y%m%d_%H%M%S')}.json"
        data = to_jsonable(res, getattr(file, "name", "webcam"))
        data["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        st.caption(f"Result saved: {out}")

st.divider()
st.caption(DISCLAIMER + " Cross-validated results on the selected dataset are in "
           "results/report.md (small sample, one data source, not clinical validation).")