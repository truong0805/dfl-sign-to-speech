"""
app_tester.py — Streamlit inference UI for Sign-to-Speech DFL.

Changes from previous version:
  - Updated preprocessing to use 160×160 resolution (was 128×128)
    to match the improved model architecture.
  - CLAHE preprocessing kept at inference — this matches what the model
    benefits from at test time for lighting normalization.
  - Confidence threshold default lowered from 0.60 → 0.55 since label
    smoothing during training makes the model output slightly less peaky
    (by design — the probabilities are better calibrated).
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import time
import threading
from collections import deque
from tensorflow.keras.models import load_model


# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sign-to-Speech DFL",
    page_icon="🤟",
    layout="wide",
)


# ---------------------------------------------------------------------------
# TTS — one fresh engine per call, avoids thread-safety crashes on Linux.
# pyttsx3 is not thread-safe when a single engine instance is reused across
# Streamlit reruns. Creating a new engine per speak() call is slightly
# heavier but reliable on all platforms (Windows, macOS, Linux + espeak).
# ---------------------------------------------------------------------------
def speak_letter(text: str) -> None:
    """Speak 'Letter X' in a daemon thread before st.rerun() fires."""
    def _speak():
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 150)
            eng.say(f"Letter {text}")
            eng.runAndWait()
            eng.stop()
        except Exception:
            pass  # TTS is non-critical — never crash the UI over it

    threading.Thread(target=_speak, daemon=True).start()


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

AVAILABLE_NODES = [f"node{i}" for i in range(1, 6)]
selected_node = st.sidebar.selectbox(
    "Model node",
    AVAILABLE_NODES,
    index=4,            # default = node5 (best performer)
    help="node5 achieved the highest val_acc — recommended.",
)

CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.30,
    max_value=0.95,
    value=0.55,         # Lowered from 0.60 — label smoothing makes outputs less peaky
    step=0.05,
    help="Predictions below this are shown as low-confidence. "
         "Raise if you're seeing wrong letters; lower if nothing is being detected.",
)

SMOOTHING_WINDOW = st.sidebar.slider(
    "Smoothing window (camera frames)",
    min_value=1,
    max_value=15,
    value=7,
    help="Majority-votes the last N camera frames. Higher = more stable but slower to update.",
)

st.sidebar.divider()
st.sidebar.subheader("Session log")

if st.sidebar.button("Clear log"):
    st.session_state.session_logs = []
    st.session_state.last_spoken  = None
    st.rerun()

for log in reversed(st.session_state.get("session_logs", [])):
    st.sidebar.markdown(log)


# ---------------------------------------------------------------------------
# Model loading — cached per selected node so swapping nodes reloads cleanly
# ---------------------------------------------------------------------------
MODEL_PATH = f"exported_models/{selected_node}_final.keras"
MAP_PATH   = f"exported_models/{selected_node}_class_map.json"


@st.cache_resource(show_spinner="Loading model…")
def load_dfl_assets(node_name: str):
    """Load model and class map for the given node. Cached per node_name."""
    m_path = f"exported_models/{node_name}_final.keras"
    c_path = f"exported_models/{node_name}_class_map.json"

    if not os.path.exists(m_path):
        return None, None, f"Model file not found: {m_path}"
    if not os.path.exists(c_path):
        return None, None, f"Class map not found: {c_path}"

    mdl = load_model(m_path)
    with open(c_path) as f:
        cmap = json.load(f)
    return mdl, cmap, None


model, class_map, load_error = load_dfl_assets(selected_node)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []

if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = None

if "pred_window" not in st.session_state:
    st.session_state.pred_window = deque(maxlen=SMOOTHING_WINDOW)
else:
    # Resize deque if slider changed
    if st.session_state.pred_window.maxlen != SMOOTHING_WINDOW:
        st.session_state.pred_window = deque(
            st.session_state.pred_window, maxlen=SMOOTHING_WINDOW
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Prepare a BGR OpenCV image for MobileNetV2 inference.

    MobileNetV2 was built with a Rescaling layer inside the graph that converts
    [0, 255] → [-1, 1] at runtime. So we must pass float32 values in [0, 255]
    without dividing by 255 ourselves — the model handles that internally.

    Pipeline:
      1. Center-square crop  — removes letterbox borders, reduces background noise
      2. CLAHE on luma       — normalises brightness variation across signers/lighting
      3. BGR → RGB           — OpenCV is BGR; MobileNetV2 expects RGB
      4. Resize to 160×160   — matches training resolution
      5. Cast to float32     — values stay in [0, 255], no division
    """
    h, w = cv_bgr.shape[:2]

    # 1. Center-square crop
    crop = min(h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    cv_bgr = cv_bgr[y0:y0 + crop, x0:x0 + crop]

    # 2. CLAHE on luma channel only (preserves colour, improves contrast)
    yuv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    cv_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # 3. BGR → RGB
    rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

    # 4. Resize to training resolution (160×160)
    rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)

    # 5. Float32 in [0, 255] — MobileNetV2 Rescaling layer handles normalisation
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_inference(cv_bgr: np.ndarray):
    tensor = np.expand_dims(preprocess_image(cv_bgr), axis=0)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    letter = class_map.get(str(idx), "?")
    return letter, conf, probs


def smoothed_prediction(letter: str, conf: float):
    """Majority-vote over the last SMOOTHING_WINDOW camera frames."""
    st.session_state.pred_window.append(letter)
    window = list(st.session_state.pred_window)
    voted  = max(set(window), key=window.count)
    return voted, conf


def render_prediction(letter: str, conf: float, low_conf: bool = False) -> None:
    color = "#FF4B4B" if not low_conf else "#999999"
    st.markdown(
        f"<h1 style='font-size:80px;color:{color};text-align:center;margin:0'>"
        f"{letter}</h1>",
        unsafe_allow_html=True,
    )
    label = (
        f"{conf * 100:.1f}% confident"
        if not low_conf
        else "Low confidence — adjust hand position"
    )
    st.markdown(
        f"<p style='text-align:center;font-size:20px'><b>{label}</b></p>",
        unsafe_allow_html=True,
    )
    st.progress(min(conf, 1.0))


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🤟 Sign Language Recognition (DFL)")
st.caption(f"Running **{selected_node}** · MobileNetV2 · 26 ASL classes · 160×160")

if load_error:
    st.error(load_error)
    st.info("Run `docker-compose up` to train and export the models first.")
    st.stop()

col_input, col_pred = st.columns([2, 1])

with col_input:
    tab_camera, tab_upload = st.tabs(["📷 Camera", "📁 Upload image"])

    img_file = None
    is_live  = False

    with tab_camera:
        st.caption("Take a photo of your hand sign — hold still for best results.")
        camera_data = st.camera_input("Position your hand sign clearly in the frame")
        if camera_data is not None:
            img_file = camera_data
            is_live  = True

    with tab_upload:
        st.caption("Upload a still image to test a single sign.")
        uploaded_data = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"]
        )
        if uploaded_data is not None:
            img_file = uploaded_data
            st.session_state.pred_window.clear()   # reset smoothing for stills

with col_pred:
    st.write("### Prediction")

    if img_file is None:
        st.info("Take or upload a photo to see the prediction.")
    else:
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if cv_img is None:
            st.error("Could not decode image — try another file.")
        else:
            letter, conf, probs = run_inference(cv_img)

            if is_live:
                letter, conf = smoothed_prediction(letter, conf)

            low_conf = conf < CONFIDENCE_THRESHOLD
            render_prediction(letter, conf, low_conf)

            # Log and speak only when a new confident letter appears
            if not low_conf and letter != st.session_state.last_spoken:
                timestamp = time.strftime("%H:%M:%S")
                log_entry = f"{timestamp} → **{letter}** ({conf * 100:.1f}%)"
                st.session_state.session_logs.append(log_entry)
                st.session_state.last_spoken = letter
                speak_letter(letter)   # fire BEFORE rerun so thread can start
                st.rerun()

            with st.expander("Top 5 predictions"):
                top5 = np.argsort(probs)[::-1][:5]
                for rank, i in enumerate(top5, 1):
                    lbl = class_map.get(str(i), "?")
                    pct = float(probs[i]) * 100
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    st.write(f"{rank}. **{lbl}** {bar} {pct:.1f}%")