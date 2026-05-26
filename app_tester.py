import streamlit as st
import cv2
import numpy as np
import json
import os
import time
import threading
from collections import deque
from tensorflow.keras.models import load_model
import pyttsx3

# --- 1. Audio Engine Initialization ---
@st.cache_resource
def get_tts_engine():
    """Initialize the Text-to-Speech engine safely."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

engine = get_tts_engine()

def speak_letter(text):
    """
    Speak the recognized sign character in a daemon thread so it never
    blocks Streamlit's main script execution.
    BUG FIX: previously speak_letter() was called AFTER st.rerun(), which
    restarts the script immediately — making the TTS call unreachable.
    Running in a thread sidesteps the rerun timing issue entirely.
    """
    def _speak():
        try:
            engine.say(f"Letter {text}")
            engine.runAndWait()
        except Exception:
            pass  # TTS failure should never crash the UI
    threading.Thread(target=_speak, daemon=True).start()


# --- 2. Page Configuration & Setup ---
st.set_page_config(page_title="Sign-to-Speech DFL Test", layout="wide")
st.title("Sign Language Recognition System (DFL Tested)")
st.subheader("Real-time decentralized model inference preview")


# --- 3. Resource Asset Loading ---
MODEL_PATH = "exported_models/node3_final.keras"
MAP_PATH   = "exported_models/node3_class_map.json"

@st.cache_resource
def load_dfl_assets():
    """Load the target decentralized Keras model and structural label mappings."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAP_PATH):
        st.error(
            f"Could not locate exported model binaries at '{MODEL_PATH}'. "
            "Ensure your training pipeline completed successfully."
        )
        return None, None
    model = load_model(MODEL_PATH)
    with open(MAP_PATH, "r") as f:
        class_map = json.load(f)
    return model, class_map

model, class_map = load_dfl_assets()


# --- 4. Session State & History Tracking ---
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []

# Rolling window for live-photo prediction smoothing.
# Stores the last N predicted letter indices; majority vote is displayed.
SMOOTHING_WINDOW = 7
if "pred_window" not in st.session_state:
    st.session_state.pred_window = deque(maxlen=SMOOTHING_WINDOW)

# Minimum confidence to display a prediction rather than "Low confidence".
CONFIDENCE_THRESHOLD = 0.45


# --- 5. Preprocessing helpers ---

def preprocess_image(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Shared preprocessing pipeline for both camera and uploaded images.

    Steps:
      1. Center-square crop  — removes most background clutter.
      2. CLAHE enhancement   — normalises brightness so dim/bright conditions
                               look more like the clean Kaggle training set.
      3. BGR→RGB conversion  — Keras/MobileNetV2 expects RGB; OpenCV loads BGR.
      4. Resize to 128×128   — matches training resolution.
      5. Normalise to [0, 1] — matches rescale=1/255 used during training.

    Returns a float32 array of shape (128, 128, 3), ready for np.expand_dims.
    """
    h, w = cv_bgr.shape[:2]

    # 1. Center-square crop
    crop = min(h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    cv_bgr = cv_bgr[y0:y0 + crop, x0:x0 + crop]

    # 2. CLAHE contrast enhancement (operates on luma channel only)
    yuv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    cv_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # 3. BGR → RGB  (critical: model was trained on RGB images)
    rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

    # 4. Resize
    rgb = cv2.resize(rgb, (128, 128))

    # 5. Normalise
    return (rgb / 255.0).astype(np.float32)


def run_inference(cv_bgr: np.ndarray):
    """
    Run model inference on a single BGR frame.
    Returns (predicted_letter, confidence, all_probs).
    """
    tensor = np.expand_dims(preprocess_image(cv_bgr), axis=0)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    letter = class_map.get(str(idx), "?")
    return letter, conf, probs


def smoothed_prediction(letter: str, conf: float):
    """
    Push the latest prediction into the rolling window and return the
    majority-vote letter + its average confidence in the window.
    This reduces flicker from single noisy frames during live capture.
    """
    st.session_state.pred_window.append(letter)
    window = list(st.session_state.pred_window)
    # Majority vote
    voted = max(set(window), key=window.count)
    # Average confidence for the voted letter is just the latest conf
    # (keeping it simple; the visual value is the stability, not the number)
    return voted, conf


def render_prediction(letter: str, conf: float, low_conf: bool = False):
    """Render the large prediction letter + confidence bar."""
    color = "#FF4B4B" if not low_conf else "#999999"
    st.markdown(
        f"<h1 style='font-size:80px;color:{color};text-align:center;margin:0'>"
        f"{letter}</h1>",
        unsafe_allow_html=True,
    )
    label = f"{conf * 100:.1f}% confident" if not low_conf else "Low confidence — adjust hand position"
    st.markdown(
        f"<p style='text-align:center;font-size:20px'><b>{label}</b></p>",
        unsafe_allow_html=True,
    )
    # Confidence bar
    st.progress(min(conf, 1.0))


# --- 6. Application UI Layout ---
if model is not None:
    st.sidebar.header("Session Log")
    for log in reversed(st.session_state.session_logs):
        st.sidebar.markdown(log)

    col_input, col_pred = st.columns([2, 1])

    with col_input:
        tab_camera, tab_upload = st.tabs(["📷 Take Live Photo", "📁 Drag & Drop Image File"])

        img_file   = None
        is_live    = False

        with tab_camera:
            camera_data = st.camera_input("Position your hand sign clearly in the frame")
            if camera_data is not None:
                img_file = camera_data
                is_live  = True

        with tab_upload:
            uploaded_data = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
            if uploaded_data is not None:
                img_file = uploaded_data
                # Clear the smoothing window when switching to a static image
                st.session_state.pred_window.clear()

    with col_pred:
        st.write("### Prediction")

        if img_file is not None:
            bytes_data = img_file.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv_img is None:
                st.error("Could not decode image. Please try another file.")
            else:
                letter, conf, probs = run_inference(cv_img)

                # For live photos, apply rolling-window majority vote
                if is_live:
                    letter, conf = smoothed_prediction(letter, conf)

                low_conf = conf < CONFIDENCE_THRESHOLD
                render_prediction(letter, conf, low_conf)

                # Log + TTS — only on new predictions above threshold
                if not low_conf:
                    timestamp = time.strftime("%H:%M:%S")
                    log_entry = f"{timestamp} &rarr; **{letter}** ({conf * 100:.1f}%)"

                    last_letter = (
                        st.session_state.session_logs[-1].split("**")[1]
                        if st.session_state.session_logs else None
                    )
                    if letter != last_letter:
                        st.session_state.session_logs.append(log_entry)
                        # BUG FIX: speak BEFORE rerun so the thread is launched
                        # while the current script execution is still live.
                        speak_letter(letter)
                        st.rerun()

                # Show top-3 alternatives in an expander for transparency
                with st.expander("Top 3 predictions"):
                    top3_idx = np.argsort(probs)[::-1][:3]
                    for rank, i in enumerate(top3_idx, 1):
                        lbl = class_map.get(str(i), "?")
                        pct = probs[i] * 100
                        st.write(f"{rank}. **{lbl}** — {pct:.1f}%")