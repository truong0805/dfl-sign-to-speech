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

# --- 1. Audio Engine ---
@st.cache_resource
def get_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

engine = get_tts_engine()

def speak_letter(text):
    """
    Run TTS in a daemon thread — speak BEFORE st.rerun() is called,
    otherwise the call is unreachable (rerun restarts the script immediately).
    """
    def _speak():
        try:
            engine.say(f"Letter {text}")
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()


# --- 2. Page config ---
st.set_page_config(page_title="Sign-to-Speech DFL Test", layout="wide")
st.title("Sign Language Recognition System (DFL Tested)")
st.subheader("Real-time decentralized model inference preview")


# --- 3. Model loading ---
MODEL_PATH = "exported_models/node3_final.keras"
MAP_PATH   = "exported_models/node3_class_map.json"

@st.cache_resource
def load_dfl_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAP_PATH):
        st.error(f"Model not found at '{MODEL_PATH}'. Run training first.")
        return None, None
    model = load_model(MODEL_PATH)
    with open(MAP_PATH) as f:
        class_map = json.load(f)
    return model, class_map

model, class_map = load_dfl_assets()


# --- 4. Session state ---
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []

# Rolling window for live-photo smoothing (majority vote over last N frames)
SMOOTHING_WINDOW = 3
if "pred_window" not in st.session_state:
    st.session_state.pred_window = deque(maxlen=SMOOTHING_WINDOW)

CONFIDENCE_THRESHOLD = 0.80


# --- 5. Preprocessing ---

def preprocess_image(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Shared preprocessing for both camera and uploaded images.

    IMPORTANT — EfficientNetB3 has a Rescaling layer that normalizes [0-255] 
    to [0, 1] (standard ImageNet preprocessing). This means:
      - Pass raw pixel values as float32 in [0, 255]
      - The model's Rescaling layer handles normalization internally

    Steps here:
      1. Center-square crop  — reduces background clutter
      2. CLAHE enhancement   — normalises brightness (operated on luma only)
      3. BGR → RGB           — OpenCV loads BGR; model expects RGB
      4. Resize to 128×128   — matches training resolution
      5. Cast to float32     — pixels stay in [0, 255]
    """
    h, w = cv_bgr.shape[:2]

    # 1. Center-square crop
    crop = min(h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    cv_bgr = cv_bgr[y0:y0 + crop, x0:x0 + crop]

    # 3. BGR → RGB
    rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

    # 4. Resize
    rgb = cv2.resize(rgb, (128, 128))

    # 5. Cast to float32, pixels in [0, 255] — EfficientNetB0 preprocesses internally
    return rgb.astype(np.float32)


def run_inference(cv_bgr: np.ndarray):
    tensor = np.expand_dims(preprocess_image(cv_bgr), axis=0)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    letter = class_map.get(str(idx), "?")
    return letter, conf, probs


def smoothed_prediction(letter: str, conf: float):
    st.session_state.pred_window.append(letter)
    window = list(st.session_state.pred_window)
    voted  = max(set(window), key=window.count)
    return voted, conf


def render_prediction(letter: str, conf: float, low_conf: bool = False):
    color = "#FF4B4B" if not low_conf else "#999999"
    st.markdown(
        f"<h1 style='font-size:80px;color:{color};text-align:center;margin:0'>"
        f"{letter}</h1>",
        unsafe_allow_html=True,
    )
    label = f"{conf*100:.1f}% confident" if not low_conf else "Low confidence — adjust hand position"
    st.markdown(
        f"<p style='text-align:center;font-size:20px'><b>{label}</b></p>",
        unsafe_allow_html=True,
    )
    st.progress(min(conf, 1.0))


# --- 6. UI ---
if model is not None:
    st.sidebar.header("Session Log")
    for log in reversed(st.session_state.session_logs):
        st.sidebar.markdown(log)

    col_input, col_pred = st.columns([2, 1])

    with col_input:
        tab_camera, tab_upload = st.tabs(["📷 Take Live Photo", "📁 Drag & Drop Image File"])

        img_file = None
        is_live  = False

        with tab_camera:
            camera_data = st.camera_input("Position your hand sign clearly in the frame")
            if camera_data is not None:
                img_file = camera_data
                is_live  = True

        with tab_upload:
            uploaded_data = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
            if uploaded_data is not None:
                img_file = uploaded_data
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

                if is_live:
                    letter, conf = smoothed_prediction(letter, conf)

                if conf < CONFIDENCE_THRESHOLD:
                    render_prediction("UNKNOWN", conf, True)
                else:
                    render_prediction(letter, conf, False)

                if not low_conf:
                    timestamp = time.strftime("%H:%M:%S")
                    log_entry = f"{timestamp} &rarr; **{letter}** ({conf*100:.1f}%)"
                    last_letter = (
                        st.session_state.session_logs[-1].split("**")[1]
                        if st.session_state.session_logs else None
                    )
                    if letter != last_letter:
                        st.session_state.session_logs.append(log_entry)
                        speak_letter(letter)   # speak BEFORE rerun
                        st.rerun()

                with st.expander("Top 3 predictions"):
                    top3 = np.argsort(probs)[::-1][:3]
                    for rank, i in enumerate(top3, 1):
                        lbl = class_map.get(str(i), "?")
                        pct = probs[i] * 100
                        st.write(f"{rank}. **{lbl}** — {pct:.1f}%")