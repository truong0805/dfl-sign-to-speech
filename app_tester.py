import streamlit as st
import cv2
import numpy as np
import json
import os
import time
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
    """Speak the recognized sign character out loud."""
    engine.say(f"Letter {text}")
    engine.runAndWait()


# --- 2. Page Configuration & Setup ---
st.set_page_config(page_title="Sign-to-Speech DFL Test", layout="wide")
st.title("Sign Language Recognition System (DFL Tested)")
st.subheader("Real-time decentralized model inference preview")


# --- 3. Resource Asset Loading ---
MODEL_PATH = "exported_models/node3_final.keras"
MAP_PATH = "exported_models/node3_class_map.json"

@st.cache_resource
def load_dfl_assets():
    """Load the target decentralized Keras model and structural label mappings."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAP_PATH):
        st.error(f"Could not locate exported model binaries at '{MODEL_PATH}'. Ensure your training pipeline completed successfully.")
        return None, None

    model = load_model(MODEL_PATH)
    with open(MAP_PATH, "r") as f:
        class_map = json.load(f)
    return model, class_map

model, class_map = load_dfl_assets()


# --- 4. Session State & History Tracking ---
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []


# --- 5. Application UI Layout and Processing ---
if model is not None:
    # Render historical log tracking in the left sidebar
    st.sidebar.header("Session Log")
    for log in reversed(st.session_state.session_logs):
        st.sidebar.markdown(log)

    # Establish main structural grid layout
    col_input, col_pred = st.columns([2, 1])

    with col_input:
        # Create tabs to switch seamlessly between Camera and Drag & Drop file upload
        tab_camera, tab_upload = st.tabs(["📷 Take Live Photo", "📁 Drag & Drop Image File"])

        img_file = None

        with tab_camera:
            camera_data = st.camera_input("Position your hand sign clearly in the frame")
            if camera_data is not None:
                img_file = camera_data

        with tab_upload:
            uploaded_data = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
            if uploaded_data is not None:
                img_file = uploaded_data

    with col_pred:
        st.write("### Prediction")

        if img_file is not None:
            # Decode file bytes into standard OpenCV matrix array format
            bytes_data = img_file.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # --- 1. CENTER CROP PROCESSING ---
            # Squashes wide-angle background clutter to closely mimic tight Kaggle image boundaries
            h, w, _ = cv_img.shape
            crop_size = min(h, w)
            start_x = (w - crop_size) // 2
            start_y = (h - crop_size) // 2
            cv_img = cv_img[start_y:start_y+crop_size, start_x:start_x+crop_size]

            # --- 2. CONTRAST & ILLUMINATION ENHANCEMENT ---
            # Converts to YUV space to isolate and boost the brightness channel locally via CLAHE
            yuv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            yuv[:,:,0] = clahe.apply(yuv[:,:,0])
            cv_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

            # --- 3. MODEL INFERENCE ---
            # Re-verify and standardize channels to align with MobileNetV2 inputs
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_img, (128, 128))
            normalized = resized / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)

            # Compute network inference
            preds = model.predict(input_tensor, verbose=0)[0]
            best_class_idx = np.argmax(preds)
            confidence = preds[best_class_idx]

            # Fetch text translation string matching class index keys
            predicted_letter = class_map.get(str(best_class_idx), "Unknown")

            # Render styled display elements using corrected string logic parameters
            st.markdown(f"<h1 style='font-size: 80px; color: #FF4B4B; text-align: center; margin: 0;'>{predicted_letter}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 20px;'><b>{confidence * 100:.1f}% confident</b></p>", unsafe_allow_html=True)

            # Package time tracking stamps
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"{timestamp} &rarr; **{predicted_letter}** ({confidence * 100:.1f}%)"

            # Append unique character shifts to log state history & execute speech audio
            if not st.session_state.session_logs or st.session_state.session_logs[-1].split("**")[1] != predicted_letter:
                st.session_state.session_logs.append(log_entry)
                st.rerun()
                speak_letter(predicted_letter)