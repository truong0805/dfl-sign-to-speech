import streamlit as st
import cv2
import numpy as np
import json
import os
import time
from tensorflow.keras.models import load_model
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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
st.set_page_config(
    page_title="Sign-to-Speech DFL Test",
    layout="wide"
)

st.title("Sign Language Recognition System (DFL Tested)")
st.subheader("Real-time decentralized model inference preview")


# --- 3. Resource Asset Loading ---
MODEL_PATH = "exported_models/node1_final.keras"
MAP_PATH = "exported_models/node1_class_map.json"


@st.cache_resource
def load_dfl_assets():
    """Load the trained model and label mappings."""

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None

    if not os.path.exists(MAP_PATH):
        st.error(f"Class map not found: {MAP_PATH}")
        return None, None

    model = load_model(MODEL_PATH)

    with open(MAP_PATH, "r") as f:
        class_map = json.load(f)

    return model, class_map


model, class_map = load_dfl_assets()


# --- 4. Session State ---
if "session_logs" not in st.session_state:
    st.session_state.session_logs = []


# --- 5. Main App ---
if model is not None:

    # Sidebar logs
    st.sidebar.header("Session Log")

    for log in reversed(st.session_state.session_logs):
        st.sidebar.markdown(log)

    # Layout
    col_input, col_pred = st.columns([2, 1])

    # ---------------- INPUT COLUMN ----------------
    with col_input:

        tab_camera, tab_upload = st.tabs([
            "📷 Take Live Photo",
            "📁 Drag & Drop Image File"
        ])

        img_file = None

        # Camera input
        with tab_camera:

            camera_data = st.camera_input(
                "Position your hand sign clearly in the frame"
            )

            if camera_data is not None:
                img_file = camera_data

        # Upload input
        with tab_upload:

            uploaded_data = st.file_uploader(
                "Choose an image file...",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_data is not None:
                img_file = uploaded_data

    # ---------------- PREDICTION COLUMN ----------------
    with col_pred:

        st.write("### Prediction")

        if img_file is not None:

            # Decode image
            bytes_data = img_file.getvalue()

            cv_img = cv2.imdecode(
                np.frombuffer(bytes_data, np.uint8),
                cv2.IMREAD_COLOR
            )

            # Match training preprocessing EXACTLY
            rgb_img = cv2.cvtColor(
                cv_img,
                cv2.COLOR_BGR2RGB
            )

            resized = cv2.resize(
                rgb_img,
                (128, 128)
            )

            normalized = preprocess_input(
                resized.astype(np.float32)
            )

            input_tensor = np.expand_dims(
                normalized,
                axis=0
            )

            # Inference
            preds = model.predict(
                input_tensor,
                verbose=0
            )[0]

            best_class_idx = np.argmax(preds)

            confidence = preds[best_class_idx]

            # Confidence threshold
            if confidence < 0.90:
                predicted_letter = "Unknown"
            else:
                predicted_letter = class_map.get(
                    str(best_class_idx),
                    "Unknown"
                )

            # Main prediction display
            st.markdown(
                f"""
                <h1 style='
                    font-size: 80px;
                    color: #FF4B4B;
                    text-align: center;
                    margin: 0;
                '>
                    {predicted_letter}
                </h1>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <p style='
                    text-align: center;
                    font-size: 20px;
                '>
                    <b>{confidence * 100:.1f}% confident</b>
                </p>
                """,
                unsafe_allow_html=True
            )

            # --- DEBUGGING: TOP 3 PREDICTIONS ---
            st.write("### Top Predictions")

            top3_idx = np.argsort(preds)[-3:][::-1]

            for idx in top3_idx:

                label = class_map.get(
                    str(idx),
                    "Unknown"
                )

                prob = preds[idx] * 100

                st.write(f"{label}: {prob:.2f}%")

            # Logging
            timestamp = time.strftime("%H:%M:%S")

            log_entry = (
                f"{timestamp} → "
                f"**{predicted_letter}** "
                f"({confidence * 100:.1f}%)"
            )

            # Prevent duplicate consecutive logs
            if (
                not st.session_state.session_logs
                or st.session_state.session_logs[-1].split("**")[1]
                != predicted_letter
            ):

                st.session_state.session_logs.append(log_entry)

                # Speak prediction
                speak_letter(predicted_letter)

                # Refresh UI
                st.rerun()