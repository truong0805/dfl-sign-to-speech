import cv2
import json
import numpy as np
import streamlit as st
import mediapipe as mp
from tensorflow.keras.models import load_model
from datetime import datetime

# --- Config ---
MODEL_PATH           = "exported_models/node2_final.keras"
CLASS_MAP            = "exported_models/node2_class_map.json"
IMG_SIZE             = (128, 128)
CONFIDENCE_THRESHOLD = 0.6
BOX_SIZE             = 300   # size of the green square crop region
BOX_COLOR            = (0, 255, 0)
BOX_THICKNESS        = 2

# --- Load model and class map ---
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(CLASS_MAP) as f:
        class_map = json.load(f)
    print(f"[DEBUG] Model loaded. Classes: {list(class_map.values())}", flush=True)
    return model, class_map

model, class_map = load_resources()

# --- MediaPipe hands ---
@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False,      # video mode for real-time
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hands, mp.solutions.drawing_utils, mp_hands

hands, mp_drawing, mp_hands = load_mediapipe()

# --- Streamlit UI ---
st.title("ASL Sign Recognition")
st.caption("Place your hand inside the green box.")

col1, col2 = st.columns([2, 1])

with col1:
    frame_box = st.empty()   # live camera feed goes here

with col2:
    st.subheader("Prediction")
    prediction_box = st.empty()
    confidence_box = st.empty()
    st.subheader("Session Log")
    log_box        = st.empty()

# --- Controls ---
run   = st.checkbox("Start Camera", value=True)
stop  = st.button("Stop")

# --- Session state ---
if "log" not in st.session_state:
    st.session_state.log = []

# --- Video loop ---
cap = cv2.VideoCapture(0)

while run and not stop:
    ret, frame = cap.read()
    if not ret:
        st.error("Could not access camera.")
        break

    frame     = cv2.flip(frame, 1)   # mirror so it feels natural
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w      = frame.shape[:2]

    # Draw green box in center of frame
    cx, cy  = w // 2, h // 2
    x1, y1  = cx - BOX_SIZE // 2, cy - BOX_SIZE // 2
    x2, y2  = cx + BOX_SIZE // 2, cy + BOX_SIZE // 2
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    # Run MediaPipe on full frame
    results = hands.process(frame_rgb)

    letter     = "—"
    confidence = 0.0

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame_rgb,
            landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Crop the green box region for prediction
        crop = frame_rgb[y1:y2, x1:x2]

        if crop.size > 0:
            # Preprocess
            crop_resized = cv2.resize(crop, IMG_SIZE)
            img_array    = np.expand_dims(crop_resized.astype("float32") / 255.0, axis=0)

            # Predict
            preds      = model.predict(img_array, verbose=0)
            class_idx  = int(np.argmax(preds))
            confidence = float(np.max(preds))
            letter     = class_map.get(str(class_idx), "?")

            print(f"[DEBUG] {letter} | {confidence*100:.1f}%", flush=True)

            # Change box color based on confidence
            box_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (255, 165, 0)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

            # Log confident predictions
            if confidence >= CONFIDENCE_THRESHOLD:
                timestamp = datetime.now().strftime("%H:%M:%S")
                entry     = f"`{timestamp}` → **{letter}** ({confidence*100:.1f}%)"
                if not st.session_state.log or st.session_state.log[-1] != entry:
                    st.session_state.log.append(entry)

    # Overlay prediction text on frame
    cv2.putText(
        frame_rgb,
        f"{letter} {confidence*100:.1f}%",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    # Display frame
    frame_box.image(frame_rgb, channels="RGB", use_column_width=True)

    # Update prediction panel
    if confidence >= CONFIDENCE_THRESHOLD:
        prediction_box.markdown(f"## {letter}")
        confidence_box.markdown(f"`{confidence*100:.1f}% confident`")
    else:
        prediction_box.markdown("## —")
        confidence_box.markdown(f"`{confidence*100:.1f}% — show hand in box`")

    log_box.markdown("\n\n".join(st.session_state.log[-10:]))

cap.release()