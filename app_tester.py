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
  - Crop margin restored to 15% (was incorrectly reduced to 5%).
    auto_crop.py uses MARGIN_RATIO=0.15 during dataset preparation;
    inference must match to avoid scale/composition mismatch at the model.
  - Background masking ON by default (landmark convex-hull approach).
    Training images have a plain grey wall background; replacing the
    webcam background with neutral grey (127,127,127) is the single
    biggest lever for reducing the train/inference domain gap.
  - Fixed smoothed_prediction() to track (letter, probs) pairs so
    Top-5 bars always correspond to the majority-voted letter, not the
    current frame — eliminating the contradictory display bug where
    the predicted letter did not match the #1 bar in Top-5.
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import time
import threading
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# Initialize MediaPipe Hands for live inference hand-cropping
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)


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
st.sidebar.subheader("Preprocessing")

CROP_MARGIN = st.sidebar.slider(
    "Crop margin (%)",
    min_value=0,
    max_value=40,
    value=15,           # 15% — matches auto_crop.py MARGIN_RATIO=0.15 used during
                        # dataset preparation. Keeping inference consistent with
                        # training avoids scale/composition mismatch at the model.
    step=1,
    help="Extra space added around the MediaPipe hand bounding box before cropping. "
         "15% matches the dataset preparation pipeline (auto_crop.py). "
         "Lower values crop tighter; higher values include more background context.",
)

BG_MASK_ENABLED = st.sidebar.toggle(
    "Background masking (landmark hull)",
    value=True,     # ON by default: fills everything outside the MediaPipe hand
                    # convex hull with neutral grey (127,127,127) — matching the
                    # plain grey-wall background in the ASL training images.
                    # This is the single biggest lever for reducing the webcam
                    # vs. training-data domain gap.
    help="Replaces the background outside the hand's convex hull with neutral grey (127,127,127). "
         "Training images have a plain grey wall background, so this toggle makes "
         "webcam crops look much closer to training data. Uses MediaPipe landmark "
         "geometry — works for any skin tone and never punches holes inside the hand. "
         "Disable only if masking causes artefacts (very unusual lighting).",
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
    # Each entry is a (letter, probs_array) pair so the Top-5 display always
    # corresponds to the majority-voted letter, not the current frame.
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
def crop_hand_from_frame(image: np.ndarray, margin_ratio: float = 0.05,
                         apply_hull_mask: bool = False) -> np.ndarray:
    """
    Detect hand via MediaPipe, crop a square bounding box centered on the hand
    including margin, clamp to boundaries, and return the BGR crop.
    Returns None if no hand is detected.

    Args:
        image:           BGR frame from camera or uploaded file.
        margin_ratio:    Extra padding around the MediaPipe bounding box as a
                         fraction of the box size. Default 0.05 (5%) — tighter
                         than the 0.15 used during dataset creation, so webcam
                         backgrounds bleed into the margin less.
        apply_hull_mask: If True, fill all pixels outside the hand convex hull
                         (computed from MediaPipe landmarks) with neutral grey
                         (127, 127, 127) — matching the plain-wall colour of the
                         training dataset. Geometry-based: works for any skin tone
                         and never punches holes inside the hand itself.
    """
    h, w = image.shape[:2]

    # Apply CLAHE on luma channel for better detection under varied lighting
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Convert BGR to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Gather coordinates of all detected hand landmarks to compute bounding box
    x_coords = []
    y_coords = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            x_coords.append(int(lm.x * w))
            y_coords.append(int(lm.y * h))

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Calculate bounding box dimensions
    box_w = x_max - x_min
    box_h = y_max - y_min
    box_size = max(box_w, box_h)

    # Determine rectangular crop size using the caller-supplied margin ratio.
    # We calculate the margin based on the maximum dimension (box_size) to keep
    # margins consistent, then apply it symmetrically to width and height.
    margin = int(box_size * margin_ratio)
    crop_w = box_w + 2 * margin
    crop_h = box_h + 2 * margin

    # Clamp dimensions if they exceed image boundaries
    if crop_w > w:
        crop_w = w
    if crop_h > h:
        crop_h = h

    # Find center of the bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate initial crop coordinates
    x_start = center_x - crop_w // 2
    x_end = x_start + crop_w
    y_start = center_y - crop_h // 2
    y_end = y_start + crop_h

    # Shift crop box if it extends outside the image boundaries
    if x_start < 0:
        x_end -= x_start
        x_start = 0
        if x_end > w:
            x_end = w
    if x_end > w:
        x_start -= (x_end - w)
        x_end = w
        if x_start < 0:
            x_start = 0

    if y_start < 0:
        y_end -= y_start
        y_start = 0
        if y_end > h:
            y_end = h
    if y_end > h:
        y_start -= (y_end - h)
        y_end = h
        if y_start < 0:
            y_start = 0

    # Extract the cropped region of interest (from the ORIGINAL image)
    cropped_hand = image[y_start:y_end, x_start:x_end]

    if cropped_hand.size == 0:
        return None

    # -----------------------------------------------------------------------
    # Landmark convex-hull background mask (optional)
    #
    # Approach: use the 21 MediaPipe hand keypoints (already detected above)
    # to compute the geometric convex hull of the hand in the cropped frame.
    # Everything outside the hull is replaced with neutral grey (127,127,127),
    # which matches the plain-wall colour of the training dataset backgrounds.
    #
    # Why geometry instead of HSV skin colour?
    #   - HSV ranges can't cover all skin tones without also matching backgrounds.
    #   - Knuckle shadows and lighter palm areas get misclassified as non-skin,
    #     creating black holes INSIDE the hand that the model has never seen.
    #   - Landmark positions are computed by MediaPipe independently of colour,
    #     so the hull accurately traces the hand regardless of skin tone.
    #
    # The dilation step extends the hull boundary outward so fingertip and
    # wrist edges (which sit just outside the convex hull) are also included.
    # -----------------------------------------------------------------------
    if apply_hull_mask:
        crop_h, crop_w = cropped_hand.shape[:2]

        # Transform landmark coords from full-frame space into crop space
        pts = []
        for hand_lm in results.multi_hand_landmarks:
            for lm in hand_lm.landmark:
                px = int(lm.x * w) - x_start
                py = int(lm.y * h) - y_start
                px = max(0, min(px, crop_w - 1))
                py = max(0, min(py, crop_h - 1))
                pts.append([px, py])

        hull = cv2.convexHull(np.array(pts, dtype=np.int32))

        # Rasterise the hull into a binary mask
        hand_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillConvexPoly(hand_mask, hull, 255)

        # Dilate generously so edges/wrist (outside the hull) stay included
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)

        # Fill background pixels with neutral grey (training-data wall colour)
        bg = np.full_like(cropped_hand, 127)
        cropped_hand = np.where(hand_mask[:, :, np.newaxis] > 0, cropped_hand, bg)

    # Pad the rectangular crop to square with black (0, 0, 0) borders
    # to match the dataset preprocessing (resize.py)
    ch, cw_px = cropped_hand.shape[:2]
    if ch != cw_px:
        size = max(ch, cw_px)
        pad_h = (size - ch) // 2
        pad_w = (size - cw_px) // 2
        cropped_hand = cv2.copyMakeBorder(
            cropped_hand, pad_h, size - ch - pad_h, pad_w, size - cw_px - pad_w,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    return cropped_hand


def preprocess_image(cv_bgr: np.ndarray, margin_ratio: float = 0.05,
                     apply_bg_mask: bool = False):
    """
    Prepare a BGR OpenCV image for MobileNetV2 inference.
    Attempts to crop the hand. Falls back to a center-square crop if no hand is detected.
    Returns (processed_image, hand_detected)

    Args:
        cv_bgr:        Input BGR image.
        margin_ratio:  Fraction of bounding-box size added as margin around the crop.
        apply_bg_mask: If True, fill outside-hull pixels with neutral grey inside
                       crop_hand_from_frame() to reduce background domain shift.

    NOTE: CLAHE is intentionally NOT applied a second time.
    crop_hand_from_frame() applies CLAHE internally for MediaPipe detection only;
    the crop itself comes from the ORIGINAL (non-enhanced) image, matching exactly
    what auto_crop.py does during dataset preparation.
    """
    cropped = crop_hand_from_frame(
        cv_bgr,
        margin_ratio=margin_ratio,
        apply_hull_mask=apply_bg_mask,
    )
    hand_detected = True

    # Check if hand crop succeeded
    if cropped is not None:
        cv_bgr = cropped
    else:
        hand_detected = False
        # Fallback: center-square crop (no masking applied)
        h, w = cv_bgr.shape[:2]
        crop = min(h, w)
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
        cv_bgr = cv_bgr[y0:y0 + crop, x0:x0 + crop]

    # Convert BGR → RGB (no extra CLAHE — training pipeline didn't double-apply it)
    rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

    # Resize cleanly to training resolution (160×160)
    rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)

    return rgb.astype(np.float32), hand_detected


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_inference(cv_bgr: np.ndarray, margin_ratio: float = 0.05,
                  apply_bg_mask: bool = False):
    processed, hand_detected = preprocess_image(
        cv_bgr,
        margin_ratio=margin_ratio,
        apply_bg_mask=apply_bg_mask,
    )
    tensor = np.expand_dims(processed, axis=0)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    letter = class_map.get(str(idx), "?")
    return letter, conf, probs, processed, hand_detected


def smoothed_prediction(letter: str, conf: float, probs: np.ndarray):
    """
    Majority-vote letter over the last SMOOTHING_WINDOW camera frames.

    Each window entry stores a (letter, probs) pair so we can return the
    probs array that belongs to the frame where the winning letter last
    appeared.  This guarantees the Top-5 bars always correspond to the
    letter shown at the top of the prediction panel — no contradictions.

    Args:
        letter: Predicted letter for the current frame.
        conf:   Confidence (max prob) for the current frame.
        probs:  Full probability vector for the current frame.

    Returns:
        (voted_letter, voted_conf, voted_probs)
    """
    st.session_state.pred_window.append((letter, probs))
    window  = list(st.session_state.pred_window)
    letters = [entry[0] for entry in window]
    voted   = max(set(letters), key=letters.count)

    # Find the probs from the most recent frame that predicted the voted letter
    voted_probs = probs  # fallback: use current frame if no match (shouldn't happen)
    for entry_letter, entry_probs in reversed(window):
        if entry_letter == voted:
            voted_probs = entry_probs
            break

    voted_conf = float(voted_probs[int(np.argmax(voted_probs))])
    return voted, voted_conf, voted_probs


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
            # Clear prediction window if a new camera photo is taken
            current_id = camera_data.id if hasattr(camera_data, 'id') else id(camera_data)
            if st.session_state.get("last_camera_id") != current_id:
                st.session_state.pred_window.clear()
                st.session_state.last_camera_id = current_id
            
            img_file = camera_data
            is_live  = True
        else:
            # Reset prediction window when camera photo is cleared
            st.session_state.pred_window.clear()
            st.session_state.last_camera_id = None

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
            letter, conf, probs, processed_rgb, hand_detected = run_inference(
                cv_img,
                margin_ratio=CROP_MARGIN / 100.0,
                apply_bg_mask=BG_MASK_ENABLED,
            )

            if is_live:
                letter, conf, probs = smoothed_prediction(letter, conf, probs)

            low_conf = conf < CONFIDENCE_THRESHOLD
            render_prediction(letter, conf, low_conf)

            # Button to read the detected letter out loud
            if st.button("🔊 Sounds", use_container_width=True):
                speak_letter(letter)

            # Show warning if hand is not detected
            if not hand_detected:
                st.warning("⚠️ No hand detected in frame. Falling back to center-crop (position hand in center).")
            else:
                mask_status = " · skin mask ON" if BG_MASK_ENABLED else ""
                st.success(f"✅ Hand detected and cropped (margin {CROP_MARGIN}%{mask_status}).")

            # Show a visual preview of what the model actually sees (the cropped hand)
            with st.expander("Show processed input (hand crop)", expanded=True):
                st.image(processed_rgb.astype(np.uint8), caption="160x160 Model Input", use_container_width=True)

            # Log and speak only when a new confident letter appears
            if not low_conf and letter != st.session_state.last_spoken:
                timestamp = time.strftime("%H:%M:%S")
                log_entry = f"{timestamp} → **{letter}** ({conf * 100:.1f}%)"
                st.session_state.session_logs.append(log_entry)
                st.session_state.last_spoken = letter
                speak_letter(letter)   # fire BEFORE rerun so thread can start
                st.rerun()

            with st.expander("Top 5 predictions"):
                if is_live:
                    st.caption(
                        "Probabilities shown for the frame that determined the "
                        f"smoothed prediction **{letter}** — consistent with the "
                        "letter displayed above."
                    )
                top5 = np.argsort(probs)[::-1][:5]
                for rank, i in enumerate(top5, 1):
                    lbl = class_map.get(str(i), "?")
                    pct = float(probs[i]) * 100
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    st.write(f"{rank}. **{lbl}** {bar} {pct:.1f}%")