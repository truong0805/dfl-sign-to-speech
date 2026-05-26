import cv2
import os
import mediapipe as mp
import numpy as np
from datetime import datetime
import random

# --- Config ---
DATASET_DIR = "resized_dataset"
IMG_SIZE = (128, 128)

# Random background offsets
MAX_OFFSET = 40

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

def get_available_classes():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        return []

    return sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

def select_class(classes):
    print("\n===== ASL Data Collector =====")

    for i, cls in enumerate(classes):
        count = len(os.listdir(os.path.join(DATASET_DIR, cls)))
        print(f"[{i}] {cls} ({count} images)")

    print("[N] Create new class")
    print("[Q] Quit")

    choice = input("\nSelect class: ").strip().upper()

    if choice == "Q":
        return None

    elif choice == "N":
        new_class = input("Enter class name: ").strip().upper()
        os.makedirs(os.path.join(DATASET_DIR, new_class), exist_ok=True)
        return new_class

    elif choice.isdigit() and int(choice) < len(classes):
        return classes[int(choice)]

    else:
        print("Invalid choice.")
        return select_class(classes)

def save_image(img_rgb, save_dir, class_name):
    """
    Resize + save image with slight random augmentation
    """

    img = cv2.resize(img_rgb, IMG_SIZE)

    # Random brightness
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-15, 15)

    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random horizontal flip (sometimes)
    if random.random() < 0.3:
        img = cv2.flip(img, 1)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{class_name}_{timestamp}.jpg"

    path = os.path.join(save_dir, filename)

    cv2.imwrite(path, img_bgr)

def collect_images(selected_class):
    save_dir = os.path.join(DATASET_DIR, selected_class)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print(f"\n--- Collecting [{selected_class}] ---")
    print("SPACE = capture")
    print("A = auto capture")
    print("S = stop auto")
    print("Q = quit")

    count = len(os.listdir(save_dir))

    auto_mode = False
    last_capture = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- RANDOMIZED CROP BOX ---
        box_size = 300

        cx = w // 2 + random.randint(-MAX_OFFSET, MAX_OFFSET)
        cy = h // 2 + random.randint(-MAX_OFFSET, MAX_OFFSET)

        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)

        x2 = min(w, cx + box_size // 2)
        y2 = min(h, cy + box_size // 2)

        results = hands.process(frame_rgb)

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True

            mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

        # Box color
        color = (0, 255, 0) if hand_detected else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # --- AUTO CAPTURE ---
        if auto_mode and hand_detected:
            now = cv2.getTickCount() / cv2.getTickFrequency()

            if now - last_capture >= 0.5:
                crop = frame_rgb[y1:y2, x1:x2]

                if crop.size > 0:
                    save_image(crop, save_dir, selected_class)

                    count += 1
                    last_capture = now

        # --- TEXT ---
        cv2.putText(
            frame,
            f"Class: {selected_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Saved: {count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Auto: {'ON' if auto_mode else 'OFF'}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if auto_mode else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE=capture | A=auto | S=stop | Q=back",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        cv2.imshow(f"Collecting: {selected_class}", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):

            if hand_detected:
                crop = frame_rgb[y1:y2, x1:x2]

                if crop.size > 0:
                    save_image(crop, save_dir, selected_class)

                    count += 1

                    print(f"Captured! Total: {count}")

        elif key == ord('a'):
            auto_mode = True
            last_capture = cv2.getTickCount() / cv2.getTickFrequency()
            print("Auto capture ON")

        elif key == ord('s'):
            auto_mode = False
            print("Auto capture OFF")

    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        classes = get_available_classes()

        selected = select_class(classes)

        if selected is None:
            print("Exiting.")
            break

        collect_images(selected)

if __name__ == "__main__":
    main()