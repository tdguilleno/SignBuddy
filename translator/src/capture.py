import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

DATA_RAW = Path("../data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    lm = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    return lm

def main():
    cap = cv2.VideoCapture(0)
    current_label = None

    print("Press a letter key (A-Z) to start collecting that class.")
    print("Press SPACE for background class.")
    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Current: {current_label or 'None'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE → background
            current_label = "background"
        elif 65 <= key <= 90:  # A-Z
            current_label = chr(key).upper()

        if current_label:
            landmarks = extract_landmarks(frame)
            if landmarks is not None:
                label_dir = DATA_RAW / current_label
                label_dir.mkdir(exist_ok=True)
                idx = len(list(label_dir.glob("*.npy")))
                np.save(label_dir / f"{idx:04d}.npy", landmarks)
                cv2.putText(frame, f"Saved {current_label} #{idx}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("ASL Capture – press key for class", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()