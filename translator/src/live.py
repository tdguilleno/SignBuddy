import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from pathlib import Path

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

MODEL_PATH = Path("../models/asl_knn.pkl")
clf, scaler = load(MODEL_PATH)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

def predict(landmarks):
    X = scaler.transform(landmarks.reshape(1, -1))
    return clf.predict(X)[0]

def main():
    cap = cv2.VideoCapture(0)
    text = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            pred = predict(landmarks)
            if pred != "background":
                text = pred
        cv2.putText(frame, f"ASL: {text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("ASL → Text (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()