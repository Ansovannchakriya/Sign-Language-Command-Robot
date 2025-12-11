# 

"""
Gesture recognition demo using laptop camera only.
Requirements: opencv, mediapipe, numpy, tflite-runtime or tensorflow.
"""

import time
import collections
import numpy as np
import cv2
import argparse
import logging

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = r"D:\Sign_Language_Robot\gesture_model.tflite"
IMG_SIZE = 224

# smoothing/debounce params
SMOOTH_WINDOW = 5
DEBOUNCE_COUNT = 3
CONFIDENCE_THRESH = 0.6

# Class order — MUST match training
CLASS_NAMES = ["go", "stop", "left", "right"]

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gesture-camera")


# ----------------------------
# TFLite Interpreter
# ----------------------------
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


# ----------------------------
# Helpers
# ----------------------------
def preprocess_roi(roi):
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def expand_bbox(xmin, ymin, xmax, ymax, scale=0.25, img_w=None, img_h=None):
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2
    cy = ymin + h / 2
    side = max(w, h) * (1 + scale)
    xmin2 = int(cx - side / 2)
    ymin2 = int(cy - side / 2)
    xmax2 = int(cx + side / 2)
    ymax2 = int(cy + side / 2)
    if img_w is not None:
        xmin2 = max(0, xmin2)
        ymin2 = max(0, ymin2)
        xmax2 = min(img_w - 1, xmax2)
        ymax2 = min(img_h - 1, ymax2)
    return xmin2, ymin2, xmax2, ymax2


# ----------------------------
# MAIN
# ----------------------------
def run(camera_id=0, model_path=MODEL_PATH):
    # Load TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_dtype = input_details[0]["dtype"]

    # MediaPipe Hands
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(False, 1, 0.5, 0.5)

    # Camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        log.error("Cannot open camera")
        return

    recent = collections.deque(maxlen=SMOOTH_WINDOW)
    consecutive_count = 0
    last_command = None

    log.info("Starting camera-only gesture recognition. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        label = "None"
        conf = 0.0

        if results.multi_hand_landmarks:
            xs = [lm.x for lm in results.multi_hand_landmarks[0].landmark]
            ys = [lm.y for lm in results.multi_hand_landmarks[0].landmark]
            xmin, ymin = int(min(xs) * w), int(min(ys) * h)
            xmax, ymax = int(max(xs) * w), int(max(ys) * h)

            x0, y0, x1, y1 = expand_bbox(xmin, ymin, xmax, ymax, 0.25, w, h)
            roi = frame[y0:y1, x0:x1].copy()
            if roi.size == 0:
                roi = frame.copy()

            input_tensor = preprocess_roi(roi).astype(in_dtype)
            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])[0]

            conf = float(np.max(out))
            idx = int(np.argmax(out))
            label = CLASS_NAMES[idx] if conf >= CONFIDENCE_THRESH else "None"

            # smoothing
            recent.append(label if label != "None" else None)
            counts = {}
            for v in recent:
                if v is None:
                    continue
                counts[v] = counts.get(v, 0) + 1
            voted = max(counts, key=counts.get) if counts else None

            # debounce
            if voted is not None and voted == last_command:
                consecutive_count += 1
            else:
                consecutive_count = 1
            last_command = voted
            label = voted if consecutive_count >= DEBOUNCE_COUNT else "None"

            # draw rectangle
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # draw label
        cv2.putText(frame, f"{label} {conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition (Camera Only)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(camera_id=args.camera, model_path=args.model)
