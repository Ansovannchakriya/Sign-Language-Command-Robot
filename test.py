import tensorflow as tf
import numpy as np
import cv2

# --- CONFIG ---
TFLITE_MODEL_PATH = "D:\\Sign_Language_Robot\\gesture_model.tflite" 
IMG_SIZE = 224
CLASS_NAMES = ["Go", "Stop", "Left", "Right"]

# --- LOAD TFLITE MODEL ---
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- CAMERA LOOP ---
cap = cv2.VideoCapture(0)  # 0 = default laptop webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    input_data = preprocess_frame(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Prediction
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)

    # Overlay results on frame
    label = f"{CLASS_NAMES[predicted_class]} ({confidence:.2f})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand_Gesture_Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()