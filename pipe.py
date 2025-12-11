import cv2
import mediapipe as mp
import numpy as np

# --- MediaPipe Hands setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Gesture detection functions ---
def get_finger_states(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    finger_states = []
    for i, tip_id in enumerate(tips_ids):
        if i == 0:  # Thumb
            if hand_landmarks.landmark[tip_id].x < hand_landmarks.landmark[tip_id - 1].x:
                finger_states.append(1)
            else:
                finger_states.append(0)
        else:  # Other fingers
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                finger_states.append(1)
            else:
                finger_states.append(0)
    return finger_states

def classify_gesture(hand_landmarks):
    finger_states = get_finger_states(hand_landmarks)
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # --- UPDATED GO GESTURE ---
    # Go = Thumb Down, and 4 fingers Up → [0,1,1,1,1]
    if finger_states == [0,1,1,1,1]:
        return "Go"

    # Stop = All fingers extended
    elif finger_states == [1,1,1,1,1]:
        return "Stop"

    # Left / Right (same as before)
    elif thumb_tip.x < index_tip.x:
        return "Right"
    elif thumb_tip.x > index_tip.x:
        return "Left"
    else:
        return "Unknown"


# --- Video capture ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    result = hands.process(rgb_frame)
    command = "No hand detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            command = classify_gesture(hand_landmarks)

            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (255, 0, 0), 2)

            cv2.putText(frame, f"Gesture: {command}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("MediaPipe Hands - Gesture Control", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
