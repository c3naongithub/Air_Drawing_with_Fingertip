import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to store previous finger position
prev_x, prev_y = 0, 0
canvas = None

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror image

    if canvas is None:
        canvas = frame.copy()

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get index fingertip (landmark 8)
            h, w, c = frame.shape
            finger_tip = handLms.landmark[8]
            x, y = int(finger_tip.x * w), int(finger_tip.y * h)

            # Draw line from previous to current
            if prev_x != 0 and prev_y != 0:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0  # Reset if hand disappears

    # Combine webcam feed and drawing
    blended = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Air Drawing", blended)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
