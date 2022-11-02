import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)

tipIds = [4, 8, 12, 16, 20]

def countFingers(image, hand_landmarks, handNo = 0):
    if hand_landmarks:
        landmarks = hand_landmarks[handNo].landmark

        text0 = ""
        text1 = ""

        handState = 0
        for lm_index in tipIds:
            finger_tip_x = landmarks[lm_index].x
            finger_bottom_x = landmarks[lm_index-2].x

            thumb_tip_y = landmarks[lm_index].y
            thumb_bottom_y = landmarks[lm_index-2].y

            if thumb_tip_y > thumb_bottom_y and finger_tip_x > finger_bottom_x:
                handState = 1
                text0 = f'Tip: {thumb_tip_y}'
                text1 = f'Bottom: {thumb_bottom_y}'

            if thumb_tip_y < thumb_bottom_y and finger_tip_x > finger_bottom_x:
                handState = 0

            if finger_tip_x < finger_bottom_x:
                handState = 2

        if handState == 0:
            text = "Dislike"
        elif handState == 1:
            text = "Like"
        elif handState == 2:
            text = "Neither"

        cv2.putText(image, text0, (150,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(image, text1, (150,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            

def drawHandLandmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    results = hands.process(image)
    hand_landmarks = results.multi_hand_landmarks
    drawHandLandmarks(image, hand_landmarks)
    countFingers(image, hand_landmarks)
    cv2.imshow("Media Controller", image)

    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()