import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from hand_tracker import HandTracker, draw_hand

tracker = HandTracker(mode="video")
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    landmarks, is_left = tracker.detect(frame)
    if landmarks:
        draw_hand(frame, landmarks)
        cv2.putText(frame, "Left" if is_left else "Right", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Test webcam - nhan q de thoat", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
tracker.close()