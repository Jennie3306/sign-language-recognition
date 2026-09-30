import time
import cv2
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"

# 21 điểm mốc: 0 = cổ tay; 4, 8, 12, 16, 20 = đầu ngón tay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),          # ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),     # ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),   # ngón áp út
    (13, 17), (17, 18), (18, 19), (19, 20),  # ngón út
    (0, 17),
]


class HandTracker:
    """Bọc MediaPipe HandLandmarker. mode='video' cho webcam, mode='image' cho ảnh tĩnh."""

    def __init__(self, mode="video", min_conf=0.5):
        running_mode = (mp.tasks.vision.RunningMode.VIDEO if mode == "video"
                        else mp.tasks.vision.RunningMode.IMAGE)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
        )
        self.mode = mode
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.t0 = time.time()

    def detect(self, frame_bgr):
        """Trả về (landmarks, is_left) hoặc (None, None) nếu không thấy tay."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            result = self.landmarker.detect_for_video(image, int((time.time() - self.t0) * 1000))
        else:
            result = self.landmarker.detect(image)
        if not result.hand_landmarks:
            return None, None
        is_left = result.handedness[0][0].category_name == "Left"
        return result.hand_landmarks[0], is_left

    def close(self):
        self.landmarker.close()


def draw_hand(frame, landmarks):
    """Tự vẽ khung xương vì MediaPipe 1.0 không còn hàm vẽ có sẵn."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)