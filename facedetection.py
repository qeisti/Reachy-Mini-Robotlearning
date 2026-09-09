import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

# Maximum head rotation (in degrees) when the face is at the edge of the frame.
MAX_YAW_DEG = 30.0
MAX_PITCH_DEG = 20.0

# Face bounding-box area as a fraction of the frame area -> distance category.
SIZE_CATEGORIES = [
    (0.10, "nah"),
    (0.05, "mittel"),
    (0.0, "weit"),
]

ANTENNAS_NORMAL_DEG = [-30 , 30]
ANTENNAS_RETRACTED_DEG = [-180, 180]


def categorize_size(box_area: float, frame_area: float) -> str:
    ratio = box_area / frame_area
    for threshold, label in SIZE_CATEGORIES:
        if ratio >= threshold:
            return label
    return SIZE_CATEGORIES[-1][1]

base_options = BaseOptions(model_asset_path="face_detector.tflite")
options = FaceDetectorOptions(base_options=base_options)
detector = FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

with ReachyMini(media_backend="no_media") as mini:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)

        # Track the largest detected face (closest to the camera).
        largest = None
        for detection in result.detections:
            bbox = detection.bounding_box
            if largest is None or bbox.width * bbox.height > largest.bounding_box.width * largest.bounding_box.height:
                largest = detection

        if largest is not None:
            bbox = largest.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            size_label = categorize_size(w * h, frame_w * frame_h)
            cv2.putText(frame, size_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Normalize the face offset from the frame center to [-1, 1], then
            # scale to a bounded yaw/pitch range instead of using raw pixels.
            offset_x = (center_x - frame_w / 2) / (frame_w / 2)
            offset_y = (center_y - frame_h / 2) / (frame_h / 2)
            yaw = -offset_x * MAX_YAW_DEG   # Negative: face left of center -> turn head left
            pitch = offset_y * MAX_PITCH_DEG

            if size_label == "nah":
                # Face very close: lean back and retract the antennas.
                close_pose = create_head_pose(z=-30, mm=True, yaw=yaw, pitch=pitch, degrees=True)
                mini.goto_target(
                    head=close_pose,
                    antennas=np.deg2rad(ANTENNAS_RETRACTED_DEG),
                    duration=0.1,
                    method="minjerk",
                )
            else:
                target_pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
                mini.goto_target(
                    head=target_pose,
                    antennas=np.deg2rad(ANTENNAS_NORMAL_DEG),
                    duration=0.1,
                    method="minjerk",
                )

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
