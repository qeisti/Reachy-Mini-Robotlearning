import threading
import time

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
    (0.15, "nah"),
    (0.07, "mittel"),
    (0.0, "weit"),
]

ANTENNAS_NORMAL_DEG = [-30, 20]
ANTENNAS_MIDDLE_DEG = [-120, 110]
ANTENNAS_RETRACTED_DEG = [-180, 180]

# How much yaw/pitch has to change before a new command is sent, and the
# minimum time between two goto_target calls, so the camera loop never
# waits on the robot.
YAW_DEADZONE_DEG = 3.0
PITCH_DEADZONE_DEG = 3.0
MIN_COMMAND_INTERVAL = 0.3

# The body turns horizontally with the head, but lagging behind: every command
# the body eases a fraction of the way toward the current head yaw. Smaller
# alpha = more delay. The body only follows yaw (horizontal), never pitch.
BODY_FOLLOW_ALPHA = 0.3


def categorize_size(box_area: float, frame_area: float) -> str:
    ratio = box_area / frame_area
    for threshold, label in SIZE_CATEGORIES:
        if ratio >= threshold:
            return label
    return SIZE_CATEGORIES[-1][1]


def mover_loop(mini, lock, desired, stop_event):
    last_sent = {"yaw": None, "pitch": None, "size_label": None}
    last_sent_time = 0.0
    body_yaw = 0.0  # lags behind the head yaw for a delayed body turn

    while not stop_event.is_set():
        with lock:
            yaw, pitch, size_label = desired["yaw"], desired["pitch"], desired["size_label"]

        category_changed = size_label != last_sent["size_label"]
        moved_enough = (
            last_sent["yaw"] is None
            or abs(yaw - last_sent["yaw"]) > YAW_DEADZONE_DEG
            or abs(pitch - last_sent["pitch"]) > PITCH_DEADZONE_DEG
        )
        enough_time_passed = time.monotonic() - last_sent_time > MIN_COMMAND_INTERVAL

        if (category_changed or moved_enough) and enough_time_passed:
            if size_label == "nah":
                # Face very close: lean back and retract the antennas.
                pose = create_head_pose(z=-30, mm=True, yaw=yaw, pitch=pitch, degrees=True)
                antennas = ANTENNAS_RETRACTED_DEG
                duration = 1.0
            elif size_label == "mittel":
                pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
                antennas = ANTENNAS_MIDDLE_DEG
                duration = 1.0
            else:
                pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
                antennas = ANTENNAS_NORMAL_DEG
                duration = 0.1

            # Body eases toward the head yaw -> follows horizontally, delayed.
            body_yaw += (yaw - body_yaw) * BODY_FOLLOW_ALPHA

            mini.goto_target(
                head=pose,
                antennas=np.deg2rad(antennas),
                duration=duration,
                method="minjerk",
                body_yaw=np.deg2rad(body_yaw),
            )

            last_sent = {"yaw": yaw, "pitch": pitch, "size_label": size_label}
            last_sent_time = time.monotonic()

        time.sleep(0.02)


base_options = BaseOptions(model_asset_path="face_detector.tflite")
options = FaceDetectorOptions(base_options=base_options)
detector = FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

with ReachyMini(media_backend="no_media") as mini:
    state_lock = threading.Lock()
    desired_state = {"yaw": 0.0, "pitch": 0.0, "size_label": "weit"}
    stop_event = threading.Event()

    mover_thread = threading.Thread(
        target=mover_loop, args=(mini, state_lock, desired_state, stop_event), daemon=True
    )
    mover_thread.start()

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

            with state_lock:
                desired_state["yaw"] = yaw
                desired_state["pitch"] = pitch
                desired_state["size_label"] = size_label

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()
    mover_thread.join(timeout=1.0)

cap.release()
cv2.destroyAllWindows()
