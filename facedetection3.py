import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

from emotions import EMOTIONS

# --- Tracking-Rechteck ------------------------------------------------------
# Statt einer Face-Detector-Bounding-Box spannen wir ein Rechteck aus vier
# FaceLandmarker-Punkten auf: die aeusseren Augenwinkel (oben) und die
# Mundwinkel (unten). Mittelpunkt = Blickziel, Flaeche = Entfernungsmass.
EYE_LEFT = 33     # linker aeusserer Augenwinkel
EYE_RIGHT = 263   # rechter aeusserer Augenwinkel
MOUTH_LEFT = 61   # linker Mundwinkel
MOUTH_RIGHT = 291  # rechter Mundwinkel
RECT_LANDMARKS = [EYE_LEFT, EYE_RIGHT, MOUTH_LEFT, MOUTH_RIGHT]

# Nur die Mund-Landmarks werden zusaetzlich gezeichnet (aus mouthdetection).
LIP_CONNECTIONS = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_LIPS]
LIP_INDICES = sorted({i for conn in LIP_CONNECTIONS for i in conn})

# Maximale Kopfdrehung (Grad), wenn das Ziel am Bildrand ist.
MAX_YAW_DEG = 30.0
MAX_PITCH_DEG = 20.0

# Rechteckflaeche als Anteil der Bildflaeche -> Entfernungskategorie. Das
# Augen-Mund-Rechteck ist kleiner als eine volle Gesichtsbox, daher niedrigere
# Schwellen als bei der alten Face-Detection.
SIZE_CATEGORIES = [
    (0.06, "nah"),
    (0.025, "mittel"),
    (0.0, "weit"),
]

ANTENNAS_NORMAL_DEG = [-30, 20]
ANTENNAS_MIDDLE_DEG = [-120, 110]
ANTENNAS_RETRACTED_DEG = [-180, 180]

YAW_DEADZONE_DEG = 3.0
PITCH_DEADZONE_DEG = 3.0
MIN_COMMAND_INTERVAL = 0.3

# Body dreht horizontal verzoegert mit dem Kopf mit (kleiner alpha = mehr Delay).
BODY_FOLLOW_ALPHA = 0.3

# --- Emotion aus Blendshapes (aus mouthdetection) ---------------------------
EMOTION_BLENDSHAPES = {
    "freude": ["mouthSmileLeft", "mouthSmileRight"],
    "traurig": ["mouthFrownLeft", "mouthFrownRight", "browInnerUp"],
    "ueberrascht": ["jawOpen", "eyeWideLeft", "eyeWideRight", "browInnerUp"],
    "wuetend": ["browDownLeft", "browDownRight", "noseSneerLeft", "noseSneerRight"],
}
EMOTION_THRESHOLD = 0.3

# Erkannte Gesichts-Emotion -> Roboter-Emotion.
EMOTION_REACTIONS = {
    "neutral": "neutral",
    "freude": "freude",
    "traurig": "traurig",
    "wuetend": "traurig",
    "ueberrascht": "neugierig",
}

# Die Gesichts-Emotion wirkt NUR bei weit: dann uebernimmt der Roboter die
# Antennen der erkannten Emotion. Bei nah/mittel wird die Gesichts-Emotion
# komplett ignoriert und nur das normale Distanz-Tracking gefahren.
def emotion_antennas(robot_emotion):
    """Antennenwinkel (Grad) der ersten Variante einer Roboter-Emotion."""
    return EMOTIONS.get(robot_emotion, EMOTIONS["neutral"])[0]["antennas"]


def detect_emotion(blendshapes):
    scores = {cat.category_name: cat.score for cat in blendshapes}
    best_name, best_score = "neutral", 0.0
    for emotion, names in EMOTION_BLENDSHAPES.items():
        value = sum(scores.get(n, 0.0) for n in names) / len(names)
        if value > best_score:
            best_name, best_score = emotion, value
    if best_score < EMOTION_THRESHOLD:
        return "neutral", best_score
    return best_name, best_score


def categorize_size(rect_area_ratio: float) -> str:
    for threshold, label in SIZE_CATEGORIES:
        if rect_area_ratio >= threshold:
            return label
    return SIZE_CATEGORIES[-1][1]


def mover_loop(mini, lock, desired, stop_event):
    last_sent = {"yaw": None, "pitch": None, "size_label": None, "emotion": None}
    last_sent_time = 0.0
    body_yaw = 0.0  # laeuft dem Kopf-Yaw hinterher

    while not stop_event.is_set():
        with lock:
            yaw, pitch = desired["yaw"], desired["pitch"]
            size_label, emotion = desired["size_label"], desired["emotion"]

        # Antennen haengen auch von der Emotion ab (nur bei weit), daher bei
        # Emotionswechsel ebenfalls neu senden.
        category_changed = (
            size_label != last_sent["size_label"] or emotion != last_sent["emotion"]
        )
        moved_enough = (
            last_sent["yaw"] is None
            or abs(yaw - last_sent["yaw"]) > YAW_DEADZONE_DEG
            or abs(pitch - last_sent["pitch"]) > PITCH_DEADZONE_DEG
        )
        enough_time_passed = time.monotonic() - last_sent_time > MIN_COMMAND_INTERVAL

        if (category_changed or moved_enough) and enough_time_passed:
            if size_label == "nah":
                # nah/mittel: Emotion ignorieren, normales Distanz-Tracking.
                pose = create_head_pose(z=-30, mm=True, yaw=yaw, pitch=pitch, degrees=True)
                antennas = ANTENNAS_RETRACTED_DEG
                duration = 1.0
            elif size_label == "mittel":
                pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
                antennas = ANTENNAS_MIDDLE_DEG
                duration = 1.0
            else:
                # weit: Roboter nimmt die Antennen der erkannten Emotion an.
                pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
                antennas = emotion_antennas(EMOTION_REACTIONS.get(emotion, "neutral"))
                duration = 0.5

            body_yaw += (yaw - body_yaw) * BODY_FOLLOW_ALPHA

            mini.goto_target(
                head=pose,
                antennas=np.deg2rad(antennas),
                duration=duration,
                method="minjerk",
                body_yaw=np.deg2rad(body_yaw),
            )

            last_sent = {"yaw": yaw, "pitch": pitch, "size_label": size_label, "emotion": emotion}
            last_sent_time = time.monotonic()

        time.sleep(0.02)


base_options = BaseOptions(model_asset_path="face_landmarker.task")
options = FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=True,
)
detector = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

with ReachyMini(media_backend="no_media") as mini:
    state_lock = threading.Lock()
    desired_state = {"yaw": 0.0, "pitch": 0.0, "size_label": "weit", "emotion": "neutral"}
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

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # Rechteck aus den vier Landmarks (normalisierte Koordinaten).
            rxs = [landmarks[i].x for i in RECT_LANDMARKS]
            rys = [landmarks[i].y for i in RECT_LANDMARKS]
            x_min, x_max = min(rxs), max(rxs)
            y_min, y_max = min(rys), max(rys)

            # Mittelpunkt des Rechtecks = Blickziel.
            center_x_n = (x_min + x_max) / 2
            center_y_n = (y_min + y_max) / 2
            # Rechteckflaeche als Bildanteil (Koordinaten sind schon 0..1).
            rect_area_ratio = (x_max - x_min) * (y_max - y_min)
            size_label = categorize_size(rect_area_ratio)

            # Emotion (nur Anzeige, damit sie das Tracking nicht ueberschreibt).
            emotion, score = detect_emotion(result.face_blendshapes[0])

            # --- Zeichnen ---
            x1, y1 = int(x_min * frame_w), int(y_min * frame_h)
            x2, y2 = int(x_max * frame_w), int(y_max * frame_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for i in RECT_LANDMARKS:
                cv2.circle(frame, (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)), 4, (255, 0, 0), -1)
            cx, cy = int(center_x_n * frame_w), int(center_y_n * frame_h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            for idx in LIP_INDICES:
                cv2.circle(frame, (int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)), 2, (0, 0, 255), -1)
            cv2.putText(frame, f"{size_label} | {emotion} {score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Blickziel -> yaw/pitch (Offset vom Bildzentrum, auf Bereich skaliert).
            offset_x = (center_x_n - 0.5) / 0.5
            offset_y = (center_y_n - 0.5) / 0.5
            yaw = -offset_x * MAX_YAW_DEG
            pitch = offset_y * MAX_PITCH_DEG

            with state_lock:
                desired_state["yaw"] = yaw
                desired_state["pitch"] = pitch
                desired_state["size_label"] = size_label
                desired_state["emotion"] = emotion

        cv2.imshow("Face Landmark Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()
    mover_thread.join(timeout=1.0)

cap.release()
cv2.destroyAllWindows()
