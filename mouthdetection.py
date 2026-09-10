import threading

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarksConnections
from reachy_mini import ReachyMini

from emotions import play_emotion

# The FaceLandmarker returns the full 478-point face mesh. We only draw the
# mouth, so keep just the lip connections and the landmark indices they touch.
LIP_CONNECTIONS = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_LIPS]
LIP_INDICES = sorted({i for conn in LIP_CONNECTIONS for i in conn})

# Each emotion is the average of a few blendshape (expression) coefficients.
# The blendshapes are 0..1 per frame; the emotion with the highest average
# above EMOTION_THRESHOLD wins.
EMOTION_BLENDSHAPES = {
    "freude": ["mouthSmileLeft", "mouthSmileRight"],
    "traurig": ["mouthFrownLeft", "mouthFrownRight", "browInnerUp"],
    "ueberrascht": ["jawOpen", "eyeWideLeft", "eyeWideRight", "browInnerUp"],
    "wuetend": ["browDownLeft", "browDownRight", "noseSneerLeft", "noseSneerRight"],
}
EMOTION_THRESHOLD = 0.3

# How the robot reacts to a detected face emotion.
EMOTION_REACTIONS = {
    "neutral": "neutral",
    "freude": "freude",
    "traurig": "traurig",
    "wuetend": "traurig",
    "ueberrascht": "neugierig",
}

# play_emotion blocks until the move finishes, so play it in a background
# thread and let the lock skip a trigger while one is still playing.
emotion_lock = threading.Lock()


def play_async(mini, emotion):
    if not emotion_lock.acquire(blocking=False):
        return
    try:
        play_emotion(mini, emotion)
    finally:
        emotion_lock.release()


def detect_emotion(blendshapes):
    """Pick the strongest emotion from the blendshape scores, or 'neutral'."""
    scores = {cat.category_name: cat.score for cat in blendshapes}
    best_name, best_score = "neutral", 0.0
    for emotion, names in EMOTION_BLENDSHAPES.items():
        value = sum(scores.get(n, 0.0) for n in names) / len(names)
        if value > best_score:
            best_name, best_score = emotion, value
    if best_score < EMOTION_THRESHOLD:
        return "neutral", best_score
    return best_name, best_score


base_options = BaseOptions(model_asset_path="face_landmarker.task")
options = FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=True,  # needed for the expression coefficients
)
detector = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

with ReachyMini(media_backend="no_media") as mini:
    last_emotion = None  # so the robot reacts once per emotion change

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)

        for face_idx, landmarks in enumerate(result.face_landmarks):
            # Emotion from this face's blendshapes.
            emotion, score = detect_emotion(result.face_blendshapes[face_idx])

            # First face drives the robot; react only when the emotion changes.
            if face_idx == 0 and emotion != last_emotion:
                reaction = EMOTION_REACTIONS.get(emotion)
                if reaction is not None:
                    print(f"{emotion} ({score:.2f}) -> {reaction}")
                    threading.Thread(target=play_async, args=(mini, reaction), daemon=True).start()
                last_emotion = emotion

            # Draw only the mouth landmarks.
            pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
            for start, end in LIP_CONNECTIONS:
                cv2.line(frame, pts[start], pts[end], (0, 255, 0), 1)
            for idx in LIP_INDICES:
                cv2.circle(frame, pts[idx], 2, (0, 0, 255), -1)

            cv2.putText(frame, f"{emotion} {score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.imshow("Mouth Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
