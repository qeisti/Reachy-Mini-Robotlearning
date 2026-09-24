"""Gesicht: Blickziel, Distanz und Emotion aus MediaPipe FaceLandmarker.

Uebernommen aus ``legacy/facedetection3.py``: ein Rechteck aus vier Landmarks
(aeussere Augenwinkel, Mundwinkel) liefert Mittelpunkt (Blickziel) und Flaeche
(Distanz). Die Emotion kommt aus den Blendshape-Koeffizienten.
"""

from __future__ import annotations

from .. import config
from ..state import PerceptionState


def categorize_size(area_ratio: float) -> str:
    for threshold, label in config.SIZE_CATEGORIES:
        if area_ratio >= threshold:
            return label
    return config.SIZE_CATEGORIES[-1][1]


def detect_emotion(blendshapes) -> tuple[str, float]:
    scores = {c.category_name: c.score for c in blendshapes}
    best, best_score = "neutral", 0.0
    for emotion, names in config.EMOTION_BLENDSHAPES.items():
        value = sum(scores.get(n, 0.0) for n in names) / len(names)
        if value > best_score:
            best, best_score = emotion, value
    if best_score < config.EMOTION_THRESHOLD:
        return "neutral", best_score
    return best, best_score


class FaceAnalyzer:
    def __init__(self, model_path=config.FACE_LANDMARKER_MODEL) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.detector = FaceLandmarker.create_from_options(options)
        self.last_rect = None  # fuer die Visualisierung

    def analyze(self, image, st: PerceptionState) -> None:
        result = self.detector.detect(image)
        if not result.face_landmarks:
            st.face_present = False
            self.last_rect = None
            return

        lm = result.face_landmarks[0]
        xs = [lm[i].x for i in config.RECT_LANDMARKS]
        ys = [lm[i].y for i in config.RECT_LANDMARKS]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        self.last_rect = (x_min, y_min, x_max, y_max)

        st.face_present = True
        st.target_yaw = -((cx - 0.5) / 0.5) * config.MAX_YAW_DEG
        st.target_pitch = ((cy - 0.5) / 0.5) * config.MAX_PITCH_DEG
        st.distance = categorize_size((x_max - x_min) * (y_max - y_min))
        st.emotion, st.emotion_score = detect_emotion(result.face_blendshapes[0])
