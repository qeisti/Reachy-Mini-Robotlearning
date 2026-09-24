"""Wahrnehmung: Frame -> PerceptionState (fuer alle Policies identisch)."""

from __future__ import annotations

import time

from ..state import PerceptionState


class Perception:
    """Buendelt Gesicht, Gesten und (optional) Sprache zu einem Zustand."""

    def __init__(self, gestures: bool = True, speech: bool = False) -> None:
        from .face import FaceAnalyzer

        self.face = FaceAnalyzer()
        self.gestures = None
        self.speech = None
        if gestures:
            from .gestures import GestureAnalyzer
            self.gestures = GestureAnalyzer()
        if speech:
            from .speech import SpeechListener
            self.speech = SpeechListener()
            self.speech.start()

    def process(self, frame_bgr, t_frame: float | None = None,
                video_time: float | None = None) -> PerceptionState:
        import cv2
        import mediapipe as mp

        t_frame = time.perf_counter() if t_frame is None else t_frame
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        st = PerceptionState(t_frame=t_frame, video_time=video_time)
        self.face.analyze(image, st)
        if self.gestures is not None:
            self.gestures.analyze(image, st)
        if self.speech is not None:
            st.speech = self.speech.poll()
        return st

    def close(self) -> None:
        if self.speech is not None:
            self.speech.stop()
        self.face.detector.close()
        if self.gestures is not None:
            self.gestures.recognizer.close()
