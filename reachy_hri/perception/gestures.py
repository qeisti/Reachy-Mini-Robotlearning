"""Hand-Gesten: MediaPipe GestureRecognizer + eigener Winken-Detektor.

Uebernommen aus ``legacy/handdetection.py``.
"""

from __future__ import annotations

import time
from collections import deque

from .. import config
from ..state import PerceptionState

WRIST, MIDDLE_MCP, MIDDLE_PIP = 0, 9, 10
FOLDED_LANDMARKS = [6, 14, 20]


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def is_middle_finger(lm) -> bool:
    ref = _dist(lm[MIDDLE_PIP], lm[WRIST])
    return all(_dist(lm[i], lm[WRIST]) < ref for i in FOLDED_LANDMARKS)


class WaveDetector:
    """Winken = Handgelenk pendelt mehrfach seitlich; Schwellen relativ zur
    Handflaechenlaenge, dadurch distanzunabhaengig."""

    def __init__(self, window=1.2, min_reversals=4, amp_factor=0.6,
                 noise_factor=0.06, cooldown=3.0) -> None:
        self.window, self.min_reversals = window, min_reversals
        self.amp_factor, self.noise_factor, self.cooldown = amp_factor, noise_factor, cooldown
        self.hist: deque = deque()
        self.last_fire = 0.0

    def update(self, wrist_x: float, palm: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        self.hist.append((now, wrist_x, palm))
        while self.hist and now - self.hist[0][0] > self.window:
            self.hist.popleft()
        if len(self.hist) < 6 or palm <= 0:
            return False
        xs = [h[1] for h in self.hist]
        palms = sorted(h[2] for h in self.hist)
        palm_med = palms[len(palms) // 2]
        if palm_med <= 0:
            return False
        amplitude = max(xs) - min(xs)
        noise = self.noise_factor * palm_med
        reversals, last_dir = 0, 0
        for a, b in zip(xs, xs[1:]):
            d = b - a
            if abs(d) < noise:
                continue
            direction = 1 if d > 0 else -1
            if last_dir and direction != last_dir:
                reversals += 1
            last_dir = direction
        if amplitude > self.amp_factor * palm_med and reversals >= self.min_reversals \
                and now - self.last_fire > self.cooldown:
            self.last_fire = now
            self.hist.clear()
            return True
        return False


class GestureAnalyzer:
    KNOWN = {"Thumb_Up", "Thumb_Down", "Open_Palm", "Victory", "Pointing_Up", "ILoveYou",
             "Closed_Fist", "winken", "mittelfinger"}

    def __init__(self, model_path=config.GESTURE_RECOGNIZER_MODEL) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)), num_hands=1)
        self.recognizer = GestureRecognizer.create_from_options(options)
        self.wave = WaveDetector()

    def analyze(self, image, st: PerceptionState) -> None:
        result = self.recognizer.recognize(image)
        if not result.hand_landmarks:
            return
        lm = result.hand_landmarks[0]
        gestures = result.gestures[0]
        name = gestures[0].category_name if gestures else None
        if is_middle_finger(lm):
            name = "mittelfinger"
        if self.wave.update(lm[WRIST].x, _dist(lm[MIDDLE_MCP], lm[WRIST])):
            name = "winken"
        # Offene Hand ohne Bewegung ist kein Ereignis; nur relevante Gesten melden.
        if name in {"Thumb_Up", "Thumb_Down", "winken", "mittelfinger", "Victory", "ILoveYou"}:
            st.gesture = name
