import threading
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python.vision import HandLandmarksConnections
from reachy_mini import ReachyMini

from emotions import play_emotion

# Landmark index pairs to draw as the hand skeleton.
HAND_CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

# Reference landmarks for scale: 0 = wrist, 9 = middle-finger MCP. Their
# distance is the palm length, which shrinks/grows with the hand's distance to
# the camera. Every wave threshold is expressed as a fraction of it, so waving
# is detected the same whether the hand is near or far.
WRIST = 0
MIDDLE_MCP = 9

# Custom "mittelfinger": nur der Mittelfinger ist gestreckt, die anderen sind
# eingeklappt. Eingeklappte Gelenke liegen naeher am Handgelenk als das PIP des
# gestreckten Mittelfingers (10), das hier als Referenzdistanz dient.
MIDDLE_PIP = 10
FOLDED_LANDMARKS = [6, 14, 20]  # Zeigefinger-PIP, Ringfinger-PIP, kleiner Finger-Tip


def is_middle_finger(landmarks):
    def dist_to_wrist(idx):
        return ((landmarks[idx].x - landmarks[WRIST].x) ** 2
                + (landmarks[idx].y - landmarks[WRIST].y) ** 2) ** 0.5

    ref = dist_to_wrist(MIDDLE_PIP)
    return all(dist_to_wrist(i) < ref for i in FOLDED_LANDMARKS)

# Which robot emotion to play for a recognized gesture (built-in + custom).
GESTURE_EMOTIONS = {
    "Thumb_Up": "freude",
    "Thumb_Down": "traurig",
    "winken": "winken",
    "mittelfinger": "traurig",
}

# Sekunden ohne erkannte Geste, bis der Roboter wieder auf neutral faehrt.
GESTURE_TIMEOUT = 2.0


class WaveDetector:
    """Detects waving from the sideways motion of the wrist over time.

    A wave is the wrist swinging left-right repeatedly. We keep a short history
    of the wrist x-position, count how often the horizontal direction reverses,
    and require the swing width to exceed a fraction of the palm length. Scaling
    by the palm length is what makes the detection distance-invariant.
    """

    def __init__(self, window=1.2, min_reversals=4, amp_factor=0.6,
                 noise_factor=0.06, cooldown=3.0):
        self.window = window              # seconds of history to look at
        self.min_reversals = min_reversals  # 4 = two full left-right swings
        self.amp_factor = amp_factor      # swing width >= amp_factor * palm
        self.noise_factor = noise_factor  # ignore jitter < noise_factor * palm
        self.cooldown = cooldown          # min seconds between two detections
        self.hist = deque()               # (time, wrist_x, palm)
        self.last_fire = 0.0

    def update(self, wrist_x, palm):
        now = time.monotonic()
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

        # Swing width relative to hand size.
        amplitude = max(xs) - min(xs)

        # Count horizontal direction reversals, ignoring tiny jitter.
        noise = self.noise_factor * palm_med
        reversals = 0
        last_dir = 0
        for a, b in zip(xs, xs[1:]):
            d = b - a
            if abs(d) < noise:
                continue
            direction = 1 if d > 0 else -1
            if last_dir and direction != last_dir:
                reversals += 1
            last_dir = direction

        if amplitude > self.amp_factor * palm_med and reversals >= self.min_reversals:
            if now - self.last_fire > self.cooldown:
                self.last_fire = now
                self.hist.clear()
                return True
        return False


base_options = BaseOptions(model_asset_path="../models/gesture_recognizer.task")
options = GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = GestureRecognizer.create_from_options(options)

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


cap = cv2.VideoCapture(0)

with ReachyMini(media_backend="no_media") as mini:
    wave = WaveDetector()
    last_gesture = None
    last_gesture_time = time.monotonic()
    is_neutral = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = recognizer.recognize(mp_image)

        triggered = None  # gesture that should fire an emotion this frame
        gesture_seen = False  # eine bekannte Geste ist gerade im Bild

        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            gestures = result.gestures[hand_idx]
            name = gestures[0].category_name if gestures else "None"

           # coords = " ".join(f"{i}:({lm.x:.3f},{lm.y:.3f},{lm.z:.3f})"
           #                   for i, lm in enumerate(landmarks))
           # print(f"Hand {hand_idx} [{name}]: {coords}")

            # Custom "winken" gesture from the first hand's motion. Palm length
            # (0 -> 9) gives the distance-invariant scale for the thresholds.
            if hand_idx == 0:
                palm = ((landmarks[MIDDLE_MCP].x - landmarks[WRIST].x) ** 2
                        + (landmarks[MIDDLE_MCP].y - landmarks[WRIST].y) ** 2) ** 0.5
                if is_middle_finger(landmarks):
                    name = "mittelfinger"
                if wave.update(landmarks[WRIST].x, palm):
                    name = "winken"
                    triggered = "winken"           # wave fires immediately
                elif name != last_gesture:
                    triggered = name               # thumb up/down fire on change
                last_gesture = name

            if name in GESTURE_EMOTIONS:
                gesture_seen = True

            pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

            wx, wy = pts[0]
            cv2.putText(frame, name, (wx, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if gesture_seen:
            last_gesture_time = time.monotonic()

        emotion = GESTURE_EMOTIONS.get(triggered)
        if emotion is not None:
            print(f"{triggered} -> {emotion}")
            threading.Thread(target=play_async, args=(mini, emotion), daemon=True).start()
            is_neutral = False
        elif not is_neutral and time.monotonic() - last_gesture_time > GESTURE_TIMEOUT:
            print("keine Geste -> neutral")
            threading.Thread(target=play_async, args=(mini, "neutral"), daemon=True).start()
            is_neutral = True
            last_gesture = None  # gleiche Geste soll danach wieder ausloesen

        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
