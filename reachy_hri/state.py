"""Symbolischer Wahrnehmungszustand und Ereignis-Erkennung.

Die Wahrnehmung liefert pro Frame einen ``PerceptionState``. Der
``EventDetector`` macht daraus diskrete Ereignisse (Person erscheint,
Emotion wechselt stabil, Geste, ...). Nur diese Ereignisse gehen an die
Policy – nicht jeder Frame. Das ist die Voraussetzung dafuer, dass ein
langsamer LLM-Agent ueberhaupt mithalten kann.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field

from . import config

EVENT_TYPES = [
    "person_appeared",
    "person_left",
    "emotion_changed",
    "distance_changed",
    "gesture",
    "speech",
]


@dataclass
class PerceptionState:
    t_frame: float                      # perf_counter beim Einlesen des Frames
    video_time: float | None = None     # Position im Stimulus-Video [s] (Replay)
    face_present: bool = False
    target_yaw: float = 0.0             # Blickziel in Grad
    target_pitch: float = 0.0
    distance: str | None = None         # nah | mittel | weit
    emotion: str = "neutral"            # neutral | freude | traurig | ueberrascht | wuetend
    emotion_score: float = 0.0
    gesture: str | None = None          # Thumb_Up | Thumb_Down | winken | mittelfinger | ...
    speech: str | None = None           # finale Vosk-Transkription


@dataclass
class Event:
    id: int
    type: str
    t_frame: float                      # Frame, in dem das Ereignis erkannt wurde
    t_event: float                      # perf_counter bei Erkennung
    video_time: float | None
    data: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    def describe(self) -> str:
        """Kurze, deutschsprachige Beschreibung fuer das LLM."""
        d = self.data
        if self.type == "person_appeared":
            return f"Eine Person ist aufgetaucht (Distanz: {d.get('distance')}, Gesicht: {d.get('emotion')})."
        if self.type == "person_left":
            return "Die Person hat das Bild verlassen."
        if self.type == "emotion_changed":
            return (f"Die Person wirkt jetzt '{d.get('emotion')}' (vorher '{d.get('previous')}', "
                    f"Konfidenz {d.get('score', 0):.2f}, Distanz {d.get('distance')}).")
        if self.type == "distance_changed":
            return f"Die Person ist jetzt '{d.get('distance')}' (vorher '{d.get('previous')}')."
        if self.type == "gesture":
            return f"Die Person zeigt die Geste '{d.get('gesture')}'."
        if self.type == "speech":
            return f"Die Person sagt: \"{d.get('text')}\"."
        return f"Ereignis {self.type}: {d}"


class EventDetector:
    """Zustandsbehafteter Filter: PerceptionState -> Liste neuer Events."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.face_present = False
        self.last_face_seen: float | None = None
        self.emotion = "neutral"
        self.distance: str | None = None
        self._emotion_candidate: tuple[str, float] | None = None
        self._distance_candidate: tuple[str, float] | None = None
        self._gesture_last: dict[str, float] = {}

    def _event(self, etype: str, st: PerceptionState, now: float, **data) -> Event:
        return Event(next(self._ids), etype, st.t_frame, now, st.video_time, data, asdict(st))

    def update(self, st: PerceptionState, now: float | None = None) -> list[Event]:
        now = time.perf_counter() if now is None else now
        events: list[Event] = []

        if st.face_present:
            self.last_face_seen = now
            if not self.face_present:
                self.face_present = True
                self.emotion, self.distance = st.emotion, st.distance
                self._emotion_candidate = self._distance_candidate = None
                events.append(self._event("person_appeared", st, now,
                                          distance=st.distance, emotion=st.emotion))
            else:
                ev = self._debounced("emotion", st.emotion, now, config.EMOTION_DEBOUNCE_S)
                if ev is not None:
                    events.append(self._event("emotion_changed", st, now, emotion=st.emotion,
                                              previous=ev, score=st.emotion_score, distance=st.distance))
                ev = self._debounced("distance", st.distance, now, config.DISTANCE_DEBOUNCE_S)
                if ev is not None:
                    events.append(self._event("distance_changed", st, now, distance=st.distance,
                                              previous=ev, emotion=st.emotion))
        elif self.face_present and self.last_face_seen is not None \
                and now - self.last_face_seen > config.FACE_LOST_TIMEOUT_S:
            self.face_present = False
            self.emotion, self.distance = "neutral", None
            events.append(self._event("person_left", st, now))

        if st.gesture and st.gesture not in ("None", ""):
            last = self._gesture_last.get(st.gesture, -1e9)
            if now - last > config.GESTURE_COOLDOWN_S:
                self._gesture_last[st.gesture] = now
                events.append(self._event("gesture", st, now, gesture=st.gesture,
                                          emotion=st.emotion, distance=st.distance))

        if st.speech:
            events.append(self._event("speech", st, now, text=st.speech,
                                      emotion=st.emotion, distance=st.distance))
        return events

    def _debounced(self, attr: str, value, now: float, hold: float):
        """Meldet einen Wechsel erst, wenn der neue Wert ``hold`` s stabil ist.
        Gibt den vorherigen Wert zurueck, falls ein Wechsel bestaetigt wurde."""
        current = getattr(self, attr)
        cand_attr = f"_{attr}_candidate"
        if value == current:
            setattr(self, cand_attr, None)
            return None
        cand = getattr(self, cand_attr)
        if cand is None or cand[0] != value:
            setattr(self, cand_attr, (value, now))
            return None
        if now - cand[1] >= hold:
            setattr(self, attr, value)
            setattr(self, cand_attr, None)
            return current
        return None
