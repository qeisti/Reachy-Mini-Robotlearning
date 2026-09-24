"""R – regelbasierte Verhaltensauswahl (Baseline).

Die Tabellen entsprechen dem Verhalten aus ``legacy/facedetection3.py`` und
``legacy/handdetection.py``: Freude spiegeln, auf Wut aengstlich, auf
Ueberraschung neugierig reagieren; nah -> Angst, mittel -> vorsichtig;
Winken erwidern.
"""

from __future__ import annotations

from ..actions import Action
from ..state import Event
from .base import DecisionInfo, Emit, Policy

FACE_EMOTION_REACTIONS = {
    "neutral": "neutral",
    "freude": "freude",
    "traurig": "traurig",
    "wuetend": "angst",
    "ueberrascht": "neugierig",
}
DISTANCE_REACTIONS = {"nah": "angst", "mittel": "vorsichtig"}
GESTURE_REACTIONS = {
    "winken": "winken",
    "Thumb_Up": "freude",
    "Thumb_Down": "traurig",
    "mittelfinger": "traurig",
    "Victory": "freude",
    "ILoveYou": "freude",
}
GREETING_WORDS = ("hallo", "hi", "servus", "guten tag", "hey")


def resolve_emotion(face_emotion: str | None, distance: str | None) -> str:
    if distance in DISTANCE_REACTIONS:
        return DISTANCE_REACTIONS[distance]
    return FACE_EMOTION_REACTIONS.get(face_emotion or "neutral", "neutral")


class RuleBasedPolicy(Policy):
    name = "rule"

    def __init__(self, distance_rules: bool = True) -> None:
        super().__init__()
        self.distance_rules = distance_rules

    def _choose(self, ev: Event) -> str:
        d = ev.data
        dist = d.get("distance") if self.distance_rules else None
        if ev.type == "person_appeared":
            return "neugierig" if dist not in DISTANCE_REACTIONS else resolve_emotion(None, dist)
        if ev.type == "person_left":
            return "neutral"
        if ev.type in ("emotion_changed", "distance_changed"):
            return resolve_emotion(d.get("emotion"), dist)
        if ev.type == "gesture":
            return GESTURE_REACTIONS.get(d.get("gesture"), "nichts")
        if ev.type == "speech":
            text = (d.get("text") or "").lower()
            return "winken" if any(w in text for w in GREETING_WORDS) else "nicken"
        return "nichts"

    def decide(self, event: Event, emit: Emit) -> DecisionInfo:
        action = Action(self._choose(event), source=self.name)
        emit(action)
        self.remember(event, [action])
        return DecisionInfo(n_tool_calls=0)
