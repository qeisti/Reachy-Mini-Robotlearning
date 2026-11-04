"""R – regelbasierte Verhaltensauswahl (Baseline).

Die Tabellen entsprechen dem Verhalten der frueheren Einzelskripte
``facedetection3.py`` und ``handdetection.py``: Freude spiegeln, auf Wut aengstlich, auf
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


def rules_prompt() -> str:
    """Die Regeln dieser Datei als Text fuer den LLM-Prompt.

    Wird aus denselben Tabellen erzeugt, die ``RuleBasedPolicy`` benutzt: Aendert
    man eine Regel, sehen Regel-Policy und LLM automatisch dasselbe Verhalten.
    Die Werte (nah, mittel, mittelfinger, Thumb_Up ...) stehen genau so im Text,
    wie sie in den Beobachtungen vorkommen.
    """
    def grouped(table: dict[str, str]) -> list[str]:
        by_action: dict[str, list[str]] = {}
        for trigger, action in table.items():
            by_action.setdefault(action, []).append(f"'{trigger}'")
        return [f"{' / '.join(t)} -> {a}" for a, t in by_action.items()]

    far = [k for k in ("weit",) if k not in DISTANCE_REACTIONS]
    lines = ["Feste Regeln (haben Vorrang vor allem anderen):"]
    lines += [f"- Die Person ist '{d}' -> {a}. Das gilt auch, wenn sie auftaucht oder ihr Gesicht sich aendert."
              for d, a in DISTANCE_REACTIONS.items()]
    lines.append(f"- Nur wenn die Person '{'/'.join(far) or 'weit'}' ist, zaehlt ihr Gesichtsausdruck: "
                 + ", ".join(grouped(FACE_EMOTION_REACTIONS)) + ".")
    lines.append(f"- Eine Person taucht auf und ist '{'/'.join(far) or 'weit'}' -> neugierig.")
    lines.append("- Die Person verlaesst das Bild -> neutral.")
    lines.append("- Gesten: " + ", ".join(grouped(GESTURE_REACTIONS)) + ". Andere Gesten -> nichts.")
    lines.append("- Die Person begruesst dich (" + ", ".join(f"'{w}'" for w in GREETING_WORDS) + ") -> winken.")
    lines.append("In allen anderen Faellen, z. B. bei anderen Aussagen, entscheidest du selbst passend "
                 "zur Situation und zum bisherigen Verlauf.")
    return "\n".join(lines)


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
