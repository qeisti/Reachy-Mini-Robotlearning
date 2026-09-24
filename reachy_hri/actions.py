"""Aktionsraum des Roboters – fuer ALLE Policies identisch.

Die Regeln waehlen daraus per Tabelle, der Agent bekommt genau diese Namen als
Tools bzw. als erlaubte Werte im Schema. Die Posen stammen aus dem frueheren
``emotions.py`` (siehe ``legacy/``).

Ein Ausdruck besteht aus Keyframes (Antennenwinkel in Grad + Kopf-Offsets), die
der Motion-Layer nicht-blockierend abspielt, waehrend der Kopf weiter dem
Gesicht folgt. ``persistent=True`` heisst: die letzte Pose bleibt als
Grundhaltung stehen (Emotionen). ``persistent=False``: danach zurueck zur
vorherigen Grundhaltung (Gesten wie Winken, Nicken).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Kopf-Offsets: yaw/pitch/roll in Grad, z in mm. Sie werden zum Tracking addiert.
# tracking=0.0 bedeutet: Kopf ignoriert das Gesicht (z. B. traurig nach unten).
EXPRESSIONS: dict[str, list[dict]] = {
    "neutral": [
        {"keyframes": [{"antennas": [-30, 30], "head": {}}], "step_s": 1.0},
    ],
    "freude": [
        {"keyframes": [{"antennas": [-90, 90], "head": {"yaw": 15, "pitch": -10, "z": 10}}], "step_s": 0.6},
        {"keyframes": [{"antennas": [-90, 90], "head": {"yaw": -15, "pitch": -10, "z": 10}}], "step_s": 0.6},
        {"keyframes": [{"antennas": [-60, 60], "head": {"pitch": -15, "z": 15}}], "step_s": 0.6},
    ],
    "angst": [
        {"keyframes": [{"antennas": [-180, 180], "head": {"pitch": 10, "z": -30}}], "step_s": 1.0},
    ],
    "neugierig": [
        {"keyframes": [{"antennas": [-120, 110], "head": {"yaw": 20, "pitch": -5, "roll": 10}}], "step_s": 1.2},
        {"keyframes": [{"antennas": [-120, 110], "head": {"yaw": -20, "pitch": -5, "roll": -10}}], "step_s": 1.2},
    ],
    "traurig": [
        {"keyframes": [{"antennas": [-10, 10], "head": {"pitch": 20, "z": -15}}], "step_s": 1.5, "tracking": 0.0},
    ],
    "vorsichtig": [
        {"keyframes": [{"antennas": [-120, 110], "head": {}}], "step_s": 1.0},
    ],
    "verwirrt": [
        {"keyframes": [{"antennas": [-90, 30], "head": {"yaw": 20, "pitch": -10, "z": 5, "roll": 15}}], "step_s": 0.5},
        {"keyframes": [{"antennas": [30, -90], "head": {"yaw": -20, "pitch": -10, "z": 5, "roll": -15}}], "step_s": 0.5},
    ],
    # --- Gesten (kehren zur Grundhaltung zurueck) ---
    "winken": [
        {
            "keyframes": [
                {"antennas": [0, 0], "head": {"pitch": -5, "z": 5}},
                {"antennas": [-90, 90], "head": {"pitch": -5, "z": 5}},
                {"antennas": [0, 0], "head": {"pitch": -5, "z": 5}},
                {"antennas": [-90, 90], "head": {"pitch": -5, "z": 5}},
                {"antennas": [0, 0], "head": {"pitch": -5, "z": 5}},
            ],
            "step_s": 0.25,
            "persistent": False,
        },
    ],
    "nicken": [
        {
            "keyframes": [
                {"head": {"pitch": 15}},
                {"head": {"pitch": -5}},
                {"head": {"pitch": 15}},
                {"head": {"pitch": 0}},
            ],
            "step_s": 0.2,
            "persistent": False,
        },
    ],
}

ROBOT_EMOTIONS = ["neutral", "freude", "angst", "neugierig", "traurig", "vorsichtig", "verwirrt"]
ROBOT_GESTURES = ["winken", "nicken"]
NO_ACTION = "nichts"
ACTION_NAMES = ROBOT_EMOTIONS + ROBOT_GESTURES + [NO_ACTION]

ACTION_DESCRIPTIONS = {
    "neutral": "entspannte Grundhaltung",
    "freude": "froehlich, Antennen hoch, Kopf leicht angehoben",
    "angst": "aengstlich, zieht Kopf ein, Antennen ganz nach hinten",
    "neugierig": "neugierig, Kopf schief gelegt",
    "traurig": "traurig, Kopf gesenkt, Antennen haengen",
    "vorsichtig": "vorsichtig/abwartend, Antennen angelegt",
    "verwirrt": "verwirrt, Kopf schief, Antennen asymmetrisch",
    "winken": "mit beiden Antennen zurueckwinken",
    "nicken": "zustimmend nicken",
    NO_ACTION: "bewusst nicht reagieren",
}


@dataclass
class Action:
    """Eine vom Policy-Layer gewaehlte Aktion."""

    name: str
    intensity: float = 1.0
    source: str = ""                       # welche Policy / welches Tool
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in ACTION_NAMES:
            raise ValueError(f"Unbekannte Aktion {self.name!r}. Erlaubt: {ACTION_NAMES}")
        self.intensity = float(min(1.0, max(0.0, self.intensity)))


def pick_variant(name: str, rng: random.Random | None = None) -> dict:
    """Zufaellige Pose-Variante, damit Wiederholungen nicht mechanisch wirken."""
    variants = EXPRESSIONS[name]
    return (rng or random).choice(variants)
