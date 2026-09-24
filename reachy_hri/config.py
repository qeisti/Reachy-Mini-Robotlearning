"""Zentrale Konfiguration: Pfade, Schwellen, Zeitkonstanten.

Alles, was man beim Experimentieren anpassen will, steht hier – nicht verstreut
in den Modulen.
"""

from pathlib import Path

# --- Pfade -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
GESTURE_RECOGNIZER_MODEL = MODELS_DIR / "gesture_recognizer.task"
VOSK_MODEL_DIR = MODELS_DIR / "vosk-model-small-de-0.15"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# --- Gesichts-Tracking (aus facedetection3.py) -------------------------------
# Rechteck aus aeusseren Augenwinkeln und Mundwinkeln: Mittelpunkt = Blickziel,
# Flaeche = Entfernungsmass.
EYE_LEFT, EYE_RIGHT, MOUTH_LEFT, MOUTH_RIGHT = 33, 263, 61, 291
RECT_LANDMARKS = [EYE_LEFT, EYE_RIGHT, MOUTH_LEFT, MOUTH_RIGHT]

MAX_YAW_DEG = 30.0
MAX_PITCH_DEG = 20.0

# Rechteckflaeche (Anteil am Bild) -> Distanzklasse, absteigend geprueft.
SIZE_CATEGORIES = [(0.06, "nah"), (0.025, "mittel"), (0.0, "weit")]

# Blendshapes -> Gesichts-Emotion (Mittelwert der Koeffizienten).
EMOTION_BLENDSHAPES = {
    "freude": ["mouthSmileLeft", "mouthSmileRight"],
    "traurig": ["mouthFrownLeft", "mouthFrownRight", "browInnerUp"],
    "ueberrascht": ["jawOpen", "eyeWideLeft", "eyeWideRight", "browInnerUp"],
    "wuetend": ["browDownLeft", "browDownRight", "noseSneerLeft", "noseSneerRight"],
}
EMOTION_THRESHOLD = 0.1

# --- Motion-Layer ------------------------------------------------------------
MOTION_RATE_HZ = 50
SMOOTHING_ALPHA = 0.4       # Glaettung Blickziel (1.0 = ungefiltert)
BODY_FOLLOW_ALPHA = 0.007   # Body dreht verzoegert mit (klein = mehr Delay)

# --- Ereignis-Erkennung ------------------------------------------------------
EMOTION_DEBOUNCE_S = 0.5    # Emotion muss so lange stabil sein, bevor sie zaehlt
DISTANCE_DEBOUNCE_S = 0.5
FACE_LOST_TIMEOUT_S = 1.0   # so lange kein Gesicht -> "person_left"
GESTURE_COOLDOWN_S = 2.0    # gleiche Geste loest fruehestens nach dieser Zeit erneut aus

# --- LLM / Agent ---------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.0
LLM_NUM_PREDICT = 128       # Obergrenze erzeugter Tokens pro Aufruf
AGENT_RECURSION_LIMIT = 6   # max. Schritte der Agent-Schleife
AGENT_HISTORY = 5           # so viele vergangene Ereignisse/Aktionen sieht der Agent
