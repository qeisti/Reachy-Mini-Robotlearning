# Reachy Mini – HRI (VLM & Agents)

Experimentier-Skripte für den **Reachy Mini**: Menschen erkennen und ansehen, auf
Winken und Gesten reagieren, Emotionen über Kopfbewegung und Antennen ausdrücken,
Sprache und Onboard-Audio nutzen.

Alle Skripte verbinden sich über `ReachyMini(...)` mit einem laufenden
**Daemon** und schicken Bewegungsbefehle. Die Wahrnehmung läuft über
**Google MediaPipe** (Gesicht, Hand, Gesten) und **Vosk** (Sprache).

---

## Voraussetzungen

Vollständige Installation siehe `SETUP.md`. Kurzfassung:

- Conda-Env `reachy_mini` (Python 3.12), Pakete: `reachy_mini`, `mediapipe`,
  `opencv-python`, `opencv-contrib-python`, `scipy`, `sounddevice`, `vosk`,
  `numpy`, `mujoco`.
- Zusätzlich über conda-forge (sonst Build-/Laufzeitfehler): `pygobject=3.46`,
  `gstreamer`, `gst-plugins-base`, `gst-plugins-good`.
- Nutzer in Gruppen `video` (Webcam) und `dialout` (Motor-Serial, nur echte HW).

### Modelldateien (im Projektordner)

| Datei | Wofür |
|---|---|
| `face_detector.tflite` | `facedetection.py`, `facedetection2.py` (Bounding-Box) |
| `face_landmarker.task` | `facedetection3.py`, `mouthdetection.py` (478-Punkt-Mesh + Blendshapes) |
| `hand_landmarker.task` | Hand-Landmarks (frühere Variante) |
| `gesture_recognizer.task` | `handdetection.py` (Gesten) |
| `vosk-model-small-de-0.15/` | `voice.py` (deutsche Spracherkennung) |

### Daemon starten (immer zuerst, eigenes Terminal)

```bash
conda activate reachy_mini
reachy-mini-daemon --sim      # MuJoCo-Simulation mit 3D-Fenster
# oder ohne --sim mit echter Hardware
```

Danach in einem zweiten Terminal die Skripte starten. OpenCV-Fenster mit **`q`**
schließen.

---

## Skripte

### Wahrnehmung + Reaktion

#### `facedetection.py`
Einfachstes Gesichts-Tracking über **FaceDetector** (`face_detector.tflite`).
Verfolgt das größte Gesicht, dreht den Kopf zum Mittelpunkt (yaw/pitch), schätzt
über die Box-Größe die Entfernung (nah/mittel/weit) und zieht bei „nah" die
Antennen ein. Single-Thread, sendet direkt im Kamera-Loop.

#### `facedetection2.py`
Wie oben, aber robuster: **Bewegungs-Thread (Mover)** entkoppelt die Roboter-
Befehle vom Kamera-Loop (Deadzone + Mindestintervall, damit die Kamera nie auf
den Roboter wartet). Drei Distanz-Stufen mit unterschiedlichen Antennen; der
**Body dreht horizontal verzögert** mit dem Kopf mit (`BODY_FOLLOW_ALPHA`).

#### `facedetection3.py`  *(aktuellste Variante)*
Vereint Face- und Mouth-Detection über **nur FaceLandmarker** (kein
FaceDetector). Spannt ein **Rechteck aus 4 Landmarks** auf – beide äußeren
Augenwinkel + beide Mundwinkel; dessen **Mittelpunkt = Blickziel**, dessen
**Fläche = Entfernung** (nah/mittel/weit). Kopf folgt dem Punkt, Body dreht
verzögert mit.
Emotionen aus **Blendshapes** (freude/traurig/wütend/überrascht):
- **nah / mittel** → Emotion ignoriert, normales Distanz-Tracking.
- **weit** → Roboter übernimmt die Antennen der erkannten Emotion (statt neutral),
  Kopf trackt weiter.

Threaded Mover wie in `facedetection2.py`. Zeichnet Rechteck, Mittelpunkt und
Mund-Landmarks.

#### `handdetection.py`
Gesten über MediaPipe **GestureRecognizer** (`gesture_recognizer.task`) plus
eigener Wave-Detektor:
- **Thumb_Up → freude**, **Thumb_Down → traurig** (Built-in-Gesten).
- **winken** (custom): trackt die Handgelenk-x-Position über ein Zeitfenster,
  zählt horizontale Richtungswechsel; die Schwellen sind auf die Handflächenlänge
  (Landmark 0→9) skaliert → **distanz-invariant**. → spielt `winken`.

Emotionen laufen im **Background-Thread** (kein Kamera-Lag), Debounce pro
Gesten-Wechsel.

#### `mouthdetection.py`
**FaceLandmarker** mit Blendshapes; zeichnet **nur die Mund-Landmarks**. Leitet
daraus eine Emotion ab (freude/traurig/überrascht/wütend, sonst neutral) und
lässt den Roboter reagieren: neutral→neutral, freude→freude, traurig→traurig,
wütend→traurig, überrascht→neugierig. Reaktion im Background-Thread, Trigger nur
bei Emotionswechsel.

#### `voice.py`
Offline-Spracherkennung mit **Vosk** (deutsches Modell) über einen
`sounddevice`-Mikrofon-Stream (16 kHz). Gibt finale und partielle Transkripte auf
der Konsole aus. Reine Speech-to-Text – noch keine Roboter-Reaktion angebunden.

#### `camera.py`
Minimales Webcam-Grundgerüst (OpenCV + Daemon-Verbindung) als Startpunkt für
eigene Erkennung. Zeigt nur das Kamerabild.

#### `audio.py`
Demo für das **Onboard-Audio** des Reachy Mini: aufnehmen, ggf. resamplen,
abspielen und die **Richtung des Schalls** (Direction of Arrival) auslesen.
Belegt Mikro/Lautsprecher exklusiv, solange es läuft.

### Roboter-Ausdruck

#### `emotions.py`
Zentrale **Emotions-Bibliothek**. `EMOTIONS`-Dict: je Emotion eine Liste von
Pose-Varianten (Kopf-Pose, Antennenwinkel in Grad, Dauer). `play_emotion()`
wählt zufällig eine Variante, damit dieselbe Emotion nicht immer identisch
aussieht, und kann optional der erkannten Gesichtsposition (yaw/pitch) folgen.
Vorhandene Emotionen: `neutral`, `freude`, `angst`, `neugierig`, `traurig`,
`vorsichtig`, `verwirrt`, `winken` (Antennen-Wackel-Sequenz 0→-90, 2×). Alle
anderen Skripte ziehen ihre Posen von hier.

#### `test_emotions.py`
Spielt Emotionen zum Testen ab. Ohne Argument alle nacheinander, mit Argument
nur eine:
```bash
python test_emotions.py            # alle
python test_emotions.py winken     # nur winken
```

### Hilfs- / Demo-Skripte

#### `hellp.py`
Minimaler Verbindungstest: verbindet zum Daemon und wackelt einmal mit den
Antennen. Gut, um zu prüfen, ob Daemon + Roboter laufen.

#### `goto.py`
Snippet/Experiment zu `goto_target` (Kopf-Pose, Antennen, Body-Yaw, Dauer,
Interpolation). **Hinweis:** aktuell nicht lauffähig – referenziert undefinierte
`offset_x`/`offset_y`; als Vorlage/Referenz gedacht.

#### `ideen.txt`
Lose Ideen-/TODO-Notizen (Emotionen global abrufbar, Sprach-Prefixe, Winken,
FaceLandmarker-Folgepunkt).

---

## Typischer Ablauf

```bash
# Terminal 1
conda activate reachy_mini
reachy-mini-daemon --sim

# Terminal 2
conda activate reachy_mini
python facedetection3.py     # Gesichts-Tracking + Emotionen
python handdetection.py      # Gesten (Thumb up/down, Winken)
python mouthdetection.py     # Mund-Emotionen
python voice.py              # Spracherkennung
python test_emotions.py      # Emotionen durchspielen
```

## Hinweise

- **Roboter-Konflikt:** Skripte, die den Kopf tracken (`facedetection*`), und
  solche, die `play_emotion` abspielen, steuern beide Kopf/Antennen. Nicht
  gleichzeitig gegen denselben Roboter laufen lassen, ohne Arbitrierung.
- **IDE-Import-Fehler** („reachy_mini could not be resolved") = falscher
  Interpreter. In VS Code den `reachy_mini`-Conda-Env auswählen.
- **Webcam-Index** in `cv2.VideoCapture(0)` ggf. auf 1/2 ändern.
