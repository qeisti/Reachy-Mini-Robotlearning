# Reachy Mini – Regeln vs. LLM-Agent

Kursprojekt *Advanced Robot Learning* (FH Technikum Wien). Der Reachy Mini
erkennt Menschen, ihre Emotionen und Gesten und reagiert darauf mit Kopf und
Antennen. Untersucht wird, **wer entscheidet, wie er reagiert**:

> *Kann ein lokaler LLM-Agent emotionsresponsives Verhalten auf dem Reachy Mini
> in Echtzeit auswählen, und was bietet er gegenüber einer regelbasierten
> Steuerung?*

Wahrnehmung, Aktionsraum und Motion-Layer sind in allen Konfigurationen
identisch, nur die Verhaltensauswahl wird ausgetauscht:

| Policy     | Paper | Beschreibung |
|------------|-------|--------------|
| `rule`     | R     | Regeltabelle (Baseline, Verhalten aus `legacy/facedetection3.py` + `handdetection.py`) |
| `agent_tc` | A-TC  | LangChain-Agent (`create_agent`) mit Tools, lokales LLM über Ollama |
| `agent_so` | A-SO  | ein LLM-Aufruf mit Structured Output (Pydantic-Schema), kein Agent-Loop |
| `hybrid`   | H     | Reflexe (Person erscheint, Geste, Distanz) per Regel, der Rest per A-SO |

## Architektur

```
Kamera/Video ─► Wahrnehmung (MediaPipe, 30 fps) ─► EventDetector ─► Policy-Worker ─► Aktions-Queue ─► Motion-Layer (50 Hz) ─► Reachy SDK
                Gesicht, Emotion, Distanz,          nur Änderungen    R | A-TC | A-SO | H                Kopf folgt Gesicht,
                Gesten, (Sprache)                   (entprellt)       (eigener Thread)                   Ausdrücke überlagert
```

* Das LLM sitzt **nicht** in der Regelschleife. Das Tracking läuft immer mit
  50 Hz, auch wenn der Agent eine Sekunde nachdenkt.
* Der Agent bekommt **keine Bilder**, sondern den symbolischen Zustand als Text
  (z. B. *„Die Person wirkt jetzt 'freude' … Distanz weit“*) plus die letzten
  fünf Ereignisse als Verlauf.
* Die Tools des Agenten legen die Aktion **sofort** in die Motion-Queue
  (`return_direct=True`, also kein zweiter LLM-Aufruf für eine Abschlussantwort).

## Ordnerstruktur

```
reachy_hri/
  config.py            alle Schwellen, Pfade, LLM-Parameter
  actions.py           Aktionsraum + Posen (vorher emotions.py) – für alle Policies gleich
  state.py             PerceptionState, Event, EventDetector (Entprellung)
  perception/          face.py (FaceLandmarker), gestures.py (Gesten + Winken), speech.py (Vosk)
  motion.py            Motion-Layer: Tracking + nicht-blockierende Ausdrücke (minimum jerk)
  robot.py             Backend: echter Reachy/Sim oder Mock
  policies/            rule_based.py, agent.py (LangChain), hybrid.py
  metrics.py           Zeitstempel je Ereignis + CPU/RAM/GPU-Sampling
  runner.py            verbindet alles, schreibt die Messdaten
  sources.py           Webcam, Video (Echtzeit-Replay), Reachy-Kamera, Ereignis-Skript
scripts/
  run.py               Policies starten / Experimente fahren
  compare.py           Auswertung: Tabellen, Plots (PDF für das Paper), HTML-Report
  record_stimulus.py   Stimulus-Video aufnehmen
  test_expressions.py  Ausdrücke am Roboter/in der Sim durchspielen
experiments/
  scenarios/demo.json  skriptierte Ereignisfolge (reiner Entscheidungstest ohne Kamera)
  annotations_template.json   Vorlage: welche Reaktion ist wann angemessen
  stimuli/             hier das aufgenommene Stimulus-Video ablegen (nicht im Git)
  results/             Messergebnisse (nicht im Git)
models/                MediaPipe-Modelle (+ Vosk-Modell, nicht im Git)
legacy/                die ursprünglichen Einzelskripte (lauffähig aus legacy/ heraus)
tests/                 pytest, inkl. Fake-LLM – läuft ohne Roboter und ohne Ollama
```

## Installation

```bash
conda activate reachy_mini            # Python 3.12, siehe SETUP.md
pip install -r requirements.txt

# Lokales LLM
# Ollama installieren: https://ollama.com/download
ollama pull qwen2.5:3b                # weitere Kandidaten: qwen2.5:1.5b, qwen2.5:7b, llama3.2:3b
```

Optional für Sprache: `vosk-model-small-de-0.15` nach `models/` entpacken.

## Starten

Schritt-für-Schritt-Ablauf als Flowchart: [docs/ABLAUF.md](docs/ABLAUF.md)

```bash
# Terminal 1
reachy-mini-daemon --sim

# Terminal 2 – live mit Webcam
python scripts/run.py --policy rule
python scripts/run.py --policy agent_tc --model qwen2.5:3b
python scripts/run.py --policy hybrid --speech
```

Das Fenster zeigt erkannten Zustand, aktuelle Aktion, Queue-Länge und
„denkt…“, solange der Agent rechnet. Mit `q` beenden.

Ohne Daemon/Roboter: `--robot mock`. Nur die Entscheidungslogik, ohne Kamera:

```bash
python scripts/run.py --policy rule agent_tc agent_so --source experiments/scenarios/demo.json --robot mock
```

## Experiment fahren

1. **Stimulus aufnehmen** (einmal): `python scripts/record_stimulus.py experiments/stimuli/stimulus.mp4`
   Ablauf z. B.: Person tritt ins Bild → lächelt → winkt → kommt nah → wirkt verärgert → geht.
2. **Annotieren:** `experiments/annotations_template.json` kopieren und Zeitfenster
   sowie akzeptable Reaktionen je Ereignis eintragen.
3. **Messen** (Video wird in Echtzeit abgespielt, zu langsame Verarbeitung verwirft Frames wie eine echte Kamera):
   ```bash
   python scripts/run.py --policy all --runs 10 --source experiments/stimuli/stimulus.mp4 \
       --no-display --tag pilot --model qwen2.5:3b
   ```
   Für einen Modellgrößen-Vergleich denselben Befehl mit `--policy agent_so --model qwen2.5:1.5b`
   bzw. `:7b` und anderem `--tag` wiederholen.
4. **Auswerten und gegenüberstellen:**
   ```bash
   python scripts/compare.py experiments/results/<session> [weitere Sessions] \
       --annotations experiments/annotations.json --threshold-ms 1000
   ```
   Ergebnis in `<session>/report/`: `report.html` (alles auf einer Seite),
   `summary.csv`, `summary_table.tex` (direkt ins Paper), sowie `latency`, `latency_timeline`,
   `quality`, `resources` und `actions` jeweils als PDF und PNG.

### Messgrößen

| Größe | Bedeutung |
|---|---|
| `decision_*` | Entscheidungslatenz t2 − t1 (Policy-Start bis erste Aktion) |
| `wait_*` | Wartezeit in der Queue, während die Policy noch beschäftigt war |
| `e2e_*` | Ende-zu-Ende t3 − t0 (Frame mit dem Ereignis bis erster Motorbefehl) |
| `share_e2e_over_…` | Anteil der Reaktionen über der Kontingenzschwelle |
| `invalid_rate` | ungültige Tool-Argumente, Parse-Fehler, keine Aktion |
| `consistency` | wie oft über die Läufe hinweg beim selben Ereignis dieselbe Aktion kommt |
| `agreement_with_rules` | Übereinstimmung mit der Regel-Baseline |
| `appropriateness` | Anteil der Reaktionen in der annotierten Menge (braucht `--annotations`) |
| CPU/RAM/GPU | eigener Prozess und LLM-Server (Ollama) getrennt |

## Anpassen

* **Prompt und Tools des Agenten:** `reachy_hri/policies/agent.py` (`SYSTEM_PROMPT`). Der
  Prompt ist eine experimentelle Variable, also im Paper angeben.
* **Regeln:** `reachy_hri/policies/rule_based.py`
* **Neue Aktion:** in `reachy_hri/actions.py` in `EXPRESSIONS`, `ACTION_DESCRIPTIONS` (und bei
  Emotionen `ROBOT_EMOTIONS`) eintragen. Sie steht dann Regeln und Agent zur Verfügung.
* **Schwellen, Entprellung, LLM-Parameter:** `reachy_hri/config.py`

## Tests

```bash
python -m pytest tests -q
```

Die Tests laufen mit einem Fake-LLM und dem Mock-Roboter, also ohne Ollama, Daemon und Kamera.
