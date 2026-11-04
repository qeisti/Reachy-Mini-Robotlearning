# Aufräumen des Repos (04.11.2026)

Ziel: Dateien entfernen, die keinen Inhalt mehr liefern, und unklare Namen ersetzen,
ohne Funktionalität zu verlieren. Insgesamt sind 32 Dateien (etwa 17 MB) entfernt.
Alles bleibt über die Git-Historie erreichbar, z. B. `git checkout d1740e5 -- legacy/`.

## Entfernt

| Was | Warum |
|---|---|
| `legacy/` (14 Dateien) | Ursprüngliche Einzelskripte, vollständig nach `reachy_hri/` übernommen (Zuordnung stand in `legacy/README.md`). Dazu Vorstufen und Notizen wie `hellp.py` und `ideen.txt`. Kein Code importiert daraus. |
| `models/face_detector.tflite`, `models/hand_landmarker.task` | Nur von `legacy/` genutzt. `reachy_hri` braucht nur `face_landmarker.task` und `gesture_recognizer.task`. |
| `experiments/distill/data.*`, `probe.*`, `Modelfile.v1` | Destillation v1 und ihr Probelauf. Altes Tool-Format, passt nicht mehr zum aktuellen Code. Abgelöst durch v2 (`reachy-fg`). |
| `experiments/distill/logs/train_attempt*.log` | Abgebrochene Trainingsläufe (2× Speicher voll, 1× für TensorBoard abgebrochen), ohne Ergebnis. Die erfolgreichen Trainings- und Eval-Logs bleiben. |
| `paper/archive/` | Frühere Paper-Version zu einem anderen Thema (Kindchenschema) und nicht mehr verwendete Literaturlisten. |
| `.gitignore`: Einträge für `MVSR_Paper_Template*`, `papert.zip`, Vosk im Wurzelordner | Diese Dateien gibt es nicht (mehr). Das Vosk-Modell liegt unter `models/`. |

## Umbenannt bzw. verschoben

| Alt | Neu | Warum |
|---|---|---|
| `experiments/distill/data_v2.*` | `dataset_raw.*` | „v2“ sagt nichts mehr, seit v1 weg ist. Es sind die Roh-Antworten des Lehrermodells. |
| `experiments/distill/data_v2_rules.*` | `dataset.*` | Der finale Trainingsdatensatz (Lehrer + Regelabgleich), aus dem `reachy-fg` trainiert wurde. |
| `scripts/test_expressions.py` | `scripts/play_expressions.py` | Kein Test, sondern ein Abspiel-Werkzeug. Der Präfix `test_` ließ pytest das Skript fälschlich einsammeln. |
| `DURCHFUEHRUNGSPLAN.md` | `docs/DURCHFUEHRUNGSPLAN.md` | Liegt jetzt bei den anderen Anleitungen. |

## Angepasst, damit alles weiter funktioniert

* Standardpfade in `scripts/distill_*.py` zeigen auf `dataset.jsonl` bzw. `dataset_raw.jsonl`.
  Vorher zeigten sie auf die gelöschte v1-Datei bzw. den Probelauf.
* Verweise in `README.md`, `docs/*.md`, `Modelfile`, `pytest.ini` und in den Docstrings von `reachy_hri/` aktualisiert.
* `README.md`: Ordnerstruktur ergänzt (Destillation, `paper/`, `standard.json`). Die Installation beschreibt jetzt
  die `.venv` mit den nötigen Zusatzschritten (mujoco, GStreamer-Typelibs). Vorher verwies sie auf ein `SETUP.md`, das nicht im Repo liegt.
* Nicht geändert: die alten Eval- und Trainings-Logs. Sie nennen noch die damaligen Dateinamen, weil sie dokumentieren, womit damals gemessen wurde.

## Geprüft

* `pytest`: 18 Tests bestanden.
* Alle Skripte starten (`--help`). Ausnahme: `play_expressions.py`, das ohne Argument-Parser sofort einen laufenden Daemon braucht (wie vorher).
* `run.py --policy rule --robot mock --source experiments/scenarios/demo.json` läuft durch.
* `distill_apply_rules.py dataset_raw.jsonl` erzeugt exakt die Entscheidungen in `dataset.jsonl`.
