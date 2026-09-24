# Durchführungsplan: Regeln vs. LLM-Agent auf dem Reachy Mini

Zwei Phasen, gleiche Software, gleiche Messung:

| Phase | Wo läuft was | Zweck |
|---|---|---|
| **1 · Simulation** | Laptop: MuJoCo-Simulation (Daemon), Wahrnehmung, Policy, Ollama | Alles entwickeln, testen und eine erste vollständige Messung fahren |
| **2 · Reachy Mini Wireless** | Roboter (Raspberry Pi CM4, 4 GB RAM, 16 GB): Daemon, Wahrnehmung, Policy und Ollama **lokal auf dem Roboter** | Gleiche Messung auf der Zielhardware, echte Motoren, eingebaute Kamera |

Beide Policies laufen in beiden Phasen: die **regelbasierte** (`rule`, braucht kein Ollama) und die
**Agenten-Varianten** (`agent_tc`, `agent_so`, `hybrid`, brauchen Ollama).

Konventionen:

* **T1, T2, …** = eigenes Terminal-Fenster. Windows-Befehle für PowerShell, Roboter-Befehle für
  die SSH-Sitzung (bash).
* Alle Befehle der Reihe nach ausführen. `✔` = was danach zu sehen sein sollte.
* Ergebnis-Ordner heißen `experiments/results/<datum>_<tag>`; die Tags unten (`sim_…`,
  `robot_…`) nicht ändern, dann passt der Vergleich am Ende.

Details zur Datenerhebung (Drehbuch, Annotation, Statistik): [docs/DATENERHEBUNG.md](docs/DATENERHEBUNG.md)

---

## Phase 1 · Simulation auf dem Laptop

### 1.1 Einmalige Einrichtung

**T1 (PowerShell)**

```powershell
# Ollama installieren (alternativ Installer von https://ollama.com/download)
winget install Ollama.Ollama

# Projekt aktualisieren
cd C:\Users\victo\myClaude\_code\Uni\S3\ARL\Reachy-Mini-Robotlearning
git pull
conda activate reachy_mini
pip install -r requirements.txt

# Modelle laden: 3b fuer den Laptop, 1.5b/0.5b fuer den spaeteren Vergleich mit dem Roboter
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:0.5b
ollama list
```

✔ `ollama list` zeigt die drei Modelle.

```powershell
# Software-Test ohne Roboter, Kamera und Ollama
python -m pytest tests -q
```

✔ `17 passed`

```powershell
# Geschwindigkeit des Modells auf dem Laptop (Wert "eval rate" notieren)
ollama run qwen2.5:3b --verbose "Sag hallo"
```

### 1.2 Simulation starten (bei jeder Sitzung)

**T1**

```powershell
conda activate reachy_mini
reachy-mini-daemon --sim
```

✔ MuJoCo-Fenster mit dem Roboter öffnet sich. Terminal offen lassen.

**T2**

```powershell
cd C:\Users\victo\myClaude\_code\Uni\S3\ARL\Reachy-Mini-Robotlearning
conda activate reachy_mini
```

### 1.3 Funktionstests (T2)

```powershell
# a) Ausdruecke in der Sim abspielen
python scripts/test_expressions.py

# b) Nur Entscheidungen, deterministisch vs. Agent (ohne Kamera)
python scripts/run.py --policy rule agent_tc agent_so hybrid --source experiments/scenarios/demo.json --robot mock --tag sim_test

# c) Live mit Webcam, deterministisch (60 s, Fenster mit q schliessen)
python scripts/run.py --policy rule --duration 60 --tag sim_live_rule

# d) Live mit Webcam, Agent
python scripts/run.py --policy agent_tc --model qwen2.5:3b --duration 60 --tag sim_live_agent
```

✔ bei b) pro Ereignis eine Zeile `[agent_tc] … -> freude  850.3 ms` ohne `FEHLER`
✔ bei c)/d) der Sim-Roboter folgt deinem Gesicht und reagiert auf Lächeln/Winken

Falls die Webcam nicht gefunden wird: `--source webcam:1`.

### 1.4 Stimulus aufnehmen und prüfen (einmalig, T2)

```powershell
# Drehbuch aus docs/DATENERHEBUNG.md spielen; Leertaste startet/stoppt
python scripts/record_stimulus.py experiments/stimuli/stimulus.mp4 --seconds 75

# Werden alle Drehbuch-Ereignisse erkannt?
python scripts/run.py --policy rule --source experiments/stimuli/stimulus.mp4 --robot mock --tag sim_check
```

✔ In `experiments/results/<…>_sim_check/rule_run00/events.csv` stehen die Ereignisse in der
Reihenfolge des Drehbuchs. Sonst Schwellen in `reachy_hri/config.py` anpassen und wiederholen.

### 1.5 Annotieren und einfrieren (einmalig, T2)

```powershell
copy experiments\annotations_template.json experiments\annotations.json
notepad experiments\annotations.json      # Zeitfenster + akzeptable Reaktionen eintragen (2 Personen)

git add experiments/annotations.json
git commit -m "Annotation und Konfiguration fuer Messung v1"
git tag messung-v1
git push; git push --tags
```

Das Video selbst liegt nicht im Git (zu groß); bei Bedarf separat teilen.

### 1.6 Pilot (T2)

```powershell
python scripts/run.py --policy all --runs 2 --source experiments/stimuli/stimulus.mp4 --no-display --tag sim_pilot
python scripts/compare.py (Get-ChildItem experiments/results/*_sim_pilot | Select-Object -Last 1).FullName --annotations experiments/annotations.json
```

✔ Keine `WARNUNG`-Zeilen, `report.html` im Ordner `report/` sieht plausibel aus.

### 1.7 Hauptmessung Simulation (T2, ca. 2,5 h)

Laptop am Netzteil, andere Programme schließen.

```powershell
# A: Entscheidungs-Benchmark (identische Ereignisse, ohne Kamera)
python scripts/run.py --policy all --runs 10 --source experiments/scenarios/standard.json --robot mock --no-display --model qwen2.5:3b --tag sim_A

# B: Ende-zu-Ende mit Stimulus-Video und Sim
python scripts/run.py --policy all --runs 10 --source experiments/stimuli/stimulus.mp4 --no-display --model qwen2.5:3b --tag sim_B

# Modellgroessen wie spaeter auf dem Roboter (fuer den direkten Vergleich)
python scripts/run.py --policy agent_so --runs 10 --source experiments/scenarios/standard.json --robot mock --no-display --model qwen2.5:1.5b --tag sim_A_1.5b
python scripts/run.py --policy agent_so --runs 10 --source experiments/scenarios/standard.json --robot mock --no-display --model qwen2.5:0.5b --tag sim_A_0.5b
```

### 1.8 Auswertung Simulation (T2)

```powershell
python scripts/compare.py (Get-ChildItem experiments/results/*_sim_A | Select-Object -Last 1).FullName
python scripts/compare.py (Get-ChildItem experiments/results/*_sim_B | Select-Object -Last 1).FullName --annotations experiments/annotations.json
```

✔ Je Ordner `report/report.html`, `stats.csv`, `summary_table.tex`, Plots als PDF.

---

## Phase 2 · Lokal auf dem Reachy Mini Wireless

Auf dem Roboter läuft der Daemon automatisch. Code, Wahrnehmung und Ollama kommen dazu.
Grenzen der Hardware: 4 GB RAM für alles zusammen, ARM-CPU ohne GPU. Deshalb kleinere Modelle
(`qwen2.5:1.5b`, notfalls `0.5b`), ohne Videofenster (`--no-display`), Gesten optional aus.

### 2.1 Verbinden (Laptop, T1)

```powershell
ping reachy-mini.local
ssh pollen@reachy-mini.local        # Passwort: root
```

✔ Prompt `pollen@reachy-mini:~$`. Alle weiteren Befehle in Phase 2 in dieser SSH-Sitzung,
außer es steht „Laptop“ dabei.

### 2.2 Einmalige Einrichtung auf dem Roboter (SSH)

```bash
# Zustand pruefen
reachyminios_check
free -h            # verfuegbarer RAM
df -h /            # freier Speicher (Modelle: 0.5b ~0,4 GB, 1.5b ~1 GB)

# Projekt holen und in die App-Umgebung des Roboters installieren
cd /home/pollen
git clone https://github.com/qeisti/Reachy-Mini-Robotlearning.git
cd Reachy-Mini-Robotlearning
git checkout messung-v1
/venvs/apps_venv/bin/pip install -r requirements.txt
/venvs/apps_venv/bin/python -m pytest tests -q
```

✔ `17 passed`

```bash
# Ollama auf dem Roboter installieren und Modelle laden
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:0.5b
ollama run qwen2.5:1.5b --verbose "Sag hallo"     # "eval rate" notieren und mit dem Laptop vergleichen
```

### 2.3 Stimulus-Video auf den Roboter kopieren (Laptop, T2)

```powershell
cd C:\Users\victo\myClaude\_code\Uni\S3\ARL\Reachy-Mini-Robotlearning
scp experiments/stimuli/stimulus.mp4 experiments/annotations.json pollen@reachy-mini.local:/home/pollen/Reachy-Mini-Robotlearning/experiments/
ssh pollen@reachy-mini.local "mkdir -p /home/pollen/Reachy-Mini-Robotlearning/experiments/stimuli && mv /home/pollen/Reachy-Mini-Robotlearning/experiments/stimulus.mp4 /home/pollen/Reachy-Mini-Robotlearning/experiments/stimuli/"
```

### 2.4 Funktionstests auf dem Roboter (SSH)

```bash
cd /home/pollen/Reachy-Mini-Robotlearning
PY=/venvs/apps_venv/bin/python

# a) Ausdruecke am echten Roboter
$PY scripts/test_expressions.py

# b) Entscheidungen ohne Kamera, deterministisch vs. Agent
$PY scripts/run.py --policy rule agent_so --source experiments/scenarios/demo.json --robot mock --model qwen2.5:1.5b --tag robot_test

# c) Live mit der eingebauten Kamera, deterministisch
$PY scripts/run.py --policy rule --source reachy --no-display --duration 60 --tag robot_live_rule

# d) Live mit der eingebauten Kamera, Agent
$PY scripts/run.py --policy agent_so --source reachy --no-display --model qwen2.5:1.5b --duration 60 --tag robot_live_agent
```

✔ Der Roboter folgt deinem Gesicht und reagiert. Wird es zu langsam oder der RAM knapp
(`free -h` in einer zweiten SSH-Sitzung): `--no-gestures` und/oder `--model qwen2.5:0.5b`.

Wenn es hakt:

* **Keine Kamerabilder** bei `--source reachy`: in `reachy_hri/config.py` `REACHY_MEDIA_BACKEND = "local"` setzen.
* **`pip install` scheitert an mediapipe**: Python-Version der App-Umgebung prüfen
  (`/venvs/apps_venv/bin/python --version`); MediaPipe gibt es für ARM64-Linux nur für bestimmte
  Python-Versionen. Dann Fehlermeldung notieren; die deterministische Policy und der
  Entscheidungs-Benchmark (Schritt A, `--robot mock`, JSON-Szenario) brauchen MediaPipe nicht.

### 2.5 Hauptmessung auf dem Roboter (SSH, ca. 3 h)

Roboter am Netzteil. Bei langen Läufen `tmux` nutzen, damit die Messung weiterläuft,
falls die SSH-Verbindung abbricht (`tmux new -s messung`, später `tmux attach -t messung`).

```bash
cd /home/pollen/Reachy-Mini-Robotlearning
PY=/venvs/apps_venv/bin/python

# A: Entscheidungs-Benchmark
$PY scripts/run.py --policy all --runs 10 --source experiments/scenarios/standard.json --robot mock --no-display --model qwen2.5:1.5b --tag robot_A
$PY scripts/run.py --policy agent_so --runs 10 --source experiments/scenarios/standard.json --robot mock --no-display --model qwen2.5:0.5b --tag robot_A_0.5b

# B: Ende-zu-Ende mit Stimulus-Video und echten Motoren
$PY scripts/run.py --policy all --runs 10 --source experiments/stimuli/stimulus.mp4 --no-display --model qwen2.5:1.5b --tag robot_B
```

### 2.6 Optional: Agent auf dem Laptop, Roboter über WLAN

Zusätzliche Bedingung „Rechnen ausgelagert“: Ollama läuft auf dem Laptop, alles andere auf dem Roboter.

**Laptop (PowerShell, Ollama-App im Tray vorher beenden)**

```powershell
$env:OLLAMA_HOST = "0.0.0.0"
ollama serve
# Windows-Firewall: eingehend TCP 11434 erlauben. IP des Laptops: ipconfig
```

**Roboter (SSH)**

```bash
$PY scripts/run.py --policy all --runs 10 --source experiments/scenarios/standard.json --robot mock --no-display --model qwen2.5:3b --ollama-url http://<LAPTOP-IP>:11434 --tag robot_A_remote
```

### 2.7 Ergebnisse auf den Laptop holen und vergleichen (Laptop, T2)

```powershell
cd C:\Users\victo\myClaude\_code\Uni\S3\ARL\Reachy-Mini-Robotlearning
scp -r pollen@reachy-mini.local:/home/pollen/Reachy-Mini-Robotlearning/experiments/results/* experiments/results/

# Roboter einzeln
python scripts/compare.py (Get-ChildItem experiments/results/*_robot_A | Select-Object -Last 1).FullName
python scripts/compare.py (Get-ChildItem experiments/results/*_robot_B | Select-Object -Last 1).FullName --annotations experiments/annotations.json

# Simulation vs. Roboter direkt gegenueber (gleiche Policy getrennt nach Ort)
python scripts/compare.py (Get-ChildItem experiments/results/*_sim_A | Select-Object -Last 1).FullName (Get-ChildItem experiments/results/*_robot_A | Select-Object -Last 1).FullName --by-session --out experiments/results/vergleich_sim_robot_A
python scripts/compare.py (Get-ChildItem experiments/results/*_sim_B | Select-Object -Last 1).FullName (Get-ChildItem experiments/results/*_robot_B | Select-Object -Last 1).FullName --by-session --annotations experiments/annotations.json --out experiments/results/vergleich_sim_robot_B
```

✔ `experiments/results/vergleich_sim_robot_*/report/report.html` zeigt jede Policy zweimal
(Sim und Roboter) nebeneinander.

---

## Checkliste

| # | Schritt | erledigt |
|---|---|---|
| 1.1 | Ollama + Modelle auf Laptop, Tests grün | ☐ |
| 1.3 | Funktionstests Sim (deterministisch + Agent) | ☐ |
| 1.4 | Stimulus aufgenommen und geprüft | ☐ |
| 1.5 | Annotation, `git tag messung-v1` | ☐ |
| 1.6 | Pilot ohne Warnungen | ☐ |
| 1.7 | Hauptmessung Sim (A, B, Modellgrößen) | ☐ |
| 1.8 | Auswertung Sim | ☐ |
| 2.2 | Roboter eingerichtet, Tests grün, Ollama läuft | ☐ |
| 2.4 | Funktionstests Roboter (deterministisch + Agent) | ☐ |
| 2.5 | Hauptmessung Roboter (A, B) | ☐ |
| 2.6 | optional: Agent ausgelagert | ☐ |
| 2.7 | Ergebnisse geholt, Vergleich Sim vs. Roboter | ☐ |
