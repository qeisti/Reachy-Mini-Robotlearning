# Ablauf: Programm ausführen (Windows, nach dem Ollama-Setup)

```mermaid
flowchart TD
    A([Start]) --> B["Terminal 1<br/>conda activate reachy_mini<br/>reachy-mini-daemon --sim"]
    B --> C{"Sim-Fenster offen?"}
    C -- nein --> B
    C -- ja --> D["Terminal 2<br/>cd Reachy-Mini-Robotlearning<br/>conda activate reachy_mini"]
    D --> E{"Erstes Mal?"}
    E -- ja --> F["pip install -r requirements.txt"]
    E -- nein --> G
    F --> G{"ollama list zeigt<br/>qwen2.5:3b?"}
    G -- nein --> H["ollama pull qwen2.5:3b"] --> G
    G -- ja --> I["1 · Trockentest ohne Kamera/Roboter<br/>python scripts/run.py --policy rule agent_tc<br/>--source experiments/scenarios/demo.json --robot mock"]
    I --> J{"Agent-Zeilen ohne FEHLER?"}
    J -- nein --> K["Ollama-App läuft?<br/>Modellname korrekt?"] --> I
    J -- ja --> L["2 · Ausdrücke in der Sim<br/>python scripts/test_expressions.py"]
    L --> M["3 · Live mit Webcam<br/>python scripts/run.py --policy rule<br/>python scripts/run.py --policy agent_tc"]
    M --> N{"Verhalten ok?"}
    N -- nein --> O["Anpassen: reachy_hri/config.py,<br/>policies/rule_based.py, policies/agent.py"] --> M
    N -- ja --> P["4 · Stimulus aufnehmen (einmal)<br/>python scripts/record_stimulus.py<br/>experiments/stimuli/stimulus.mp4"]
    P --> Q["5 · Annotieren<br/>annotations_template.json kopieren<br/>→ experiments/annotations.json ausfüllen"]
    Q --> R["6 · Experiment<br/>python scripts/run.py --policy all --runs 10<br/>--source experiments/stimuli/stimulus.mp4<br/>--no-display --tag pilot"]
    R --> S["7 · Auswertung<br/>python scripts/compare.py experiments/results/&lt;session&gt;<br/>--annotations experiments/annotations.json"]
    S --> T["report.html ansehen;<br/>latency.pdf + summary_table.tex ins Paper"]
    T --> U([Fertig])
```

## Hinweise

* **Ollama** läuft unter Windows als App im Hintergrund (Tray-Symbol). Test: `ollama run qwen2.5:3b "hallo"`.
* **Daemon** muss laufen, bevor `run.py` oder `test_expressions.py` startet. Ohne Daemon: `--robot mock`.
* **Webcam** nicht gefunden: `--source webcam:1` (oder 2) probieren.
* **Beenden**: im Videofenster `q`, sonst `Strg+C`.
* **Anderes Modell**: `--model qwen2.5:7b` (vorher `ollama pull qwen2.5:7b`).
* **Ergebnisse** landen in `experiments/results/<datum>_<tag>/`, je Policy und Lauf ein Ordner mit `events.csv`, `resources.csv` und `meta.json`.
