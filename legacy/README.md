# legacy/

Die ursprünglichen Einzelskripte (Stand vor der Umstrukturierung). Sie sind
Referenz für die Implementierung in `reachy_hri/` und bleiben lauffähig.
**Aus diesem Ordner heraus starten**, da die Modellpfade relativ sind:

```bash
cd legacy
python facedetection3.py
```

| Skript | übernommen nach |
|---|---|
| `facedetection3.py` | `reachy_hri/perception/face.py`, `reachy_hri/motion.py`, `policies/rule_based.py` |
| `handdetection.py` | `reachy_hri/perception/gestures.py` |
| `voice.py` | `reachy_hri/perception/speech.py` |
| `emotions.py` | `reachy_hri/actions.py` |
| `test_emotions.py` | `scripts/test_expressions.py` |
| `facedetection.py`, `facedetection2.py`, `mouthdetection.py`, `camera.py`, `audio.py`, `hellp.py`, `goto.py` | Vorstufen / Experimente |
