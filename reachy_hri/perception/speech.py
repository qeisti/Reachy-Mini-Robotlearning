"""Offline-Spracherkennung mit Vosk in einem Hintergrund-Thread."""

from __future__ import annotations

import json
import queue
import threading

from .. import config

SAMPLE_RATE = 16000


class SpeechListener:
    def __init__(self, model_dir=config.VOSK_MODEL_DIR) -> None:
        from vosk import KaldiRecognizer, Model

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Vosk-Modell fehlt: {model_dir}. Download: "
                "https://alphacephei.com/vosk/models (vosk-model-small-de-0.15) und nach models/ entpacken.")
        self.recognizer = KaldiRecognizer(Model(str(model_dir)), SAMPLE_RATE)
        self._audio: queue.Queue[bytes] = queue.Queue()
        self._texts: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll(self) -> str | None:
        try:
            return self._texts.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            self._audio.put(bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, channels=1,
                               dtype="int16", callback=callback):
            while not self._stop.is_set():
                try:
                    data = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self.recognizer.AcceptWaveform(data):
                    text = json.loads(self.recognizer.Result()).get("text", "")
                    if text:
                        self._texts.put(text)
