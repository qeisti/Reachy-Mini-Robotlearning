import json
import queue

import sounddevice as sd
from vosk import KaldiRecognizer, Model

MODEL_PATH = "../models/vosk-model-small-de-0.15"
SAMPLE_RATE = 16000

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
audio_queue: queue.Queue[bytes] = queue.Queue()


def callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, channels=1, dtype="int16", callback=callback):
    print("Listening... (Strg+C zum Beenden)")
    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            text = json.loads(recognizer.Result()).get("text", "")
            if text:
                print(text)
        else:
            partial = json.loads(recognizer.PartialResult()).get("partial", "")
            if partial:
                print(f"... {partial}", end="\r")
