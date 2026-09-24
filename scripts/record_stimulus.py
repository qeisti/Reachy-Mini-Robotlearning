"""Nimmt das Stimulus-Video mit der Webcam auf (einmalig, danach fuer alle
Konfigurationen wiederverwendet).

    python scripts/record_stimulus.py experiments/stimuli/stimulus.mp4 --seconds 40

Leertaste startet/stoppt die Aufnahme, q beendet.
"""

import argparse
import time

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--fps", type=float, default=30)
args = ap.parse_args()

cap = cv2.VideoCapture(args.camera)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = None
start = None
print("Leertaste = Aufnahme starten/stoppen, q = beenden")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if writer is not None:
        writer.write(frame)
        elapsed = time.time() - start
        cv2.putText(frame, f"REC {elapsed:5.1f}s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if elapsed > args.seconds:
            break
    cv2.imshow("Stimulus", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            start = time.time()
        else:
            break
if writer is not None:
    writer.release()
    print(f"gespeichert: {args.out}")
cap.release()
cv2.destroyAllWindows()
