import cv2
from reachy_mini import ReachyMini

cap = cv2.VideoCapture(0)  # 0 = erste erkannte Webcam, ggf. 1, 2, ... probieren

with ReachyMini(media_backend="no_media") as mini:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("Webcam", frame)
        # hier z.B. Gesichts-/Objekterkennung -> mini.goto_target(...) aufrufen

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
