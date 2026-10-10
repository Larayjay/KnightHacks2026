import cv2, sys, time
from ultralytics import YOLO

CAM  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MODEL = "yolo11n.pt"

cap = cv2.VideoCapture(CAM)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"could not open /dev/video{CAM}")

print("loading model (first run downloads it)...")
model = YOLO(MODEL)
print("running - press q in the window to quit")

fps, t0 = 0.0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = model(frame, classes=[0], conf=0.5, imgsz=320, verbose=False)
    out = r[0].plot()

    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - t0, 1e-3)
    t0 = now

    n = len(r[0].boxes)
    cv2.putText(out, f"people: {n}   {fps:.1f} fps", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO live", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
