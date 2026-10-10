"""
live_yolo_v2.py - patient detection + fatigue monitoring (PERCLOS).

PATIENT = a person (YOLO) who has the target color on them (same color
range as color_detect.py). Fatigue analysis only runs on the patient:
the face is looked for inside the patient's box, so other people in the
room are ignored. With no patient, fatigue waits.

PERCLOS = share of the last 60 s the eyes were mostly closed, measured with
MediaPipe face landmarks against the person's own open-eye baseline
(calibrated for 3 s when a face first appears).

Run:   python live_yolo_v2.py        # camera /dev/video8 (Logitech)
       python live_yolo_v2.py 0      # another camera index
Keys:  q = quit    c = recalibrate (look at the camera, eyes normally open)

Needs: pip install mediapipe   (without it, runs person detection only)
"""
import cv2, sys, time, urllib.request
from collections import deque
from pathlib import Path

import numpy as np
from ultralytics import YOLO

CAM  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MODEL = "yolo11n.pt"

# ---- fatigue settings (same as person_detect.py) ----
PERCLOS_WINDOW_S = 60     # rolling window
CALIBRATION_S = 3         # seconds of open-eye baseline
CLOSED_RATIO = 0.45       # "closed" when EAR < baseline * this (~P80)
DROWSY_PERCLOS = 0.15     # >=15% of the window closed -> drowsy
SEVERE_PERCLOS = 0.30     # >=30% -> very drowsy
LONG_CLOSURE_S = 2.0      # eyes closed this long -> alert

FACE_MODEL_PATH = Path("face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe face mesh eye points: outer corner, 2 upper lid, inner corner, 2 lower lid
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---- patient color (copy your tuned numbers from color_detect.py) ----
COLOR_LOWER = np.array((140, 35, 60))     # OpenCV HSV minimums
COLOR_UPPER = np.array((179, 130, 240))   # OpenCV HSV maximums
MIN_COLOR_PIXELS = 800    # matching pixels needed inside a person's box
PATIENT_HOLD_S = 1.5      # keep "patient" this long after the color drops out (stops flicker)
COLOR_KERNEL = np.ones((5, 5), np.uint8)

GREEN, ORANGE, RED, GRAY, YELLOW, WHITE =(0, 200, 0), (0, 165, 255), (0, 0, 255), (160, 160, 160), (0, 255, 255), (255, 255, 255)


def eye_aspect_ratio(p):
    d = lambda a, b: np.linalg.norm(np.subtract(a, b))
    return (d(p[1], p[5]) + d(p[2], p[4])) / (2.0 * d(p[0], p[3]) + 1e-6)


class FatigueMonitor:
    """Tracks eye closure over time. Call update() once per frame."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not FACE_MODEL_PATH.exists():
            print("downloading face landmark model...")
            part = FACE_MODEL_PATH.with_suffix(".part")
            urllib.request.urlretrieve(FACE_MODEL_URL, part)
            part.rename(FACE_MODEL_PATH)   # only keep it if the download finished

        self.mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )
        self.last_ts = 0
        self.recalibrate()

    def recalibrate(self):
        self.history = deque()     # (time, eyes_closed)
        self.calib = []
        self.calib_start = None
        self.baseline = None
        self.closed_since = None

    def update(self, frame, out):
        """frame: BGR camera image. Draws eye points on `out`.
        Returns (status, color, ear, perclos, warming_up_seconds)."""
        now = time.monotonic()
        h, w = frame.shape[:2]

        ts = max(int(now * 1000), self.last_ts + 1)   # must strictly increase
        self.last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts)

        ear = None
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            pts = lambda idx: [(lm[i].x * w, lm[i].y * h) for i in idx]
            left, right = pts(LEFT_EYE), pts(RIGHT_EYE)
            ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2
            for x, y in left + right:
                cv2.circle(out, (int(x), int(y)), 2, YELLOW, -1)

            if self.baseline is None:
                self.calib_start = self.calib_start or now
                self.calib.append(ear)
                if now - self.calib_start >= CALIBRATION_S:
                    self.baseline = float(np.percentile(self.calib, 75))  # robust to blinks
            else:
                closed = ear < self.baseline * CLOSED_RATIO
                self.history.append((now, closed))
                self.closed_since = (self.closed_since or now) if closed else None
        else:
            self.closed_since = None

        while self.history and now - self.history[0][0] > PERCLOS_WINDOW_S:
            self.history.popleft()

        n = len(self.history)
        perclos = sum(c for _, c in self.history) / n if n else 0.0
        filled = (now - self.history[0][0]) if n else 0.0
        warming = 0 if filled >= PERCLOS_WINDOW_S * 0.9 else filled

        if ear is None:
            status, color = "No face visible", GRAY
        elif self.baseline is None:
            left_s = max(0, CALIBRATION_S - (now - self.calib_start))
            status, color = f"Calibrating - eyes open ({left_s:.0f}s)", YELLOW
        elif self.closed_since and now - self.closed_since >= LONG_CLOSURE_S:
            status, color = f"EYES CLOSED {now - self.closed_since:.0f}s", RED
        elif perclos >= SEVERE_PERCLOS:
            status, color = "VERY DROWSY", RED
        elif perclos >= DROWSY_PERCLOS:
            status, color = "DROWSY", ORANGE
        else:
            status, color = "Alert", GREEN

        return status, color, ear, perclos, warming

    def close(self):
        self.landmarker.close()


def color_mask(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, COLOR_KERNEL)    # remove specks
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, COLOR_KERNEL)   # fill small holes


def find_patient(boxes, mask):
    """Return (x1, y1, x2, y2) of the person with the most target color on them,
    or None if nobody has enough of it."""
    best, best_px = None, MIN_COLOR_PIXELS - 1
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        px = cv2.countNonZero(mask[max(0, y1):y2, max(0, x1):x2])
        if px > best_px:
            best, best_px = (x1, y1, x2, y2), px
    return best


def put(img, text, row, color):
    cv2.putText(img, text, (10, 28 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 28 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


cap = cv2.VideoCapture(CAM)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"could not open /dev/video{CAM}")

print("loading model (first run downloads it)...")
model = YOLO(MODEL)

fatigue = None
try:
    fatigue = FatigueMonitor()
    print("fatigue monitoring on")
except ImportError:
    print("mediapipe not installed - running without fatigue (pip install mediapipe)")
except Exception as e:
    print(f"fatigue monitoring disabled: {e}")

print("running - press q in the window to quit, c to recalibrate")
print("patient = person wearing the target color; fatigue only runs on the patient")

fps, t0 = 0.0, time.time()
last_status = None
patient_box, patient_seen_at, last_patient = None, 0.0, None
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
    put(out, f"people: {n}   {fps:.1f} fps", 0, GREEN)

    # ---- patient = person with the target color on them ----
    found = find_patient(r[0].boxes, color_mask(frame))
    if found:
        patient_box, patient_seen_at = found, now
    patient = patient_box is not None and now - patient_seen_at <= PATIENT_HOLD_S

    if patient:
        x1, y1, x2, y2 = patient_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(out, "PATIENT", (x1 + 4, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    if patient != last_patient:
        print("[patient] DETECTED" if patient else "[patient] not detected")
        last_patient = patient

    if fatigue and not patient:
        put(out, "PATIENT: not detected - fatigue paused", 1, GRAY)
        last_status = "paused"   # so the fatigue status prints again when the patient returns

    if fatigue and patient:
        # look for the face only inside the patient's box (with a small margin),
        # so other people's faces are ignored
        h, w = frame.shape[:2]
        mx, my = (x2 - x1) // 10, (y2 - y1) // 10
        cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
        status, color, ear, perclos, warming = fatigue.update(
            frame[cy1:cy2, cx1:cx2], out[cy1:cy2, cx1:cx2])   # eye dots drawn in place
        base = fatigue.baseline
        put(out, (f"EAR: {ear:.3f}" + (f" (base {base:.3f})" if base else "")) if ear else "EAR: -", 1, WHITE)
        warm = f"  (warming up {warming:.0f}/{PERCLOS_WINDOW_S}s)" if warming else ""
        put(out, f"PERCLOS: {perclos * 100:.0f}%{warm}", 2, WHITE)
        put(out, status, 3, color)

        # print only when the status changes, so the terminal stays readable
        key = status
        for prefix in ("Calibrating", "EYES CLOSED"):   # these include a live countdown
            if status.startswith(prefix):
                key = prefix
        if key != last_status:
            print(f"[fatigue] {status}  PERCLOS {perclos * 100:.0f}%")
            last_status = key

    cv2.imshow("YOLO live", out)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("c") and fatigue:
        fatigue.recalibrate()
        print("[fatigue] recalibrating")

cap.release()
cv2.destroyAllWindows()
if fatigue:
    fatigue.close()
