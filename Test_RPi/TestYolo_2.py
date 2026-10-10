#!/usr/bin/env python3
"""
TestYolo_1.py - first vision-to-wheels test.

Runs on the Pi. Uses the Pi camera and the team's YOLO person model
(same yolo11n.pt and class filter as vision_model/person_detect.py).

Behavior, deliberately simple for a first test:

    person in frame  ->  both wheels forward, straight
    no person        ->  stop

No steering yet. Steering comes next, once this proves the whole path
works end to end.

Watching from your laptop: the script serves the annotated camera feed
as an MJPEG stream. Open this in any browser on the same network:

    http://<pi-address>:8000

Run it:

    pi$  python3 TestYolo_1.py              # drives the motors
    pi$  python3 TestYolo_1.py --dry-run    # vision only, motors never move

Start with --dry-run and the wheels off the ground.

Fatigue monitoring (from person_detect.py): PERCLOS = share of the last
60 s the eyes were mostly closed, measured with MediaPipe face landmarks
against the person's own open-eye baseline (calibrated for 3 s whenever
a face first appears). Shown on the stream overlay and printed when the
status changes. It never affects the motors.

    pip install mediapipe                    # once, in the venv
    python3 TestYolo_1.py --no-fatigue       # skip it (faster)
    http://<pi-address>:8000/recalibrate     # redo the eye baseline

If mediapipe isn't installed, the script still runs, just without fatigue.
"""

import argparse
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------

MODEL = "yolo11n.pt"     # same model as the vision branch
CONF = 0.5               # same confidence threshold
PERSON_CLASS = 0         # COCO "person"

BASE_SPEED = 45          # deadband is 20 and 30 is already medium,
                         # so the usable band is roughly 20-70
INFER_SIZE = 320         # smaller = faster on the Pi's CPU
STREAM_PORT = 8000

# fatigue (PERCLOS) - same values as person_detect.py
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


# --------------------------------------------------------------------
# fatigue monitor (PERCLOS)
# --------------------------------------------------------------------

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
            print("[fatigue] downloading face landmark model...")
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

    def update(self, frame, annotated):
        """frame: BGR camera image. Draws eye points on `annotated`.
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
                cv2.circle(annotated, (int(x), int(y)), 2, (0, 255, 255), -1)

            if self.baseline is None:
                self.calib_start = self.calib_start or now
                self.calib.append(ear)
                if now - self.calib_start >= CALIBRATION_S:
                    self.baseline = float(np.percentile(self.calib, 75))
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
            status, color = "no face", (160, 160, 160)
        elif self.baseline is None:
            left_s = max(0, CALIBRATION_S - (now - self.calib_start))
            status, color = f"calibrating - eyes open ({left_s:.0f}s)", (0, 255, 255)
        elif self.closed_since and now - self.closed_since >= LONG_CLOSURE_S:
            status, color = f"EYES CLOSED {now - self.closed_since:.0f}s", (0, 0, 255)
        elif perclos >= SEVERE_PERCLOS:
            status, color = "VERY DROWSY", (0, 0, 255)
        elif perclos >= DROWSY_PERCLOS:
            status, color = "DROWSY", (0, 165, 255)
        else:
            status, color = "alert", (0, 200, 0)

        return status, color, ear, perclos, warming

    def close(self):
        self.landmarker.close()


_recalibrate_requested = threading.Event()

# --------------------------------------------------------------------
# shared frame for the MJPEG stream
# --------------------------------------------------------------------

_latest_jpeg = None
_frame_lock = threading.Lock()


def publish(frame):
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _frame_lock:
            _latest_jpeg = buf.tobytes()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # keep the console readable

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = b"""<!doctype html>
<html><head><title>Pi camera</title>
<style>
  body{margin:0;background:#111;display:flex;align-items:center;
       justify-content:center;height:100vh}
  img{max-width:100%;max-height:100vh}
</style></head>
<body><img src="/stream"></body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if self.path == "/recalibrate":
            _recalibrate_requested.set()
            msg = b"Recalibrating - look at the camera with eyes normally open for 3 s."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        if self.path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        try:
            while True:
                with _frame_lock:
                    jpeg = _latest_jpeg
                if jpeg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_stream(port):
    srv = HTTPServer(("0.0.0.0", port), StreamHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --------------------------------------------------------------------
# camera
# --------------------------------------------------------------------

class Camera:
    """Pi camera via Picamera2, falling back to a USB cam through OpenCV."""

    def __init__(self, width=640, height=480):
        self.picam = None
        self.cap = None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            cfg = self.picam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam.configure(cfg)
            self.picam.start()
            time.sleep(1.0)
            print("[cam] using Picamera2 (CSI camera)")
        except Exception as e:
            print(f"[cam] Picamera2 unavailable ({e}); trying OpenCV")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise SystemExit(
                    "No camera. Check the ribbon cable seating, and that "
                    "'libcamera-hello --list-cameras' sees the sensor."
                )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print("[cam] using OpenCV VideoCapture(0)")

    def read(self):
        if self.picam is not None:
            return self.picam.capture_array()
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="vision only, never touch the motors")
    ap.add_argument("--speed", type=int, default=BASE_SPEED)
    ap.add_argument("--port", type=int, default=STREAM_PORT)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--no-fatigue", action="store_true",
                    help="skip PERCLOS fatigue monitoring")
    args = ap.parse_args()

    fatigue = None
    if not args.no_fatigue:
        try:
            fatigue = FatigueMonitor()
            print("[fatigue] PERCLOS monitoring on")
        except ImportError:
            print("[fatigue] mediapipe not installed - running without fatigue "
                  "(pip install mediapipe)")
        except Exception as e:
            print(f"[fatigue] disabled: {e}")

    bot = None
    if not args.dry_run:
        from robot import Robot
        bot = Robot()
        print(f"[bot] ESP32 ping -> {bot.ping()}")
    else:
        print("[bot] DRY RUN - motors will not move")

    print(f"[yolo] loading {MODEL} (first run downloads it)")
    model = YOLO(MODEL)

    cam = Camera()
    start_stream(args.port)
    print(f"[stream] open http://<pi-address>:{args.port} on your Mac")

    moving = False
    last_fatigue_status = None
    fps_t = time.time()
    fps = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            results = model(frame, classes=[PERSON_CLASS],
                            conf=args.conf, imgsz=INFER_SIZE, verbose=False)
            boxes = results[0].boxes
            seen = len(boxes) > 0

            annotated = results[0].plot()
            h, w = annotated.shape[:2]

            # what a steering command WOULD be, for the next step.
            # not sent anywhere yet.
            turn = 0.0
            if seen:
                areas = [(b.xyxy[0][2] - b.xyxy[0][0]) *
                         (b.xyxy[0][3] - b.xyxy[0][1]) for b in boxes]
                big = boxes[int(np.argmax([float(a) for a in areas]))]
                x1, _, x2, _ = [float(v) for v in big.xyxy[0]]
                cx = (x1 + x2) / 2
                turn = 0.15 * (cx - w / 2)
                cv2.line(annotated, (int(cx), 0), (int(cx), h),
                         (0, 255, 255), 2)

            cv2.line(annotated, (w // 2, 0), (w // 2, h), (120, 120, 120), 1)

            # drive
            if seen and not moving:
                if bot:
                    bot.forward(args.speed)
                moving = True
            elif not seen and moving:
                if bot:
                    bot.stop()
                moving = False

            # overlay
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
            fps_t = now

            status = "PERSON -> FORWARD" if seen else "clear -> stopped"
            color = (0, 255, 0) if seen else (0, 0, 255)
            cv2.rectangle(annotated, (0, 0), (w, 64), (0, 0, 0), -1)
            cv2.putText(annotated, status, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            sub = f"{fps:4.1f} fps   speed={args.speed}   turn would be {turn:+.0f}"
            if args.dry_run:
                sub += "   [DRY RUN]"
            cv2.putText(annotated, sub, (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # fatigue (PERCLOS)
            if fatigue:
                if _recalibrate_requested.is_set():
                    _recalibrate_requested.clear()
                    fatigue.recalibrate()
                    print("[fatigue] recalibrating")
                f_status, f_color, ear, perclos, warming = fatigue.update(frame, annotated)

                line = f"fatigue: {f_status}   PERCLOS {perclos * 100:.0f}%"
                if warming:
                    line += f" (warming up {warming:.0f}/{PERCLOS_WINDOW_S}s)"
                cv2.rectangle(annotated, (0, h - 30), (w, h), (0, 0, 0), -1)
                cv2.putText(annotated, line, (10, h - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, f_color, 2)

                # print only when the status changes, so the console stays readable
                key = f_status
                for prefix in ("calibrating", "EYES CLOSED"):   # these include a live countdown
                    if f_status.startswith(prefix):
                        key = prefix
                if key != last_fatigue_status:
                    print(f"[fatigue] {f_status}  PERCLOS {perclos * 100:.0f}%")
                    last_fatigue_status = key

            publish(annotated)

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if bot:
            bot.close()
        if fatigue:
            fatigue.close()
        cam.close()


if __name__ == "__main__":
    main()
