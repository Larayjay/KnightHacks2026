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
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

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
    args = ap.parse_args()

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

            publish(annotated)

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if bot:
            bot.close()
        cam.close()


if __name__ == "__main__":
    main()
