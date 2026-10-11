#!/usr/bin/env python3
"""
monitor_server.py - the hub behind the 5" screen UI.

It does three things:
  1. Reads the pulse-sensor ESP32 over USB serial (lines like "BPM: 72").
  2. Receives the camera video + fatigue status from live_yolo_v2.py (via ui_link.py).
  3. Serves the UI (index.html) at http://localhost:8000

Setup (once):   pip install pyserial
Run:            python monitor_server.py
Heart port:     found automatically (/dev/ttyUSB* or /dev/ttyACM*). If the motor
                ESP32 is also plugged in by USB, pick the right one, e.g.:
                HEART_PORT=/dev/ttyUSB1 python monitor_server.py
"""
import glob
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import serial  # pyserial
except ImportError:
    serial = None

PORT = int(os.environ.get("UI_PORT", 8000))
HEART_PORT = os.environ.get("HEART_PORT")   # None = auto-detect
HEART_BAUD = 115200
BPM_STALE_S = 5       # show "--" if no beat for this long
FATIGUE_STALE_S = 3   # fatigue script considered stopped after this long
HISTORY_S = 90        # BPM history kept for the little graph

HERE = Path(__file__).resolve().parent
lock = threading.Lock()
new_frame = threading.Condition(lock)
state = {
    "bpm": None, "bpm_t": 0.0, "bpm_hist": [], "heart_port": None,
    "fatigue": None, "fatigue_t": 0.0,
    "frame": None, "frame_t": 0.0,
}


# ---------------- heart rate (ESP32 over USB serial) ----------------
def find_heart_port():
    if HEART_PORT:
        return HEART_PORT
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def heart_reader():
    if serial is None:
        print("[heart] pyserial not installed - run: pip install pyserial")
        return
    bpm_re = re.compile(rb"BPM:\s*(\d+)")
    warned = False
    while True:
        port = find_heart_port()
        if not port:
            if not warned:
                print("[heart] no ESP32 found on USB - waiting for it to be plugged in")
                warned = True
            time.sleep(2)
            continue
        try:
            with serial.Serial(port, HEART_BAUD, timeout=1) as s:
                print(f"[heart] reading {port}")
                warned = False
                with lock:
                    state["heart_port"] = port
                while True:
                    m = bpm_re.search(s.readline())
                    if m:
                        bpm, now = int(m.group(1)), time.time()
                        with lock:
                            state["bpm"], state["bpm_t"] = bpm, now
                            hist = state["bpm_hist"]
                            hist.append((now, bpm))
                            while hist and now - hist[0][0] > HISTORY_S:
                                hist.pop(0)
        except Exception as e:
            print(f"[heart] {port}: {e} - retrying")
            with lock:
                state["heart_port"] = None
            time.sleep(2)


def current_status():
    now = time.time()
    with lock:
        fresh_bpm = now - state["bpm_t"] < BPM_STALE_S
        fresh_fatigue = now - state["fatigue_t"] < FATIGUE_STALE_S
        return {
            "heart": {
                "connected": state["heart_port"] is not None,
                "bpm": state["bpm"] if fresh_bpm else None,
                "history": [[round(t - now, 1), b] for t, b in state["bpm_hist"]],
            },
            "fatigue": state["fatigue"] if fresh_fatigue else None,
            "camera": now - state["frame_t"] < FATIGUE_STALE_S,
        }


# ---------------- web server ----------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.send(200, json.dumps(current_status()).encode(), "application/json")
        elif path == "/video":
            self.stream_video()
        else:
            self.send(404, b"not found", "text/plain")

    def stream_video(self):
        """MJPEG stream: the browser shows it with a plain <img src="/video">."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while True:
                with new_frame:
                    new_frame.wait_for(lambda: state["frame"] is not last, timeout=5)
                    jpg = state["frame"]
                if jpg is None or jpg is last:
                    continue
                last = jpg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/api/frame":
            with new_frame:
                state["frame"], state["frame_t"] = body, time.time()
                new_frame.notify_all()
        elif self.path == "/api/fatigue":
            try:
                data = json.loads(body)
            except ValueError:
                return self.send(400, b"bad json", "text/plain")
            with lock:
                state["fatigue"], state["fatigue_t"] = data, time.time()
        else:
            return self.send(404, b"not found", "text/plain")
        self.send(200, b"ok", "text/plain")


if __name__ == "__main__":
    threading.Thread(target=heart_reader, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print(f"UI running at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
