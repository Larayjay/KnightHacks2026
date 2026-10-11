"""
ui_link.py - sends live_yolo_v2.py's video and fatigue status to monitor_server.py.

Add to live_yolo_v2.py:
    import ui_link                                   # at the top, after "import cv2"
    ui_link.publish(status=..., perclos=..., ...)    # once per frame (see README)

Importing it is enough for the video: every frame passed to cv2.imshow() is also
sent to the screen UI. Everything happens in a background thread, so if the server
isn't running the fatigue script keeps working normally.

UI_HEADLESS=1 python live_yolo_v2.py   -> don't open the OpenCV window
                                          (use this when the UI is full screen;
                                           stop with Ctrl+C instead of q)
"""
import json
import os
import threading
import time
import urllib.request

import cv2

SERVER = os.environ.get("UI_SERVER", "http://127.0.0.1:8000")
SHOW_WINDOW = os.environ.get("UI_HEADLESS", "0") != "1"
MAX_FPS = 12          # video frames per second sent to the UI
JPEG_QUALITY = 70

_latest = {"frame": None, "status": None}
_wake = threading.Condition()


def _plain(v):
    """numpy numbers/bools -> plain Python so they can go into JSON."""
    return v.item() if hasattr(v, "item") else v


def publish(status=None, perclos=None, ear=None, patient=None, **extra):
    """Send the current fatigue status to the UI. Call once per frame.
    status:  the text from FatigueMonitor.update(), e.g. "Alert", "DROWSY",
             "VERY DROWSY", "EYES CLOSED 3s", "Calibrating - eyes open (2s)",
             or "paused" when no patient is in view
    perclos: 0.0-1.0     ear: eye aspect ratio     patient: True/False"""
    data = {"status": status, "perclos": perclos, "ear": ear, "patient": patient, **extra}
    with _wake:
        _latest["status"] = {k: _plain(v) for k, v in data.items()}
        _wake.notify()


def _post(path, body, ctype):
    req = urllib.request.Request(SERVER + path, data=body, method="POST",
                                 headers={"Content-Type": ctype})
    urllib.request.urlopen(req, timeout=0.5).read()


def _sender():
    last_frame_t, warned = 0.0, False
    while True:
        with _wake:
            _wake.wait(timeout=0.5)
            frame, status = _latest["frame"], _latest["status"]
            _latest["frame"] = _latest["status"] = None
        try:
            if status is not None:
                _post("/api/fatigue", json.dumps(status).encode(), "application/json")
            if frame is not None and time.time() - last_frame_t >= 1 / MAX_FPS:
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    _post("/api/frame", jpg.tobytes(), "image/jpeg")
                    last_frame_t = time.time()
            warned = False
        except Exception:
            if not warned:
                print("[ui_link] can't reach monitor_server.py - is it running? (will keep trying)")
                warned = True
            time.sleep(1)


# Send every frame shown with cv2.imshow() to the UI too.
_real_imshow = cv2.imshow


def _imshow(name, img):
    with _wake:
        _latest["frame"] = img.copy()
        _wake.notify()
    if SHOW_WINDOW:
        _real_imshow(name, img)


cv2.imshow = _imshow
threading.Thread(target=_sender, daemon=True).start()
