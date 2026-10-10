"""
color_detect.py - detects ONE target color and says DETECTED / NOT DETECTED.

The default range comes from your color_test.py readings of the pink/mauve
object (OpenCV HSV: hue 145-178, saturation 35-97, brightness 84-224),
with a little margin.

Shows:
  - a box around the biggest patch of the target color, and its center
  - DETECTED (green) or NOT DETECTED (red) at the top
  - a small black/white mask in the corner (white = pixels that match)

A "Tuning" window has sliders for the HSV range, so you can adjust it live.
Press p to print the current range, then paste those numbers into the
LOWER / UPPER lines below to keep them.

Run:   python color_detect.py        # camera /dev/video8 (Logitech)
       python color_detect.py 0      # another camera index
Keys:  q = quit    p = print current range
"""
import cv2, sys
from collections import deque

import numpy as np

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 8

# ---- target color range (OpenCV HSV: H 0-179, S 0-255, V 0-255) ----
LOWER = (140, 35, 60)     # hue, saturation, brightness minimums
UPPER = (179, 120, 240)   # hue, saturation, brightness maximums

MIN_AREA = 1500   # pixels; smaller patches are ignored as noise (frame is 640x480)
SMOOTH_N = 5      # frames; "detected" only if seen in most of the last N frames


def nothing(_):
    pass


cap = cv2.VideoCapture(CAM)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"could not open /dev/video{CAM}")

cv2.namedWindow("Color detect")
cv2.namedWindow("Tuning")
for name, val, mx in (("H min", LOWER[0], 179), ("H max", UPPER[0], 179),
                      ("S min", LOWER[1], 255), ("S max", UPPER[1], 255),
                      ("V min", LOWER[2], 255), ("V max", UPPER[2], 255),
                      ("Min area", MIN_AREA, 20000)):
    cv2.createTrackbar(name, "Tuning", val, mx, nothing)

kernel = np.ones((5, 5), np.uint8)
recent = deque(maxlen=SMOOTH_N)
last_state = None
print("running - q to quit, p to print the current range")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]

    g = lambda n: cv2.getTrackbarPos(n, "Tuning")
    lower = np.array([g("H min"), g("S min"), g("V min")])
    upper = np.array([g("H max"), g("S max"), g("V max")])
    min_area = g("Min area")

    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea) if contours else None
    area = cv2.contourArea(biggest) if biggest is not None else 0
    found_now = area >= min_area

    recent.append(found_now)
    detected = sum(recent) > len(recent) / 2   # majority vote -> no flicker

    out = frame.copy()
    if found_now:
        x, y, bw, bh = cv2.boundingRect(biggest)
        cx, cy = x + bw // 2, y + bh // 2
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(out, f"({cx},{cy})  area {int(area)}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    label, color = ("DETECTED", (0, 200, 0)) if detected else ("NOT DETECTED", (0, 0, 255))
    cv2.rectangle(out, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # small mask preview in the bottom-right corner
    small = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (w // 4, h // 4))
    out[h - h // 4:h, w - w // 4:w] = small
    cv2.rectangle(out, (w - w // 4, h - h // 4), (w - 1, h - 1), (255, 255, 255), 1)

    if detected != last_state:   # print only on change
        print("DETECTED" if detected else "not detected")
        last_state = detected

    cv2.imshow("Color detect", out)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("p"):
        print(f"LOWER = ({lower[0]}, {lower[1]}, {lower[2]})")
        print(f"UPPER = ({upper[0]}, {upper[1]}, {upper[2]})")
        print(f"MIN_AREA = {min_area}")

cap.release()
cv2.destroyAllWindows()
