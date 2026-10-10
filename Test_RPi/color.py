"""
color_test.py - shows the color the camera sees, with its name and color code.

Looks at a small square in the middle of the frame (or wherever you click)
and shows the color name, hex code (#RRGGBB), RGB and HSV values.

Run:   python color_test.py        # camera /dev/video8 (Logitech)
       python color_test.py 0      # another camera index
Keys:  q = quit    click = check that spot    r = reset to the center
"""
import cv2, sys
import numpy as np

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BOX = 10   # half-size of the sampled square, in pixels (averages out noise)

point = None   # None = center of the frame


def on_click(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)


def color_name(h, s, v):
    """Rough color name from OpenCV HSV (H 0-179, S and V 0-255)."""
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 200 else "gray"
    if h < 8 or h >= 170:
        return "red"
    if h < 20:
        return "orange" if v > 150 else "brown"
    if h < 33:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


def put(img, text, row, color=(255, 255, 255)):
    cv2.putText(img, text, (10, 28 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 28 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


cap = cv2.VideoCapture(CAM)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"could not open /dev/video{CAM}")

cv2.namedWindow("Color")
cv2.setMouseCallback("Color", on_click)
print("running - q to quit, click to pick a spot, r to go back to the center")

last_name = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    x, y = point if point else (w // 2, h // 2)
    x0, x1 = max(0, x - BOX), min(w, x + BOX)
    y0, y1 = max(0, y - BOX), min(h, y + BOX)

    b, g, r = [int(c) for c in frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)]
    hh, ss, vv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    name = color_name(int(hh), int(ss), int(vv))
    hex_code = f"#{r:02X}{g:02X}{b:02X}"

    out = frame.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.rectangle(out, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (0, 0, 0), 1)

    # swatch of the detected color in the top-right corner
    cv2.rectangle(out, (w - 110, 10), (w - 10, 110), (b, g, r), -1)
    cv2.rectangle(out, (w - 110, 10), (w - 10, 110), (255, 255, 255), 2)

    put(out, f"{name.upper()}   {hex_code}", 0)
    put(out, f"RGB: {r}, {g}, {b}", 1)
    put(out, f"HSV: {hh}, {ss}, {vv}", 2)

    if name != last_name:   # print only when the color changes
        print(f"{name:7s} {hex_code}  RGB({r},{g},{b})")
        last_name = name

    cv2.imshow("Color", out)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("r"):
        point = None

cap.release()
cv2.destroyAllWindows()
