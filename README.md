# Patient Monitor screen (5" DSI, 800x480)

Shows the fatigue monitor (camera + PERCLOS) and the pulse-sensor BPM on the Pi's screen.

## Files (put all in the same folder as live_yolo_v2.py)
- `monitor_server.py` - hub: reads the ESP32 heart sensor, serves the UI at http://localhost:8000
- `ui_link.py` - lets live_yolo_v2.py send its video + fatigue status to the hub
- `index.html` - the screen UI (open it directly on a laptop to see a demo with fake data)

## One-time setup
```bash
pip install pyserial
```
Plug the pulse-sensor ESP32 into the Pi by USB. It must be running the BPM sketch that prints `BPM: 72` lines at 115200 baud.

## Edit live_yolo_v2.py (3 lines)
1. At the top, after `import cv2`:
   ```python
   import ui_link
   ```
   That alone sends the camera window's video to the screen.
2. Right after the line that calls `fatigue.update(...)` (it returns `status, color, ear, perclos, ...`):
   ```python
   ui_link.publish(status=status, perclos=perclos, ear=ear, patient=True)
   ```
3. In the branch that draws "PATIENT: not detected - fatigue paused":
   ```python
   ui_link.publish(status="paused", patient=False)
   ```
If your variable names are different, use whatever holds the same values.

## Run (two terminals)
```bash
python monitor_server.py
UI_HEADLESS=1 python live_yolo_v2.py     # UI_HEADLESS=1 hides the OpenCV window; stop with Ctrl+C
```
Then open it full screen on the 5" display:
```bash
chromium-browser --kiosk --noerrdialogs http://localhost:8000
```
(On newer Pi OS the command is `chromium`.)

## If something's wrong
- **"Heart sensor not connected"**: the ESP32 isn't found. Run `ls /dev/ttyUSB* /dev/ttyACM*`. If the motor ESP32 is also on USB, choose the right port: `HEART_PORT=/dev/ttyUSB1 python monitor_server.py`
- **"Waiting for camera"**: live_yolo_v2.py isn't running, or `import ui_link` is missing.
- **"Monitor not running"**: the `ui_link.publish(...)` lines are missing.
- Other laptops on the same Wi-Fi can open `http://<pi-ip>:8000` too.
