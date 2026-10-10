from rplidar import RPLidar

lidar = RPLidar('/dev/ttyUSB0', baudrate=115200)
print(lidar.get_info())
print(lidar.get_health())   # should say 'Good'

try:
    for i, scan in enumerate(lidar.iter_scans()):
        print(f"scan {i}: {len(scan)} points, first: {scan[0]}")
        if i >= 10:
            break
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
