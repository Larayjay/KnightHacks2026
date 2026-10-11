# sender.py - run this on Pi B (the sending Pi)
import socket
import sys

# Pi A's hostname or IP address. You can also pass it on the command line:
#   python3 sender.py 10.42.0.1
HOST = sys.argv[1] if len(sys.argv) > 1 else "pi-left.local"
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to {HOST}:{PORT}...")
sock.connect((HOST, PORT))
print("Connected! Type a message and press Enter. Ctrl+C to quit.")

try:
    while True:
        msg = input("> ")
        sock.sendall((msg + "\n").encode())
except (KeyboardInterrupt, EOFError):
    print("\nClosing.")
finally:
    sock.close()
