# receiver.py - run this on Pi A (the listening Pi)
import socket

PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allows quick restarts
server.bind(("0.0.0.0", PORT))
server.listen(1)
print(f"Listening on port {PORT}... (Ctrl+C to stop)")

try:
    while True:
        conn, addr = server.accept()
        print("Connected from", addr[0])
        with conn:
            buffer = ""
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Sender disconnected, waiting for a new connection...")
                    break
                buffer += data.decode()
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    print("Got:", line)
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    server.close()
