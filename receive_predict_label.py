import socket

def start_udp_receiver(host='127.0.0.1', port = 8000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    print(f"[UDP Receiver] Listening on {host}:{port}...")

    try:
        while True:
            data, addr = sock.recvfrom(2048)
            print(f"[Received] From {addr}: {data.decode()}")
    except KeyboardInterrupt:
        print("\n[UDP Receiver] Stopped by user.")
    finally:
        sock.close()


if __name__ == "__main__":
    start_udp_receiver()