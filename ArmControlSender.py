import socket

def start_server(host='192.168.43.1', port=9000):
    print(f"Server starting on {host}:{port}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    conn, addr = server_socket.accept()
    print(f"Connection from {addr}")

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            try:
                message = data.decode('ascii')
                print(f"Received: {message}")

                # 解析数据
                parts = message.strip().split(',')
                if len(parts) == 3:
                    timestamp, action_type, delta = parts
                    print(f"→ Time: {timestamp}, Action: {action_type}, Delta: {delta}")

                    # 在此处执行操作，比如控制机械臂
                else:
                    print("Invalid message format.")

            except Exception as e:
                print("Decode or process error:", e)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        conn.close()
        server_socket.close()
        print("Server stopped.")

if __name__ == "__main__":
    start_server()
