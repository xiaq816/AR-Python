import socket
import time

def start_server():
    host = '192.168.43.1'  # Server IP
    port = 6666            # Port number
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"Server started on {host}:{port}. Waiting for connection...")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        print("Client disconnected")
                        break
                    
                    # 直接输出接收到的原始数据
                    received_data = data.decode('utf-8').strip()
                    print(received_data)
                    
                except ConnectionResetError:
                    print("Client forcibly closed the connection")
                    break
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    break

if __name__ == "__main__":
    while True:
        try:
            start_server()
            print("Restarting server in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nServer stopped by user")
            break