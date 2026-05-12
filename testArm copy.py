import socket
import time
from datetime import datetime

def start_server():
    host = '192.168.43.1'  # Server IP (Python side)
    port = 6666             # Port number
    
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
                    
                    # Decode and process the message
                    message = data.decode('ascii').strip()
                    parts = message.split(',')
                    
                    if len(parts) == 3:
                        # Format: "2,timestamp,commandCode"
                        message_type = parts[0]
                        timestamp = int(parts[1])
                        command_code = int(parts[2])
                        
                        # Convert timestamp to readable format
                        human_time = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        
                        # Map command codes to names
                        command_name = {
                            100: "START",
                            101: "STOP",
                            102: "PHOTO"
                        }.get(command_code, f"UNKNOWN_COMMAND({command_code})")
                        
                        print(f"Received command: {command_name} at {human_time}")
                    else:
                        print(f"Received malformed message: {message}")
                        
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