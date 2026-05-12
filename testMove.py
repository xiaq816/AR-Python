import socket
import time
from datetime import datetime

def start_server():
    host = '192.168.43.1'  # Server IP (Python side)
    port = 9000             # Port number
    
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
                    message = data.decode('utf-8').strip()
                    parts = message.split(',')
                    
                    if len(parts) == 6:  
                        device_id = parts[0]   
                        timestamp = int(parts[1])
                        trans_x = float(parts[2])  
                        trans_y = float(parts[3]) 
                        trans_z = float(parts[4]) 
                        rot_y = float(parts[5])    
                   
                        human_time = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        
                        print(f"{device_id},{human_time},{trans_x:.4f},{trans_y:.4f},{trans_z:.4f},{rot_y:.4f}\n")
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