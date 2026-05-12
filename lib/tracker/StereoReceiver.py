import numpy as np
import cv2
import socket
import struct
import time
from datetime import datetime
from collections import deque
import threading

# Configuration
SERVER_IP = '192.168.43.1'  # Listen on all interfaces # 网线连接应该是'192.168.43.1'
SERVER_PORT = 8808
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
DISPLAY_HEIGHT = 960  # 480*2
FPS_WINDOW_SIZE = 30  # FPS calculation window
MAX_QUEUE_SIZE = 30  # Max number of frames in the queue


class StereoReceiver:
    def __init__(self):
        self.server_socket = None
        self.conn = None
        self.frame_times = deque(maxlen=FPS_WINDOW_SIZE)
        self.running = False
        self.current_frame = None
        self.current_timestamp = 0
        self.current_width = 0
        self.current_height = 0
        self.frame_queue = deque(maxlen=MAX_QUEUE_SIZE)  # Queue to store frames
        self.lock = threading.Lock()  # Lock for thread-safe queue operations
        self.unity_socket = None  # Socket for sending data to Unity
        self.unity_socket_lock = threading.Lock()  # Lock for Unity socket operations

    def connect_to_unity(self, unity_ip='192.168.43.2', unity_port=8888):  # 网线连接应该是'192.168.43.2'
        """Connect to Unity for sending gesture recognition data"""
        try:
            self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.unity_socket.connect((unity_ip, unity_port))
            print(f"[{datetime.now()}] Connected to Unity at {unity_ip}:{unity_port}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to Unity: {str(e)}")
            return False

    def landmarks_to_unity(self, landmarks, predict_label, valid_tracking):
        """Send gesture recognition data to Unity"""
        if not self.unity_socket:
            print("[WARNING] Unity socket not connected, skipping data send")
            return

        valid_tracking_list = []
        for i, vaild in enumerate(valid_tracking):
            if vaild:
                valid_tracking_list.append(1.0)
            else:
                valid_tracking_list.append(0.0)

        landmarks = landmarks.astype(np.float32)
        landmarks_list = landmarks.flatten().tolist()
        landmarks_list.append(float(predict_label))
        landmarks_list = landmarks_list + valid_tracking_list

        try:
            with self.unity_socket_lock:  # Ensure thread-safe socket access
                # 将数据转换为字符串并发送，末尾添加 \n 作为分隔符
                data = ','.join(map(str, landmarks_list)) + '\n'
                self.unity_socket.sendall(str.encode(data))
                # print("成功发送landmarks_list")
                # print(len(landmarks_list))  # 应该是 129
        except Exception as e:
            print(f"[ERROR] Failed to send data to Unity: {e}")

    def videocapture(self):
        """Initialize the video capture (similar to cv2.VideoCapture)"""
        try:
            # Set up server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((SERVER_IP, SERVER_PORT))
            self.server_socket.listen(1)

            print(f"[{datetime.now()}] Stereo Receiver started on {SERVER_IP}:{SERVER_PORT}")
            print(f"[{datetime.now()}] Waiting for connection...")
            self.conn, addr = self.server_socket.accept()
            print(f"[{datetime.now()}] Connected to {addr}")

            self.running = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize video capture: {str(e)}")
            return False

    def read(self):
        """Read a frame (similar to cv2.VideoCapture.read())"""
        if not self.running or not self.conn:
            return False, None

        # Try to get the frame from the queue
        with self.lock:  # Ensure thread-safe operation when accessing the queue
            if len(self.frame_queue) > 0:
                frame_data = self.frame_queue.popleft()  # Pop the oldest frame
                self.current_frame = frame_data['frame']
                self.current_timestamp = frame_data['timestamp']
                self.current_width = frame_data['width']
                self.current_height = frame_data['height']
                return True, self.current_frame
            else:
                return False, None

    def recv_exactly(self, size):
        """Receive exactly size bytes from socket"""
        data = bytearray()
        while len(data) < size:
            remaining = size - len(data)
            packet = self.conn.recv(remaining)
            if not packet:
                raise ConnectionError(f"Connection lost (received {len(data)}/{size} bytes)")
            data.extend(packet)
        return data

    def process_frames(self):
        """Process incoming frames and add them to the queue"""
        try:
            while self.running:
                # 1. First receive packet length (4 bytes, little-endian)
                length_bytes = self.recv_exactly(4)
                packet_length = struct.unpack('<I', length_bytes)[0]

                # 2. Receive complete packet
                packet_data = self.recv_exactly(packet_length)

                # 3. Parse header (16 bytes, little-endian)
                header = packet_data[:16]
                timestamp = struct.unpack('<d', header[:8])[0]  # double (8 bytes)
                width = struct.unpack('<i', header[8:12])[0]  # int (4 bytes)
                height = struct.unpack('<i', header[12:16])[0]  # it (4 bytes)

                # 4. Verify and process frame
                frame_data = packet_data[16:]
                expected_size = width * height
                if len(frame_data) != expected_size:
                    print(f"[ERROR] Size mismatch! Expected {expected_size}, got {len(frame_data)}")
                    continue

                # Convert to numpy array (frame is already vertically stacked)
                img_np = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width))
                img_np = np.fliplr(img_np)  # 水平翻转
                # Add frame data to the queue
                with self.lock:  # Ensure thread-safe operation when accessing the queue
                    self.frame_queue.append({
                        'frame': img_np,
                        'timestamp': timestamp,
                        'width': width,
                        'height': height
                    })

                # Store frame timestamp for FPS calculation
                self.frame_times.append(datetime.now())

        except ConnectionError as e:
            print(f"[ERROR] Connection lost: {str(e)}")
            self.running = False
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {str(e)}")

    def calculate_fps(self):
        """Calculate current FPS based on frame times"""
        if len(self.frame_times) < 2:
            return 0.0
        time_diff = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / time_diff.total_seconds()

    def release(self):
        """Release resources (similar to cv2.VideoCapture.release())"""
        self.running = False
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        if self.unity_socket:
            self.unity_socket.close()
            self.unity_socket = None