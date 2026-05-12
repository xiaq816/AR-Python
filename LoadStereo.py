import numpy as np
import cv2
import socket
import struct
import os
from datetime import datetime
from collections import deque

# Configuration
SERVER_IP = '192.168.43.1'  # Listen on all interfaces  实验室地址应该为192.168.5.111
SERVER_PORT = 8808
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
DISPLAY_HEIGHT = 960  # 480*2
FPS_WINDOW_SIZE = 30  # FPS calculation window
SAVE_DIR = "calibration_images"  # Directory to save calibration images
LEFT_DIR = os.path.join(SAVE_DIR, "left")  # Left eye images subfolder
RIGHT_DIR = os.path.join(SAVE_DIR, "right")  # Right eye images subfolder


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
        self.frame_counter = 0
        
        # Create directories if they don't exist
        os.makedirs(LEFT_DIR, exist_ok=True)
        os.makedirs(RIGHT_DIR, exist_ok=True)

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

        try:
            # 1. First receive packet length (4 bytes, little-endian)
            length_bytes = self.recv_exactly(4)
            packet_length = struct.unpack('<I', length_bytes)[0]

            # 2. Receive complete packet
            packet_data = self.recv_exactly(packet_length)

            # 3. Parse header (16 bytes, little-endian)
            header = packet_data[:16]
            timestamp = struct.unpack('<d', header[:8])[0]  # double (8 bytes)
            width = struct.unpack('<i', header[8:12])[0]  # int (4 bytes)
            height = struct.unpack('<i', header[12:16])[0]  # int (4 bytes)

            # 4. Verify and process frame
            frame_data = packet_data[16:]
            expected_size = width * height
            if len(frame_data) != expected_size:
                print(f"[ERROR] Size mismatch! Expected {expected_size}, got {len(frame_data)}")
                return False, None

            # Convert to numpy array (frame is already vertically stacked)
            img_np = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width))

            # Store frame data
            self.current_frame = img_np
            self.current_timestamp = timestamp
            self.current_width = width
            self.current_height = height
            self.frame_times.append(datetime.now())

            return True, self.current_frame
        except ConnectionError as e:
            print(f"[ERROR] Connection lost: {str(e)}")
            self.running = False
            return False, None
        except Exception as e:
            print(f"[ERROR] Frame read failed: {str(e)}")
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

    def calculate_fps(self):
        """Calculate current FPS based on frame times"""
        if len(self.frame_times) < 2:
            return 0.0
        time_diff = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / time_diff.total_seconds()

    def save_calibration_images(self, frame):
        """Save left and right eye images to separate folders"""
        if frame is None:
            return
            
        # Split the frame into left and right images
        height = frame.shape[0]
        if height == DISPLAY_HEIGHT:  # Check if frame is vertically stacked
            left_img = frame[:TARGET_HEIGHT, :]
            right_img = frame[TARGET_HEIGHT:, :]
            
            # Save images with sequential numbering
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            left_path = os.path.join(LEFT_DIR, f"left_{self.frame_counter:04d}.png")
            right_path = os.path.join(RIGHT_DIR, f"right_{self.frame_counter:04d}.png")
            
            cv2.imwrite(left_path, left_img)
            cv2.imwrite(right_path, right_img)
            
            print(f"Saved calibration images: {left_path}, {right_path}")
            self.frame_counter += 1

    def release(self):
        """Release resources (similar to cv2.VideoCapture.release())"""
        self.running = False
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None


if __name__ == '__main__':
    # VideoCapture-like usage
    print("=== Running VideoCapture-like implementation ===")
    cap = StereoReceiver()
    if not cap.videocapture():
        print("Failed to initialize video capture")
        exit(1)

    try:
        while True:
            success, frame = cap.read()
            if success:
                # Print frame info
                print(f"Frame received - TS: {cap.current_timestamp:.3f}, "
                      f"Size: {cap.current_width}x{cap.current_height}, "
                      f"True: {frame.shape[0]}x{frame.shape[1]},"
                      f"FPS: {cap.calculate_fps():.1f}")

                # Display the already stacked frame (left eye on top, right eye on bottom)
                if frame is not None:
                    # Add info overlay
                    display_img = frame.copy()
                    fps = cap.calculate_fps()
                    cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display_img, f"TS: {cap.current_timestamp:.3f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Show which eye is which
                    cv2.putText(display_img, "Left Eye", (10, TARGET_HEIGHT - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_img, "Right Eye", (10, DISPLAY_HEIGHT - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow('Stereo View (Left Top, Right Bottom)', display_img)
                    
                    # Save calibration images when 's' key is pressed
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        cap.save_calibration_images(frame)
            else:
                print("Frame read failed")
                break

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User requested exit")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()