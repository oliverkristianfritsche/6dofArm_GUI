import socket
import struct
import cv2
import numpy as np
import threading
from queue import Queue
from image_utils import detect_white_spheroids, compute_depth_map
import signal
import sys

class StereoCameraServer:
    def __init__(self, server_ip, ports, focal_length, baseline,depth_offset=0):
        self.server_ip = server_ip
        self.ports = ports
        self.focal_length = focal_length
        self.baseline = baseline
        self.detected_points_queues = {port: Queue() for port in ports}
        self.average_depth_values = Queue()
        self.queue_images = Queue()
        self.running = True
        self.threads = []
        self.depth_offset = depth_offset

    def receive_frame(self, conn):
        try:
            conn.settimeout(5)
            frame_size_data = conn.recv(4)
            if not frame_size_data:
                print("Connection closed by the client")
                return None
            frame_size = struct.unpack('!I', frame_size_data)[0]
            frame_data = b''
            while len(frame_data) < frame_size:
                bytes_to_receive = frame_size - len(frame_data)
                chunk = conn.recv(bytes_to_receive)
                if not chunk:
                    print(f"Received only {len(frame_data)} bytes out of {frame_size}")
                    return None
                frame_data += chunk
            return frame_data
        except Exception as e:
            print(f"Error while receiving frame: {e}")
            return None

    def calculate_depth(self, point_cam1, point_cam2):
        x1, _ = point_cam1
        x2, _ = point_cam2
        disparity = abs(x1 - x2)
        depth = (self.focal_length * self.baseline) / disparity if disparity != 0 else 0
        return depth

    def calculate_depth_advanced(self, stereo_pair):
        left_img, right_img = stereo_pair
        # Convert images to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        left_blurred = cv2.GaussianBlur(left_gray, (5, 5), 0)
        right_blurred = cv2.GaussianBlur(right_gray, (5, 5), 0)
        # Create StereoSGBM object and compute disparity
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=5)
        disparity = stereo.compute(left_blurred, right_blurred).astype(np.float32) / 16.0
        # Calculate depth map
        depth_map = np.zeros(disparity.shape)
        mask = disparity > 0
        depth_map[mask] = (self.focal_length * self.baseline) / (disparity[mask])
        return depth_map
    
    def handle_connection(self, conn, addr, port,radius=5):
        print(f"Connection from {addr}, port: {port}")
        with conn:
            while True:
                frame_data = self.receive_frame(conn)
                if frame_data is None:
                    break
                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Implement your object detection logic here
                detected_points = detect_white_spheroids(img, show=False)
                if detected_points is not None:
                    
                    (x, y)= (detected_points[0][0][0],detected_points[0][0][1])
                    cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
                    self.detected_points_queues[port].put((x, y))
                    if not self.detected_points_queues[8082 if port == 8083 else 8083].empty():
                        other_point = self.detected_points_queues[8082 if port == 8083 else 8083].get_nowait()
                        depth = self.calculate_depth((x, y), other_point)
                        if depth > 0:
                            self.average_depth_values.put(depth)
                            while self.average_depth_values.qsize() > 10:
                                self.average_depth_values.get()
                        avg_depth = sum(list(self.average_depth_values.queue)) / self.average_depth_values.qsize() if self.average_depth_values.qsize() > 0 else 0
                        cv2.putText(img, f"Depth: {avg_depth:.2f} mm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        if port == 8082:
                            self.queue_images.put((img, avg_depth, self.depth_offset, (x, y)))

                if port == 8082:
                    cv2.imshow(f"Camera {port}", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cv2.destroyAllWindows()
    
    def server_thread(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.server_ip, port))
            server_socket.listen()
            server_socket.settimeout(1)  # Set timeout for the accept call
            print(f"Listening on {self.server_ip}:{port}")
            while self.running:
                try:
                    conn, addr = server_socket.accept()
                except socket.timeout:
                    continue  # Continue looping, checking if self.running is False
                t = threading.Thread(target=self.handle_connection, args=(conn, addr, port))
                t.start()
                self.threads.append(t)


    def stop(self):
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join()

def signal_handler(signal, frame):
        print('Signal received, shutting down...')
        stereo_server.stop()
        sys.exit(0)

if __name__ == "__main__":
    SERVER_IP = "0.0.0.0"
    PORTS = [8082, 8083]
    FOCAL_LENGTH = 528  # in pixels
    BASELINE = 25  # in millimeters

    # Create server instance
    stereo_server = StereoCameraServer(SERVER_IP, PORTS, FOCAL_LENGTH, BASELINE)

    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server threads for each port
    for port in PORTS:
        t = threading.Thread(target=stereo_server.server_thread, args=(port,))
        t.start()
        stereo_server.threads.append(t)

    # Wait for all threads to complete
    for t in stereo_server.threads:
        t.join()

    print("Server shutdown gracefully.")
