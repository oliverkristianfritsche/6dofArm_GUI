import socket
import struct
import cv2
import numpy as np
import time
import threading
from image_utils import detect_white_spheroids, compute_depth_map,calculate_depth
from queue import Queue
SERVER_IP = "0.0.0.0"
PORTS = [8082, 8083]  # Example ports
# Create two queues for the two camera images
queue_cam1 = Queue()
queue_cam2 = Queue()



def receive_frame(conn):
    try:
        conn.settimeout(5)
        # Receive frame size (4 bytes for an int)
        frame_size_data = conn.recv(4)

        if not frame_size_data:
            print("Connection closed by the client")
            return None

        frame_size = struct.unpack('!I', frame_size_data)[0]  # Unpack to integer
        # Receive the actual frame data
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

def calculate_depth(focal_length, baseline, point_cam1, point_cam2):
    """Calculate the depth of a point from the stereo camera setup."""
    x1, _ = point_cam1
    x2, _ = point_cam2
    disparity = abs(x1 - x2)
    depth = (focal_length * baseline) / disparity if disparity != 0 else 0
    return depth

def calculate_depth_advanced(stereo_pair, focal_length, baseline,show=False):
    left_img, right_img = stereo_pair
    # Preprocessing
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    left_blurred = cv2.GaussianBlur(left_gray, (5, 5), 0)
    right_blurred = cv2.GaussianBlur(right_gray, (5, 5), 0)

    # Stereo Rectification (Assuming rectification parameters are computed)
    # This step would typically use cv2.stereoRectify, cv2.initUndistortRectifyMap, etc.

    # Disparity Calculation using Semi-Global Block Matching
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=5)
    disparity = stereo.compute(left_blurred, right_blurred).astype(np.float32) / 16.0

    # Optional: Refine disparity using WLS filter or other methods

    # Depth Calculation
    depth_map = np.zeros(disparity.shape)
    mask = disparity > 0
    depth_map[mask] = (focal_length * baseline) / (disparity[mask])


    return depth_map

def handle_connection(conn, addr, port, my_queue, other_queue,images_queue=None,depth_offset=-100):
    print(f"Connection from {addr}, port: {port}")
    sensor_width_mm = 3.2  # Example sensor width in millimeters
    sensor_height_mm = 2.4  # Example sensor height in millimeters
    image_width_pixels = 352  # CIF resolution width in pixels
    image_height_pixels = 288  # CIF resolution height in pixels

    # Calculate the horizontal field of view (HFOV) and vertical field of view (VFOV)
    hfov_rad = 2 * np.arctan((sensor_width_mm / 2) / focal_length)
    vfov_rad = 2 * np.arctan((sensor_height_mm / 2) / focal_length)
    
    with conn:
        while True:
            frame_data = receive_frame(conn)
            if frame_data is None:
                break
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            detected_point = detect_white_spheroids(img, show=False)
            
            if detected_point is not None:
                # Highlight the detected spheroid
                cv2.circle(img, (detected_point[0][0][0], detected_point[0][0][1]), 10, (0, 255, 0), 2)
                
                # Add the detected point to this camera's queue
                my_queue.put(detected_point)
                
                if not other_queue.empty():
                    other_detected_point = other_queue.get_nowait()
                    depth = calculate_depth(focal_length, baseline, detected_point[0][0], other_detected_point[0][0])
                    
                    # Inside handle_connection, after calculating depth
                    if depth > 0:  # Ensure depth is a valid measurement
                        average_depth_values.put(depth)
                        if average_depth_values.qsize() > 10:  # Keep the last 10 measurements
                            average_depth_values.get()  # Remove the oldest measurement
                    # Calculate X in millimeters
                    px_x = detected_point[0][0][0] - (image_width_pixels / 2)
                    x_mm = np.tan(hfov_rad / 2) * (px_x / (image_width_pixels / 2)) * depth

                    # Calculate Y in millimeters
                    px_y = detected_point[0][0][1] - (image_height_pixels / 2)
                    y_mm = np.tan(vfov_rad / 2) * (px_y / (image_height_pixels / 2)) * depth
                    
                    avg_depth = sum(list(average_depth_values.queue)) / average_depth_values.qsize()
                    cv2.putText(img, f"Avg Depth (Z): {avg_depth+depth_offset:.2f} mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"X: {x_mm:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Y: {y_mm:.2f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if(port==8083):
                        images_queue.put((img,avg_depth,depth_offset,(detected_point[0][0][0], detected_point[0][0][1])))

                    
            cv2.imshow(f"Camera {port}", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()




def server_thread(port,queue, other_queue,images_queue=None,depth_offset=0):
    """Thread for handling connections from one of the cameras."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((SERVER_IP, port))
        server_socket.listen()
        print(f"Listening on {SERVER_IP}:{port}")
        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=handle_connection, args=(conn, addr, port,queue, other_queue,images_queue=images_queue,depth_offset=depth_offset)).start()

SERVER_IP = "0.0.0.0"
PORTS = [8082, 8083]  # Ports for the two cameras
focal_length = 528  # Example focal length in pixels
baseline = 25  # Example baseline in millimeters

# Queues for holding detected points from both cameras
detected_points_queue_cam1 = Queue()
detected_points_queue_cam2 = Queue()
average_depth_values = Queue()
queue_images = Queue()
def main():
    

    if __name__ == "__main__":
        main()

