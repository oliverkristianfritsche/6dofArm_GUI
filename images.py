import socket
import struct
import cv2
import numpy as np
import time
import threading
from image_utils import detect_white_spheroids, compute_depth_map
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

def handle_connection(conn, addr,image_queue, port):
    print(f"Connection from {addr,port}")
    fps_list = []
    previous_time = time.time()
    #clear buffer before starting
    while not image_queue.empty():
        image_queue.get()

    conn.settimeout(5)
    with conn:
        while True:
            frame_data = receive_frame(conn)
            if frame_data is None:
                break
            image_queue.put(frame_data)
            # Convert the byte data to a numpy array
            nparr = np.frombuffer(frame_data, np.uint8)

            # Decode the numpy array to an image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            # Update FPS list and calculate average FPS
            fps_list.append(fps)
            if len(fps_list) > 100:  # Keep the last 100 FPS values
                fps_list.pop(0)
            average_fps = sum(fps_list) / len(fps_list)

            # Process image (e.g., detect white spheroids)
            find_white = detect_white_spheroids(img, show=False)
            if find_white is not None:
                cv2.circle(img, (find_white[0][0][0], find_white[0][0][1]), 10, (255, 0, 0), 2)
            
            # Display FPS on the image
            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Average FPS: {average_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the image
            cv2.imshow(f'Received Image from {addr}', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

def server_thread(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, port))
        server_socket.listen()
        print(f"Listening on {SERVER_IP}:{port}")

        while True:
            try:
                conn, addr = server_socket.accept()
            except Exception as e:
                print(f"Error accepting connection: {e}")
                return
            
            if port == 8082:
                image_queue = queue_cam1
            else:
                image_queue = queue_cam2
            thread = threading.Thread(target=handle_connection, args=(conn, addr,image_queue, port))
            thread.start()

            # handle_connection(conn, addr,image_queue, port)

def get_depth_map(queue_cam1, queue_cam2, focal_length, baseline, show=True):
    while True:
        if not queue_cam1.empty() and not queue_cam2.empty():
            imgL_data = queue_cam1.get()
            imgR_data = queue_cam2.get()

            nparrL = np.frombuffer(imgL_data, np.uint8)
            nparrR = np.frombuffer(imgR_data, np.uint8)
            imgL = cv2.imdecode(nparrL, cv2.IMREAD_COLOR)
            imgR = cv2.imdecode(nparrR, cv2.IMREAD_COLOR)

            # Compute the colorized depth map for better visualization
            colored_depth_map = compute_depth_map(imgL, imgR, focal_length, baseline)

            # Detect white spheroids in the left image and get the largest one for simplicity
            largest_contour = detect_white_spheroids(imgL, show=False)
            
            if largest_contour is not None:
                # Extract the depth information for the detected spheroids
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x, center_y = x + w // 2, y + h // 2

                # Depth value at the center of the spheroid in the depth map
                depth_value = colored_depth_map[center_y, center_x]

                # Annotate the depth on the depth map
                annotation_text = f"Depth: {depth_value}"
                cv2.drawContours(colored_depth_map, [largest_contour], -1, (0, 255, 0), 2)
                cv2.putText(colored_depth_map, annotation_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 255), 2)

            # Display the colorized and annotated depth map
            if show:
                cv2.imshow('Depth Map', colored_depth_map)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

def main():
    threads = []
    
    for port in PORTS:
        t = threading.Thread(target=server_thread, args=(port,))
        t.start()
        threads.append(t)

    processing_thread = threading.Thread(target=get_depth_map, args=(queue_cam1, queue_cam2, 3.6, 25))
    processing_thread.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
