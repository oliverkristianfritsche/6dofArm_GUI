import socket
import struct
import cv2
import numpy as np
import time
from image_utils import detect_white_spheroids
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8082

def receive_frame(conn):
    try:
        # Receive frame size (4 bytes for an int)
        frame_size_data = conn.recv(4)
  
        if not frame_size_data:
            print("Connection closed by the client")
            return None

        frame_size = struct.unpack('!I', frame_size_data)[0] # Unpack to integer
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
    

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen()

        print(f"Listening on {SERVER_IP}:{SERVER_PORT}")
        
        try:
            conn, addr = server_socket.accept()
        except Exception as e:
            print(f"Error accepting connection: {e}")
            return

        fps_list = []
        previous_time = time.time()

        with conn:
            print(f"Connection from {addr}")
            while True:
                frame_data = receive_frame(conn)
                if frame_data is None:
                    break

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
                if len(fps_list) > 100:  # Keep the last 100 FPS values for a more relevant average
                    fps_list.pop(0)
                average_fps = sum(fps_list) / len(fps_list)

                # find_reds = detect_light_orange(img)
                find_white = detect_white_spheroids(img,show=True)
                
                #draw a circle around the largest spheroid
                if find_white is not None:
                    cv2.circle(img, (find_white[0][0][0],find_white[0][0][1]), 10, (255, 0, 0), 2)
                # find_lighter = detect_lighter_flame(img)    
                # Display FPS on the image
                cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Average FPS: {average_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the image
                cv2.imshow('Received Image', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
