import socket
import numpy as np

def send_joint_positions_to_pi(joint_positions):
    #add 90 degrees to the joint positions
    joint_positions = [int(np.degrees(i)+90) for i in joint_positions]
    joint_positions[1] = 180 - joint_positions[1]
    # joint_positions[3] = 180 - joint_positions[3]
    joint_positions[4] = 180 - joint_positions[4]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(('192.168.3.157', 12345)) 
            message = ','.join(map(str, joint_positions[:-1]))  # Convert list of positions to comma-separated string
            print(f"Sending joint positions to Raspberry Pi: {message}")
            s.sendall(message.encode('utf-8'))
        except ConnectionError as e:
            print(f"Failed to send data to Raspberry Pi: {e}")


