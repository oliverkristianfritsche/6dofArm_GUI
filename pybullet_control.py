import pybullet as p
import pybullet_data
import numpy as np
from networking import send_joint_positions_to_pi
import cv2
import time

send_to_pi = True

class PyBulletControl:
    
    def __init__(self):
        self.cameras = []
        self.past_positions = []  # Initialize a list to store past positions
        self.max_trail_length = 100  # Limit the number of positions to store to keep the trail a manageable length
        self.initialize_simulation()
        

    def initialize_simulation(self,working_path="descriptions/armsimplified-cigarette-new/armsimplified-cigarette-new.urdf"):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = self.load_robot(working_path)
        self.marker = self.create_end_effector_marker()
        self.numJoints = p.getNumJoints(self.robotId)
        self.configure_simulation_environment()

    def reset_simulation(self):
        p.disconnect()  # Disconnect the current session
        self.initialize_simulation()  # Reinitialize the simulation

    def create_camera(self, camera_position, camera_target, camera_up, fov=60, aspect=1, near=0.02, far=2):

        # Add a camera at the specified position and target
        camera_params = [camera_position, camera_target, camera_up, fov, aspect, near, far]
        
        # Add the camera to the list of cameras
        self.cameras.append(camera_params)
        print("Camera added at position:", camera_position, "and target:", camera_target, "with up vector:", camera_up)

    def capture_camera_image(self, width=640, height=480, camera_fov=60, camera_aspect=1, camera_near=0.02, camera_far=2):
        #append all the camera images to a list
        images = []
        for camera_id in self.cameras:
            # Capture an image
            width,height,rgb_img, depth_img, seg_img = p.getCameraImage(
                width=width, height=height,
                viewMatrix=p.computeViewMatrix(camera_id[0], camera_id[1], camera_id[2]),
                projectionMatrix=p.computeProjectionMatrixFOV(fov=camera_fov, aspect=camera_aspect, nearVal=camera_near, farVal=camera_far)
            )
            # Convert the RGB image to a format that OpenCV expects
            rgb_array = np.reshape(np.array(rgb_img), (height, width, 4))
            rgb_array = rgb_array[:, :, :3]
        
            # Convert RGB to BGR, which OpenCV uses
            rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            images.append(rgb_array)

        images = np.array(images)
        #resize the images to 320, 240
        images = [cv2.resize(image, (320,240)) for image in images]
        return images

    def draw_grid_mm(self, grid_size=1, step=0.1, line_color=[0, 0, 0, 0.5]):  # step=0.1 for 100mm spacing
        """
        Draw a grid on the floor with millimeter precision.

        Parameters:
        - grid_size: The length of each side of the grid in meters.
        - step: The distance between adjacent lines in the grid in meters (0.1 for 100mm).
        - line_color: The color of the grid lines as [R, G, B, Alpha], values between 0 and 1.
        """
        # Adjustments for finer grid lines
        for y in np.arange(-grid_size, grid_size + step, step):
            p.addUserDebugLine([-grid_size, y, 0.001], [grid_size, y, 0.001], lineColorRGB=line_color[:3], lineWidth=1, lifeTime=0)
        for x in np.arange(-grid_size, grid_size + step, step):
            p.addUserDebugLine([x, -grid_size, 0.001], [x, grid_size, 0.001], lineColorRGB=line_color[:3], lineWidth=1, lifeTime=0)

  
    def create_floor(self, extent=1000, height=-0.001,color=[0,0,0,1]):  # A large extent ensures the floor covers the whole area
        """
        Create a white floor by adding a large, thin box.

        Parameters:
        - extent: The half-extent of the floor box, essentially controlling its size.
        - height: The height at which the floor is placed, slightly below 0 to avoid z-fighting with the grid.
        """
        floor_col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[extent, extent, 0.001])
        floor_vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[extent, extent, 0.001], rgbaColor=color)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_col_shape, baseVisualShapeIndex=floor_vis_shape, basePosition=[0, 0, height])


    def configure_simulation_environment(self):
        # Set camera parameters (as before)
        self.set_initial_camera_view()
        
        # Hide synthetic camera RGB data and other unnecessary elements
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # Disable RGB data preview
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # Disable depth buffer preview
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # Disable segmentation mark preview
        
        # Optionally, hide other GUI elements as needed
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Hide the GUI
        # self.create_floor()  # Create a white floor
        self.draw_grid_mm()  # Draw a grid on the floor

    def capture_camera_image_test(self):
        # Camera configuration parameters (same as before)
        camera_eye_position = [1, .5, 0]
        camera_target_position = [0, 0, 0]
        camera_up_vector = [0, 0, 1]
        camera_fov = 60
        camera_aspect = 1
        camera_near = 0.02
        camera_far = 2

        # Capture an image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=640, height=480,
            viewMatrix=p.computeViewMatrix(camera_eye_position, camera_target_position, camera_up_vector),
            projectionMatrix=p.computeProjectionMatrixFOV(fov=camera_fov, aspect=camera_aspect, nearVal=camera_near, farVal=camera_far)
        )

        # Convert the RGB image to a format that OpenCV expects
        rgb_array = np.reshape(np.array(rgb_img), (height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # Drop the alpha channel
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR, which OpenCV uses

        # Display the image using OpenCV
        cv2.imshow('Camera Image', rgb_array)
        cv2.waitKey(1)  # Refresh the display window. Use cv2.waitKey(0) to pause until a key is pressed if needed.

        rgb_img = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for use with other libraries
        depth_img = np.reshape(np.array(depth_img), (height, width))
        seg_img = np.reshape(np.array(seg_img), (height, width))

        return rgb_img, depth_img, seg_img


    def set_initial_camera_view(self):
        # Set camera parameters
        camera_distance = 0.5  # Adjust this value to zoom in; smaller numbers are closer
        camera_yaw = 45  # Adjust as needed
        camera_pitch = -30  # Adjust as needed
        camera_target_position = [0, 0, 0.25]  # Adjust as needed to focus on a specific part of your robot or environment
        
        # Apply camera settings
        p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                     cameraYaw=camera_yaw,
                                     cameraPitch=camera_pitch,
                                     cameraTargetPosition=camera_target_position)
        
    def load_robot(self, working_path):
        cubeStartPos = [0, 0, 0]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        return p.loadURDF(working_path, cubeStartPos, cubeStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=1)

    def create_end_effector_marker(self):
        marker_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseVisualShapeIndex=marker_visual_shape, basePosition=[0, 0, 0])

    def set_position_with_ik_and_constraints(self, x, y, z, steps=10):
        target_position = [x, y, z]
        jointPoses = p.calculateInverseKinematics(self.robotId, self.numJoints-1, target_position, maxNumIterations=30000, residualThreshold=1e-5)
        
        # Apply joint constraints
        constrained_joint_poses = self.apply_joint_constraints(jointPoses)
        
        # Get current joint positions
        current_joint_positions = [p.getJointState(self.robotId, i)[0] for i in range(self.numJoints)]
        
        # Interpolate between current and target positions in 'steps' steps
        for step in range(1, steps + 1):
            interpolated_positions = [(1 - step / steps) * current + (step / steps) * target for current, target in zip(current_joint_positions, constrained_joint_poses)]
            
            # Temporarily apply interpolated positions to check for collision
            for i, pose in enumerate(interpolated_positions):
                p.resetJointState(self.robotId, i, pose)
            
            if self.check_collision(self.robotId):
                print("Collision detected during smoothing at step {}. Adjusting position or planning path is required.".format(step))
                return  # Skip setting the joints to avoid collision
            
            # Apply the interpolated positions without collision
            for i, pose in enumerate(interpolated_positions):
                p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=pose)
            
            # Optionally, wait a bit between steps for a smoother transition
            # time.sleep(some_small_value)
        
        if(send_to_pi):
            send_joint_positions_to_pi(constrained_joint_poses)  # Send final joint positions to Raspberry Pi
        # Update the end effector marker
        self.update_end_effector_marker(target_position)

    def check_collision(self, test_robot_id, excluded_links=[]):
        collisions = p.getContactPoints(bodyA=test_robot_id)
        for collision in collisions:
            if collision[3] in excluded_links:  # Ignore specified links
                continue
            return True  # Collision detected
        return False  # No collision detected

    def apply_joint_constraints(self, joint_positions):
        constrained_positions = []
        
        for i,pos in enumerate(joint_positions):
            pos_deg = np.degrees(pos)
            if(i!=1):
                clamped_deg = np.clip(pos_deg, -90, 90)  # Joint constraints [0, 180] degrees
            else:
                clamped_deg = np.clip(pos_deg, -67, 67)
            clamped_rad = np.radians(clamped_deg)
            constrained_positions.append(clamped_rad)

        return constrained_positions

    def update_end_effector_marker(self, position):
        p.resetBasePositionAndOrientation(self.marker, position, [0, 0, 0, 1])

    def dance(self, duration=10):
        start_time = time.time()
        while time.time() - start_time < duration:
            # Example dance movements: Moving in a square pattern
            self.set_position_with_ik_and_constraints(0.2, 0.2, 0.2)
            time.sleep(0.5)
            self.set_position_with_ik_and_constraints(-0.2, 0.2, 0.2)
            time.sleep(0.5)
            self.set_position_with_ik_and_constraints(-0.2, -0.2, 0.2)
            time.sleep(0.5)
            self.set_position_with_ik_and_constraints(0.2, -0.2, 0.2)
            time.sleep(0.5)

    def smooth_dance(self, duration=10, step_duration=1):
        # Define keyframes for the dance, each as (x, y, z) tuples
        keyframes = [
            (0.2, 0, 0.2),  # Front
            (0, 0.2, 0.2),  # Right
            (-0.2, 0, 0.2), # Back
            (0, -0.2, 0.2), # Left
            (0.1, 0.1, 0.3), # Diagonal up front-right
            (-0.1, 0.1, 0.3), # Diagonal up back-right
            (-0.1, -0.1, 0.3), # Diagonal up back-left
            (0.1, -0.1, 0.3), # Diagonal up front-left
        ]

        start_time = time.time()
        keyframe_index = 0

        while time.time() - start_time < duration:
            # Calculate the current phase of the step
            current_time = time.time() - start_time
            phase = (current_time % step_duration) / step_duration

            # Determine the current and next keyframe
            current_keyframe = keyframes[keyframe_index]
            next_keyframe = keyframes[(keyframe_index + 1) % len(keyframes)]

            # Interpolate between the current and next keyframe
            interpolated_position = [
                current_keyframe[i] + (next_keyframe[i] - current_keyframe[i]) * phase for i in range(3)
            ]

            # Apply the interpolated position
            self.set_position_with_ik_and_constraints(*interpolated_position)

            # Step the simulation to update the movements
            # self.step_simulation()

            # Check if we should move to the next keyframe
            if phase >= 1.0:
                keyframe_index = (keyframe_index + 1) % len(keyframes)

            time.sleep(0.02)  # Small sleep to prevent spamming the simulation with too many updates

    def bobbing_dance(self, steps=20, amplitude=0.1, frequency=1):
        step_duration = 0.1  # Duration of each step in seconds
        for step in range(steps):
            # Calculate the progress through the dance
            progress = step / float(steps)
            time_in_cycle = progress * 2 * np.pi * frequency
            
            # Harmonic motion for horizontal movement
            x = amplitude * np.sin(time_in_cycle)
            y = amplitude * np.cos(time_in_cycle)
            
            # Bobbing motion for vertical movement, peaking in the middle of the sequence
            z = 0.2 + 0.05 * np.sin(time_in_cycle * 2)  # Double frequency for up-and-down motion
            
            # Update the position using inverse kinematics
            self.set_position_with_ik_and_constraints(x, y, z)
            
            # Step the simulation to reflect the update
            self.step_simulation()
            
            # Wait before the next step
            time.sleep(step_duration)
    
        # Ensure the robot returns to its starting position
        self.set_position_with_ik_and_constraints(0, 0, 0.2)
        self.step_simulation()

    def draw_sine_wave_around_z(self, revolutions=2, steps_per_revolution=200, amplitude=0.05, radius=0.12, sine_frequency=1, sine_wavelength=1):
        total_steps = revolutions * steps_per_revolution
        step_duration = 0.05  # Time between steps, adjust for speed of drawing
        
        # Calculate the number of sine wave cycles over the entire path based on sine_frequency
        sine_cycles = revolutions * sine_frequency
        
        for step in range(total_steps):
            # Calculate the angle for the current step
            angle = (step / float(steps_per_revolution)) * 2 * np.pi
            
            # Calculate X and Y positions for circular path around Z-axis
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Calculate Z position for sine wave with adjustable frequency and wavelength
            # Adjust the sine wave's frequency to complete 'sine_cycles' cycles over the total path
            z_wave_angle = sine_cycles * angle * sine_wavelength
            z = amplitude * np.sin(z_wave_angle) + 0.2  # 0.2 to offset from the ground or base level
            
            # Update the position using inverse kinematics
            self.set_position_with_ik_and_constraints(x, y, z)
            
            # Step the simulation to reflect the update
            self.step_simulation()
            
            # Wait before the next step
            time.sleep(step_duration)


    def step_simulation(self):
        # self.capture_camera_image_test()
        p.stepSimulation()

