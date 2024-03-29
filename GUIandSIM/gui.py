import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import ttk, Label, PhotoImage
from pybullet_control import PyBulletControl
import numpy as np
import threading
import time
from images import StereoCameraServer, signal_handler
from queue import Queue
import signal

virtual_camera = False #broken in this version
class PyBulletControlGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Robot Joint Control")

        # Initialize the PyBullet control once to avoid redundancy
        self.pybullet_control = PyBulletControl()

        # Initialize GUI elements
        self.init_gui_elements()

        # Start simulation update loop
        self.update_simulation()

        # Initialize the Stereo Camera Server
        self.stereo_server = StereoCameraServer("0.0.0.0", [8082, 8083], 528, 25)

        # Start server threads for each port - assuming `start_server_thread` is correctly implemented in `StereoCameraServer`
        self.start_server_threads()

        # Attempt to apply the theme; handle failure gracefully
        try:
            self.master.tk.call('source', 'azure.tcl')
            self.master.tk.call("set_theme", "light")
        except tk.TclError:
            print("Failed to apply theme, defaulting to standard Tkinter appearance.")

        # # Start a camera update thread - no need to start it twice as in the original code
        if(virtual_camera):
            self.camera_update_thread = threading.Thread(target=self.capture_images_thread, daemon=True)
            self.camera_update_thread.start()

    def start_server_threads(self):
        for port in self.stereo_server.ports:
            t = threading.Thread(target=self.stereo_server.server_thread, args=(port,))
            t.daemon = True
            t.start()

    def update_simulation(self):
        self.pybullet_control.step_simulation()

        # Check if stereo_server is defined before accessing it
        if hasattr(self, 'stereo_server') and not self.stereo_server.queue_images.empty():
            print(self.stereo_server.queue_images.get_nowait()[1:])
        
        # Schedule the next update
        self.master.after(50, self.update_simulation)

    def init_gui_elements(self):
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))

        self.coordinate_frame = ttk.Frame(self.master, padding="10 10 10 10")
        self.coordinate_frame.pack(fill='x', expand=True)
        
        self.desired_position_label = ttk.Label(self.master, text="Desired Position: X=0.00 m\nY=0.00 mm\nZ=500.00 mm", font=("Arial", 12))
        self.desired_position_label.pack(pady=10)

        self.coordinates = {'x': 0, 'y': 0, 'z': 0.5}

        # Create sliders and text inputs for X, Y, Z coordinates
        self.coordinate_entries = {}
        for coord in 'xy':
            self.create_coordinate_control(coord, slider_range=(-.3, .3))
        self.create_coordinate_control('z', slider_range=(0, .5))

        # Button to set position from text inputs
        self.set_position_button = ttk.Button(self.master, text="Set Position", command=self.set_position_from_entries)
        self.set_position_button.pack(pady=10)

        # # Button to clear old dots
        # self.clear_dots_button = ttk.Button(self.master, text="Clear Old Dots", command=self.clear_old_dots)
        # self.clear_dots_button.pack(pady=10)

        self.set_dance_button = ttk.Button(self.master, text="Dance", command=self.pybullet_control.draw_sine_wave_around_z)
        self.set_dance_button.pack(pady=10)
        self.pybullet_control.create_camera([-0.1625, .15, 0.036], [-0.1625, 0, 0.036], [0, 0, 1])
        self.pybullet_control.create_camera([-.1375, .15, 0.036], [-.1375, 0, 0.036], [0, 0, 1])
        # Image display areas for the camera images
        self.camera_image_labels = []
        for i in range(len(self.pybullet_control.cameras)):
            label = ttk.Label(self.master)
            label.pack(side="bottom", padx=15)
            self.camera_image_labels.append(label)
        

    def capture_images_thread(self):
        while True:
            if(not self.stereo_server.queue_images.empty()):
                real_camera_tuple = self.stereo_server.queue_images.get_nowait()
                camera_images = np.array(self.pybullet_control.capture_camera_image(), real_camera_tuple[0])
                self.master.after(0, lambda: self.update_camera_images(camera_images))
                print(real_camera_tuple[1:])
            

    def capture_and_display_images(self):
        camera_images = np.array(self.pybullet_control.capture_camera_image())
        self.update_camera_images(camera_images)
        self.schedule_capture_and_display_images()

    def update_camera_images(self, camera_images):
        camera_images = np.array(camera_images)
        for i, img in enumerate(camera_images):
            # Convert to PIL format
            pil_img = Image.fromarray(camera_images[i])
            # Convert to Tkinter format
            tk_img = ImageTk.PhotoImage(image=pil_img)

            # Update the label with the new image
            self.camera_image_labels[i].configure(image=tk_img)
            self.camera_image_labels[i].image = tk_img  # Keep a reference!

    def create_coordinate_control(self, coord, slider_range = (-.1, .1)):
        frame = ttk.Frame(self.coordinate_frame, padding="5 5 5 5")
        frame.pack(fill='x', expand=True)
        label = ttk.Label(frame, text=f"{coord.upper()} (meters):", font=("Arial", 10))
        label.pack(side='left')

        # Slider
       
        slider = tk.Scale(frame, from_=slider_range[0], to=slider_range[1], orient='horizontal', resolution=0.001,
                          command=lambda value, c=coord: self.update_robot_position(c, float(value)), length=150)
        slider.set(self.coordinates[coord])
        slider.pack(side='left', padx=5)

        # Entry
        entry = ttk.Entry(frame, width=10)
        entry.pack(side='left', padx=5)
        self.coordinate_entries[coord] = entry

    def set_position_from_entries(self):
        try:
            x = float(self.coordinate_entries['x'].get())
            y = float(self.coordinate_entries['y'].get())
            z = float(self.coordinate_entries['z'].get())
            self.update_robot_position('x', x)
            self.update_robot_position('y', y)
            self.update_robot_position('z', z)
        except ValueError:
            # Handle the case where one or more entries are not valid numbers
            print("Please enter valid numbers for X, Y, and Z coordinates.")

    def update_robot_position(self, coord, value):
        self.coordinates[coord] = value
        self.pybullet_control.set_position_with_ik_and_constraints(self.coordinates['x'], self.coordinates['y'], self.coordinates['z'])
        self.update_desired_position_display(self.coordinates['x'], self.coordinates['y'], self.coordinates['z'])

        # Update slider to reflect the text input
        for c, slider_frame in self.coordinate_frame.children.items():
            if coord in c:
                slider = slider_frame.winfo_children()[1]  # Assuming the slider is the second widget in the frame
                slider.set(value)

    
    def update_desired_position_display(self, x, y, z):
        self.desired_position_label.config(text=f"Desired Position:\nX={x*1000:.2f} mm\nY={y*1000:.2f} mm\nZ={z*1000:.2f} mm")

    
    

def main():
    root = tk.Tk()
    app = PyBulletControlGUI(root)
    root.mainloop()
    
   

if __name__ == "__main__":
    main()
