# BD 2023
# This class is intended to serve as a GUI display manager for the body forces simulation research project.


import numpy as np
import math
import threading

import cv2

from tkinter import *
from PIL import Image
from PIL import ImageTk

import livestream_mediapipe_class as lsmp   # custom class, handles mediapipe


### OPTIONS

# model to use for mediapipe
pose_landmarker = './landmarkers/pose_landmarker_full.task'

# load and prep placeholder image for program initialization
no_image_path = './no_image.png'            # placeholder image location
no_image = Image.fromarray(cv2.cvtColor(cv2.imread(no_image_path), cv2.COLOR_BGR2RGB))


# setup runnable class for management of the GUI
class SimGUI():

    # initialization
    def __init__(self) -> None:
        
        ### DATA AND CONSTANTS
        
        #self.cur_frame = cv2.imread(no_image_path)                  # current frame to display in window
        
        # set up dictionary to read from for gui display of data
        self.calculated_data = {
            "bicep_force": math.nan,
            "elbow_angle": math.nan,
            "uarm_spher_coords": [math.nan, math.nan, math.nan],
            "farm_spher_coords": [math.nan, math.nan, math.nan]
        }

        # initialize mediapipe
        self.mediapipe_runtime = lsmp.Pose_detection(pose_landmarker)
        self.mediapipe_runtime_thread = threading.Thread(target = self.mediapipe_runtime.run, args = ())

        ### GUI SETUP
        
        # initialize root of the tkinter gui display
        self.root = Tk()

        # configure UI
        self.image_panel = Label(self.root, image = ImageTk.PhotoImage(no_image))                                     # initialize image panel
        self.image_panel.pack(side = "left", padx = 10, pady = 10)
        #self.bicep_force = Label(self.root, text = "Bicep force: %s" % self.calculated_data["bicep_force"])
    
    # start/run the gui display
    def start(self):
        # start updater loop
        self.update_display()
    
        # start the display
        self.root.mainloop()

    # update the data being displayed
    def update_display(self):#, new_frame, data_dict):
        # handle frame/image data
        self.image_panel.setvar(image = ImageTk.PhotoImage(Image.fromarray(self.mediapipe_runtime.get_cur_frame())))

        # handle numerical data

        # call next update cycle
        self.root.after(17, self.update_display())      # update approximately 60 times per second

gui = SimGUI()
gui.start()