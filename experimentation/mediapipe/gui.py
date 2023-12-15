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

        # set up dictionary to read from for gui display of data
        self.calculated_data = {
            "bicep_force": "NaN",
            "elbow_angle": "NaN",
            "uarm_spher_coords": "NaN",#["NaN", "NaN", "NaN"],
            "farm_spher_coords": "NaN"#["NaN", "NaN", "NaN"]
        }

        # initialize mediapipe
        self.mediapipe_runtime = lsmp.Pose_detection(pose_landmarker)
        self.mediapipe_runtime.start()

        ### GUI SETUP

        # delay between frames
        self.delay = 15

        # data update interval
        self.update_interval = 15

        # image height/width
        self.height, self.width = self.mediapipe_runtime.get_height_width()
        
        # initialize root of the tkinter gui display
        self.root = Tk()
        self.root.title("Biomechanics Simulation")

        # create frame
        self.gui = Frame(self.root)#, bg = "white")
        self.gui.grid()

        # create image label in frame
        self.image_label = Label(self.gui)
        self.image_label.grid(row = 0, column = 0)
        self.image_label.photo = None


        # set up data displays 
        self.data = Frame(self.gui)
        self.data.grid(row = 0, column = 1)

        self.bicep_var = StringVar()
        self.bicep_var.set("Bicep force: %s" % self.calculated_data["bicep_force"])
        self.bicep_force = Label(self.data, textvariable = self.bicep_var, height = 1, width = 30)
        self.bicep_force.grid(row = 0, column = 0)

        self.elbow_var = StringVar()
        self.elbow_var.set("Elbow angle: %s" % self.calculated_data["elbow_angle"])
        self.elbow_angle = Label(self.data, textvariable = self.elbow_var, height = 1, width = 30)
        self.elbow_angle.grid(row = 1, column = 0)

        # set up height and weight inputs
        #self.input_height = Text(self.root, height = 1, width = 8, bg = "gray")
        #self.update_height = Button(self.root, height = 1, width = 8, text = "Update", 
        #                            command = self.mediapipe_runtime.set_height(self.input_height.get("1.0", "end-1c")))
        #self.input_weight = Text(self.root, height = 1, width = 8, bg = "gray")
        #self.update_weight = Button(self.root, height = 1, width = 8, text = "Update", 
        #                            command = self.mediapipe_runtime.set_weight(self.input_weight.get("1.0", "end-1c")))

        #self.bicep_force = Label(self.root, text = "Bicep force: %s" % self.calculated_data["bicep_force"])
    
    # start/run the gui display
    def start(self):
        # start updater loops
        self.update_display()                               # update display
        #self.update_data()                                  # update numerical data
        #self.mediapipe_runtime.run()

        # handle program close
        self.root.protocol("WM_DELETE_WINDOW", self.__del__)

        # start the display
        self.root.mainloop()

    # update the data being displayed
    def update_display(self):#, new_frame, data_dict):
        # handle frame/image data
        ret, frame = self.mediapipe_runtime.get_cur_frame()
        if ret:                                             # only update if frame is presenty
            self.image_label.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.image_label.configure(image = self.image_label.photo)
            self.calculated_data = self.mediapipe_runtime.get_calculated_data()

        # call next update cycle
        self.root.after(self.delay, self.update_display)    # update approximately 60 times per second

    # update numerical data in gui
    def update_data(self):
        # update data
        self.bicep_var.set("Bicep force: %s" % self.calculated_data["bicep_force"])
        self.elbow_var.set("Elbow angle: %s" % self.calculated_data["elbow_angle"])

        # call next update cycle
        self.root.after(self.update_interval, self.update_data)

    # handle end of runtime
    def __del__(self):
        # stop gui
        self.root.destroy()

        # stop mediapipe
        if self.mediapipe_runtime.webcam_stream.isOpened():
            self.mediapipe_runtime.webcam_stream.release()
        self.mediapipe_runtime.set_stop(True)
        self.mediapipe_runtime.join()

        # end gui
        #self.root.destroy()

gui = SimGUI()
gui.start()