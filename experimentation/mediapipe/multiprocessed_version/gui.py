# BD 2023-24
# This class is intended to serve as a GUI display manager for the body forces simulation research project.

## TODO:
#   - fix sync issues with root.mainloop blocking proper updating of the manual calibration toggle
#       when calling function from outside the main program loop
#       - basically, by changing the value of self.manual_calibration from a function call that
#           didn't originate from the main loop (i.e. from within root.mainloop()), the update
#           attempt is blocked by root.mainloop()


import numpy as np
import math
from matplotlib import pyplot as plt

import threading
import multiprocessing

import cv2

from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL import ImageTk

import livestream# as lsmp   # custom class, handles mediapipe


### OPTIONS

# model to use for mediapipe
#pose_landmarker = './landmarkers/pose_landmarker_full.task'

# load and prep placeholder image for program initialization
#no_image_path = './no_image.png'            # placeholder image location
#no_image = Image.fromarray(cv2.cvtColor(cv2.imread(no_image_path), cv2.COLOR_BGR2RGB))


# setup runnable class for management of the GUI
class Sim_GUI(multiprocessing.Process):

    # initialization
    def __init__(self, stop,
                extrap_to_gui = multiprocessing.Pipe(), gui_to_extrap = multiprocessing.Pipe(),
                stream_to_gui = multiprocessing.Pipe(), gui_to_stream = multiprocessing.Pipe()):# -> None:
        
        # base constructor
        multiprocessing.Process.__init__(self)

        # handle args
        # pipes
        self.extrap_to_gui = extrap_to_gui
        self.gui_to_extrap = gui_to_extrap
        self.stream_to_gui = stream_to_gui
        self.gui_to_stream = gui_to_stream
        
        # process stop condition
        self.stop = stop

        # initialize send_extrap
        self.send_extrap = [0, 0, 0, 0]     # used to send user input to extrapolation process

        # set up thread for separate handling of calculated data handling
        self.extrap_handler = threading.Thread(target = self.handle_extrap_pipes)
        self.stream_handler = threading.Thread(target = self.handle_stream_pipes)


        ### DATA AND CONSTANTS
        
        # temp store frame data
        self.ret = None
        self.frame = None

        # variable for dynamic width of settings
        self.settings_width = 20

        # set up dictionary to read from for gui display of data
        self.calculated_data = {
            "right_bicep_force": "NaN",
            "right_elbow_angle": "NaN",
            "left_bicep_force": "NaN",
            "left_elbow_angle": "NaN",
            "uarm_spher_coords": "NaN",#["NaN", "NaN", "NaN"],
            "farm_spher_coords": "NaN"#["NaN", "NaN", "NaN"]
        }

        # store past bicep force calculations
        self.history_bicep_force = np.ndarray((1), dtype = "float32")
        self.history_elbow_angle = np.ndarray((1), dtype = "float32")
        self.hbf_max_len = 1000             # max length for history of bicep force

        # initialize mediapipe thread
        #self.mediapipe_runtime = lsmp.Pose_detection(pose_landmarker)
        #self.mediapipe_runtime.start()

        # allow entry in imperial (instead of metric)
        self.use_imperial = False

        # allow auto update of graph (WARNING: lags current setup)
        self.auto_update_graph = False

        # toggle manual conversion/calibration ratio/factor
        self.manual_calibration = False


        ### GUI SETUP

        # delay between frames
        self.delay = 15

        # data update interval
        self.update_interval = 200      # update every 200 milliseconds

        # image height/width
        self.height, self.width = livestream.HEIGHT, livestream.WIDTH   # initialize to initial height/width from livestream class
        
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


        ### GUI ORGANIZATION 

        # grid section for containing all textual info
        self.data = Frame(self.gui)
        self.data.grid(row = 0, column = 1)

        # grid section for settings/calibration
        self.settings = LabelFrame(self.data, text = "Settings:")
        self.settings.grid(row = 0, column = 0)

        # settings for unit conversion factor (metric <--> sim units)
        #self.ucf_label = Label(self.settings, text = "Unit conversion factor: ", height = 1, width = self.settings_width)
                               #cursor = "Approximate conversion ratio between metric units and simulation units.\n" + 
                               #         "Only intended for use with \"Manual\" functionality.")
        #self.ucf_var = StringVar()
        #self.ucf_entry = Entry(self.settings, textvariable = self.ucf_var)
        #self.ucf_toggle_var = IntVar()
        #self.ucf_toggle = Checkbutton(self.settings, text = "Manual", variable = self.ucf_toggle_var,       # now defunct, does effectively nothing with new calibration system
        #                              onvalue = 1, offvalue = 0, height = 1, width = 10, command = self.toggle_manual_conversion)
        #self.ucf_submit = Button(self.settings, text = "Submit", command = self.set_conversion_ratio)
        #self.ucf_label.grid(row = 1, column = 0)
        #self.ucf_entry.grid(row = 1, column = 1)
        #self.ucf_toggle.grid(row = 2, column = 0)
        #self.ucf_submit.grid(row = 2, column = 1)

        # biacromic scale factor
        self.bsf_scale = Scale(self.settings, from_ = 0.22, to = 0.24, orient = "horizontal", length = 200, 
                                label = "Biacromic (shoulder width) Scale", showvalue = True, resolution = 0.001, command = self.set_bsf)
        self.bsf_scale.grid(row = 3, columnspan = 2)

        # allow entry in imperial unit system (as opposed to metric)
        self.ms_label = Label(self.settings, text = "Use imperial system? (Default: metric)")
        self.ms_var = IntVar()
        self.ms_toggle = Checkbutton(self.settings, variable = self.ms_var, onvalue = 1, offvalue = 0, command = self.toggle_imperial)
        self.ms_label.grid(row = 4, column = 0)
        self.ms_toggle.grid(row = 4, column = 1)

        # allow adjustment of image height and width
        #self.image_adjust = LabelFrame(self.settings, text = "Image adjustment:")   # set up lavel frame to section off this part of the settings
        #self.image_adjust.grid(row = 5, column = 0)
        # height
        #self.image_height_label = Label(self.image_adjust, text = "Image height: ", height = 1, width = self.settings_width)
        #self.image_height_var = StringVar()
        #self.image_height_var.set("640")
        #self.image_height_entry = Entry(self.image_adjust, textvariable = self.image_height_var)
        #self.image_height_label.grid(row = 1, column = 0)
        #self.image_height_entry.grid(row = 1, column = 1) 
        # width
        #self.image_width_label = Label(self.image_adjust, text = "Image width: ", height = 1, width = self.settings_width)
        #self.image_width_var = StringVar()
        #self.image_width_var.set("480")
        #self.image_width_entry = Entry(self.image_adjust, textvariable = self.image_width_var)
        #self.image_width_label.grid(row = 2, column = 0)
        #self.image_width_entry.grid(row = 2, column = 1) 
        #submit button
        #self.submit_image_hw = Button(self.image_adjust, text = "Submit", command = self.set_livestream_hw)
        #self.submit_image_hw.grid(row = 3, column = 1)


        # grid section for user input
        self.user_input = LabelFrame(self.data, text = "User input:")
        self.user_input.grid(row = 1, column = 0)

        # height user input
        self.height_label = Label(self.user_input, text = "User height (meters): ", height = 1, width = self.settings_width)
        self.height_var = StringVar()
        self.height_var.set("1.78")                    # set to default value
        self.height_entry = Entry(self.user_input, textvariable = self.height_var)
        self.height_label.grid(row = 1, column = 0)
        self.height_entry.grid(row = 1, column = 1)

        # weight user input
        self.weight_label = Label(self.user_input, text = "User weight (kilograms): ", height = 1, width = self.settings_width)
        self.weight_var = StringVar()
        self.weight_var.set("90")                    # set to default value
        self.weight_entry = Entry(self.user_input, textvariable = self.weight_var)
        self.weight_label.grid(row = 2, column = 0)
        self.weight_entry.grid(row = 2, column = 1)

        # ball mass user input
        self.bm_label = Label(self.user_input, text = "Ball mass (kilograms): ", height = 1, width = self.settings_width)
        self.bm_var = StringVar()
        self.bm_var.set("3")                    # set to default value
        self.bm_entry = Entry(self.user_input, textvariable = self.bm_var)
        self.bm_label.grid(row = 3, column = 0)
        self.bm_entry.grid(row = 3, column = 1)

        # button to submit height and weight and ball mass
        self.submit_hw = Button(self.user_input, text = "Submit", command = self.hw_submit)
        self.submit_hw.grid(row = 4, column = 1)


        # grid section for data output
        self.data_output = LabelFrame(self.data, text = "Data output:")
        self.data_output.grid(row = 2, column = 0)

        # RIGHT ARM:
        self.do_right = LabelFrame(self.data_output, text = "Right arm:")
        self.do_right.grid(row = 1, column = 0)
        # bicep force output
        self.right_bicep_label = Label(self.do_right, text = "Bicep force (Newtons): ", height = 1, width = self.settings_width)
        self.right_bicep_var = StringVar()
        self.right_bicep_var.set(str(self.calculated_data["right_bicep_force"]))
        self.right_bicep_force = Label(self.do_right, textvariable = self.right_bicep_var, height = 1, width = int(self.settings_width / 2), relief = GROOVE)
        self.right_bicep_label.grid(row = 1, column = 0)
        self.right_bicep_force.grid(row = 1, column = 1)

        # elbow angle output
        self.right_elbow_label = Label(self.do_right, text = "Elbow angle (Degrees): ", height = 1, width = self.settings_width)
        self.right_elbow_var = StringVar()
        self.right_elbow_var.set(str(self.calculated_data["right_elbow_angle"]))
        self.right_elbow_angle = Label(self.do_right, textvariable = self.right_elbow_var, height = 1, width = int(self.settings_width / 2), relief = GROOVE)
        self.right_elbow_label.grid(row = 2, column = 0)
        self.right_elbow_angle.grid(row = 2, column = 1)

        # LEFT ARM:
        self.do_left = LabelFrame(self.data_output, text = "Left arm:")
        self.do_left.grid(row = 2, column = 0)
        # bicep force output
        self.left_bicep_label = Label(self.do_left, text = "Bicep force (Newtons): ", height = 1, width = self.settings_width)
        self.left_bicep_var = StringVar()
        self.left_bicep_var.set(str(self.calculated_data["left_bicep_force"]))
        self.left_bicep_force = Label(self.do_left, textvariable = self.left_bicep_var, height = 1, width = int(self.settings_width / 2), relief = GROOVE)
        self.left_bicep_label.grid(row = 1, column = 0)
        self.left_bicep_force.grid(row = 1, column = 1)

        # elbow angle output
        self.left_elbow_label = Label(self.do_left, text = "Elbow angle (Degrees): ", height = 1, width = self.settings_width)
        self.left_elbow_var = StringVar()
        self.left_elbow_var.set(str(self.calculated_data["left_elbow_angle"]))
        self.left_elbow_angle = Label(self.do_left, textvariable = self.left_elbow_var, height = 1, width = int(self.settings_width / 2), relief = GROOVE)
        self.left_elbow_label.grid(row = 2, column = 0)
        self.left_elbow_angle.grid(row = 2, column = 1)


        # set up bicep force scatter plot
        self.fig, self.ax = plt.subplots()
        #self.hist_bf_plot = self.ax.scatter(self.history_elbow_angle, self.history_bicep_force)
        self.ax.set_ylim(ymin = 0, ymax = 1000)
        self.ax.set_xlim(xmin = 0, xmax = 180)
        self.ax.set_xlabel("Elbow angle")
        self.ax.set_ylabel("Bicep force")
        self.ax.set_title("Bicep force vs Elbow angle (left arm)")

        # setup plot in tkinter gui (alongside button to update it)
        self.forces_graph = FigureCanvasTkAgg(self.fig, master = self.gui)
        self.forces_graph_widget = self.forces_graph.get_tk_widget()
        self.forces_graph_widget.grid(row = 1, column = 0)
        self.forces_graph_grid = LabelFrame(self.gui, text = "Graph settings:")
        self.forces_graph_grid.grid(row = 1, column = 1)
        self.forces_graph_update = Button(self.forces_graph_grid, text = "Update graph", command = self.update_scatterplot)
        self.forces_graph_au_var = IntVar()
        self.forces_graph_autoupdate = Checkbutton(self.forces_graph_grid, text = "Auto update graph", variable = self.forces_graph_au_var, onvalue = 1, offvalue = 0, command = self.toggle_graph_autoupdate)
        self.forces_graph_update.grid(row = 0, column = 0)
        self.forces_graph_autoupdate.grid(row = 1, column = 0)



    
    # start/run the gui display
    def start(self):
        
        # initialize pipes
        self.gui_to_stream.send(None)
        self.gui_to_extrap.send(None)

        # start threads for handling data
        self.extrap_handler.start()
        self.stream_handler.start()

        # start updater loops
        self.update_display()                               # update display
        self.update_data()                                  # update numerical data
        #self.mediapipe_runtime.run()
        #self.root.update_idletasks()
        #self.gui.update_idletasks()

        # handle program close
        self.root.protocol("WM_DELETE_WINDOW", self.__del__)

        # start the gui
        self.root.mainloop()


    # update the data being displayed
    def update_display(self):#, new_frame, data_dict):
        # handle frame/image data
        #ret, frame = self.mediapipe_runtime.get_cur_frame()

        #stream_data = self.stream_to_gui.recv()
        #print(stream_data)

        if self.ret:                                             # only update if frame is present
            self.frame = cv2.cvtColor(cv2.flip(self.frame,1), cv2.COLOR_BGR2RGB)      # converting back to RGB for display
            self.image_label.photo = ImageTk.PhotoImage(image = Image.fromarray(self.frame))
            self.image_label.configure(image = self.image_label.photo)
            #self.calculated_data = self.mediapipe_runtime.ep.get_calculated_data()

        # call next update cycle
        self.root.after(self.delay, self.update_display)    # update approximately 60 times per second

    # update numerical data in gui
    def update_data(self):

        if not self.stop.is_set():  # check if program set to stop
            # update data
            self.right_bicep_var.set(str(self.calculated_data["right_bicep_force"]))
            self.right_elbow_var.set(str(self.calculated_data["right_elbow_angle"]))
            self.left_bicep_var.set(str(self.calculated_data["left_bicep_force"]))
            self.left_elbow_var.set(str(self.calculated_data["left_elbow_angle"]))

            # update manual calibration
            #self.manual_calibration = self.mediapipe_runtime.toggle_auto_calibrate
            # check if using manual calibration
            #if not self.manual_calibration:
            #    self.ucf_var.set(str("%0.5f" % self.mediapipe_runtime.ep.get_conversion_ratio()))

            # update elbow angle and bicep force data
            self.update_bicep_array()
            # optional live plot updater
            if self.auto_update_graph:
                self.update_scatterplot()

            # call next update cycle
            self.gui.after(self.update_interval, self.update_data)
        else:   # handle if program set to stop
            self.__del__()



    # handle keeping track of the past n timesteps of (left arm) body force calculations
    def update_bicep_array(self):
        # if above certain n value, remove the oldest data before adding the newest
        if (np.shape(self.history_bicep_force)[0] >= self.hbf_max_len):
            np.delete(self.history_bicep_force, 0)
            np.delete(self.history_elbow_angle, 0)  # assume same is true for elbow data
        
        # append newest data to array for use by matplotlib to display graph
        self.history_bicep_force = np.append(self.history_bicep_force, float(self.calculated_data["left_bicep_force"]))
        self.history_elbow_angle = np.append(self.history_elbow_angle, float(self.calculated_data["left_elbow_angle"]))

    # update scatterplot of bicep forces vs elbow angle
    def update_scatterplot(self):
        # update plot
        self.ax.scatter(self.history_elbow_angle, self.history_bicep_force)
        self.fig.canvas.draw()

    
    # handle data pipes to/from extrapolation process
    def handle_extrap_pipes(self):
        while not self.stop.is_set():
            # receive data from extrap from pipe
            self.calculated_data = self.extrap_to_gui.recv()

            # send user input to extrapolation process if needed, otherwise let extrap know gui ready for next data
            self.gui_to_extrap.send(self.send_extrap)
            # reset user input
            self.send_extrap = [0, 0, 0, 0]
    
    # handle data pipes to/from livestream
    def handle_stream_pipes(self):
        while not self.stop.is_set():
            # get frame data from livestream
            self.ret, self.frame = self.stream_to_gui.recv()
            # let livestream know gui is ready for new frame
            self.gui_to_stream.send(None)



    ### USER INPUT HANDLERS

    # set bsf 
    def set_bsf(self, bsf_data):
        bsf = float(bsf_data)
    
        #self.mediapipe_runtime.ep.set_biacromial(bsf)

        # send to extrapolation
        self.send_extrap[3] = bsf

    # height and weight submission
    def hw_submit(self):
        height = float(self.height_entry.get())
        weight = float(self.weight_entry.get())
        ball = float(self.bm_entry.get())

        # convert from imperial to metric (if used)
        if self.use_imperial:
            height = height / 3.28084   # feet to meters
            weight = weight / 2.2046    # pounds to kilograms
            ball = ball / 2.2046        # pounds to kilograms

        #self.mediapipe_runtime.ep.set_hwb(height, weight, ball)     
        
        # send height weight ball to extrapolation.py instance
        self.send_extrap = [height, weight, ball, self.send_extrap[3]]

    # handle toggleable measurement system
    def toggle_imperial(self):
        toggle = bool(self.ms_var.get())

        self.use_imperial = toggle

        # change GUI to accomodate
        if toggle:
            self.height_label.config(text = "User height (feet): ")
            self.weight_label.config(text = "User weight (pounds): ")
            self.bm_label.config(text = "Ball mass (pounds): ")
        # if untoggled, return to default
        else:
            self.height_label.config(text = "User height (meters): ")
            self.weight_label.config(text = "User weight (kilograms): ")
            self.bm_label.config(text = "Ball mass (kilograms): ")
    
    # handle toggle of auto graph update
    def toggle_graph_autoupdate(self):
        toggle = bool(self.forces_graph_au_var)

        self.auto_update_graph = toggle

    # handle togglable manual conversion factor/ratio
    #def toggle_manual_conversion(self):
    #    toggle = bool(self.ucf_toggle_var.get())
    #
    #    self.mediapipe_runtime.toggle_auto_conversion(toggle)

    # set manual conversion factor/ratio
    #def set_conversion_ratio(self):
    #    ratio = float(self.ucf_entry.get())
    #
    #    self.mediapipe_runtime.ep.set_conversion_ratio(ratio)

    # set height and width of display/camera picture view
    #def set_livestream_hw(self):
    #    # set local variables
    #    self.height = int(self.image_height_var.get())
    #    self.width = int(self.image_width_var.get())
    #
    #    # set opencv video stream/source height and width
    #    self.mediapipe_runtime.set_image_hw(self.height, self.width)



    # handle end of runtime (run when tkinter window closes)
    def __del__(self):

        # redundant set stop to true for all processes
        if not self.stop.is_set():
            self.stop.set()

        # stop gui
        #self.root.destroy()

        # stop threads
        self.extrap_handler.join()
        self.stream_handler.join()

        # stop mediapipe
        #if self.mediapipe_runtime.webcam_stream.isOpened():
        #    self.mediapipe_runtime.webcam_stream.release()
        #self.mediapipe_runtime.set_stop(True)
        #self.mediapipe_runtime.join()

        # end gui
        self.root.destroy()


