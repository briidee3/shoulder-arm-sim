# BD 2023
# This file used for testing mediapipe functionality for use in simulating forces pertaining to the human arm system
# as well as implementing and integrating a custom algorithm for discerning depth from 2D skeleton data

# testing code from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#live-stream to start out with


# TODO: 
#   - bring in stuff from previous iteration of the project
#   - ensure depth calculations are atomically locked per frame
#       - work on most recent frame
#       - don't start another until the current one is finished
#   - set up multithreading
#       - main process is running, then one thread for handling live stream, and another one for calculating skeleton
#   - set up TKinter for GUI stuff
#       - display annotated_image in tkinter window
#       - set up a class variable for reading and writing calc_bicep_forces output (python dict object)


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import math

import cv2
import threading
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# functions for data calculations and manipulations
import extrapolation

# functions for GUI setup/management
from tkinter import *
from PIL import Image
from PIL import ImageTk



### MEDIAPIPE OPTIONS

# model to be used as "Pose Landmarker"
pose_landmarker = './landmarkers/pose_landmarker_full.task'
WIDTH = 640
HEIGHT = 480

# use opencv VideoCapture to start webcam capture
#ret, cur_frame = webcam_stream.read()   # called pre-loop for access outside of the loop

# PoseLandmarker task object callback references
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



### CLASS

# set up a class to be used for running the pose detection program
class Pose_detection(threading.Thread):

    # initialization
    def __init__(self, model_path) -> None:
        # initialize thread
        threading.Thread.__init__(self)

        # make video capture object via webcam
        self.webcam_stream = cv2.VideoCapture(0)
        # set height and width accordingly
        #HEIGHT = self.webcam_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #WIDTH = self.webcam_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        
        # initialization of image (updated asynchronously)
        self.annotated_image = np.zeros(((int)(HEIGHT), (int)(WIDTH), 3), np.uint8)

        # options for pose landmarker
        options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = model_path),
            running_mode = VisionRunningMode.LIVE_STREAM,
            result_callback = self.draw_landmarks_on_frame
        )
        self.detector = PoseLandmarker.create_from_options(options)     # load landmarker model for use in detection
        

        ### SET UP PIPELINE

        # thread for running calculations
        self.stop = False

        # allow use of current frame from external program (GUI)
        self.ret = None
        self.cur_frame = None
        
        # helps with counting frames across functions
        self.frame_counter = 0                                          # used to keep track of which frame is which
        self.tick_length = 60                                           # num of frames between periodic updater functions (e.g. calibration)

        # user input data
        self.user_height = 1.78                                         # in meters
        self.user_weight = 90                                           # in kilograms

        # initialize extrapolation and body force calculation object
        self.ep = extrapolation.Extrapolate_forces()
        print("Initialized Pose_detection()")

    # run the program
    def run(self):
        try:
            # display and update video stream
            if self.webcam_stream.isOpened() == False:
                print("Error opening webcam")
            else:
                # main program loop
                while not self.stop:    #((cv2.waitKey(1) & 0xFF == ord('q'))):    # or ret != True):'normal' == self.root.state():     # run while gui root is running     
                    # get current millisecond for use by detector
                    cur_msec = (int)(time.time() * 1000)

                    # capture video for each frame
                    self.ret, self.cur_frame = self.webcam_stream.read()                       # ret is true if frame available, false otherwise; cur_frame is current frame (image)

                    # run detector callback function, updates annotated_image
                    self.detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), cur_msec )
        finally:
            self.stop = True
            # release capture object from memory
            self.webcam_stream.release()
            # get rid of windows still up
            cv2.destroyAllWindows()
            print("Program closed.")

    # helper function for use by GUI, returns current frame
    def get_cur_frame(self):
        return self.ret, self.annotated_image
    
    # allow setting of height via external package/program
    def set_height(self, height):
        self.user_height = height
        return self.user_height
    
    # allow setting of weight via external package/program
    def set_weight(self, weight):
        self.user_weight = weight
        return self.user_weight
    
    # get function for video height and width
    def get_height_width(self):
        return HEIGHT, WIDTH
    
    # callback function to terminate program
    def stop_program(self):
        self.stop = True


    ### DEPTH EXTRAPOLATION and BODY FORCE CALCULATIONS

    # given 2D motion tracking data for a single frame, return 3D motion tracking data for a single frame
    def extrapolate_depth(self, mediapipe_output):
        # set the data for the current frame
        self.ep.update_current_frame(mediapipe_output, self.frame_counter)                       # update mediapipe data
        # calculations that don't need to run each frame (hence run every "tick")
        if (self.frame_counter % self.tick_length == 0):
            self.ep.calc_conversion_ratio(real_height_metric = self.user_height)  # calculate conversion ratio (mediapipe units to meters)

        # calculate depth for given frame
        self.ep.set_depth()                      # get depth_dict and calculate y axes values

    # calculate forces involved with muscles in the body
    def calc_body_forces(self):
        # force calculations
        self.ep.calc_bicep_force()                                            # calculate forces

        # display forces graph
        #self.ep.plot_picep_forces().show()                                   # display a graph depicting calculated bicep forces


    # detector callback function
    # annotate and display frame with skeleton
    def draw_landmarks_on_frame(self, detection_result: PoseLandmarkerResult, rgb_image: mp.Image, _):  #(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image.numpy_view())
        mediapipe_out = np.ndarray((8, 3))

        # loop thru detected poses to visualize
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # draw the pose landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            # note: the following may seem sub-optimal, however it prevents many unnecessary if statements in run time
            # check if arms, shoulders, or wrists. if so, save and send that data to further process
            if (idx >= 11) and (idx <= 16):                         # 11 = left shoulder, 17 = right wrist. 
                for landmark in pose_landmarks:
                    pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) 
                    ])
                    index_offset = idx - 11
                    mediapipe_out[index_offset, 0] = landmark.x     # get landmark data (x)
                    mediapipe_out[index_offset, 1] = 0              # set y (depth) data
                    mediapipe_out[index_offset, 2] = landmark.y     # get landmark data (z) (using landmark.y due to different coordinate system)
            # check for index fingers (wingspan)
            elif (idx == 19 or idx == 20):                          # 19 = left index, 20 = right index (wingspan)
                for landmark in pose_landmarks:
                    pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) 
                    ])
                    index_offset = idx - 13
                    mediapipe_out[index_offset, 0] = landmark.x     # get landmark data (x)
                    mediapipe_out[index_offset, 1] = 0              # set y (depth) data
                    mediapipe_out[index_offset, 2] = landmark.y     # get landmark data (z) (using landmark.y due to different coordinate system
             # don't save data in mediapipe_out if not arms, shoulders, or wrists
            else:
                for landmark in pose_landmarks:
                    pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) 
                    ])

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        # set current frame to annotated image
        self.annotated_image = annotated_image  # set object's annotated_image variable to the local (to the function) one
        
        
        ### DEPTH AND FORCES CALCULATIONS
        if (pose_landmarks_list) and not self.stop:   # check if results exist (and that program isn't stopping) before attempting calculations
            print("Extrapolating depth...")
            print("DEBUG: pose_landmarks_list: %s" % mediapipe_out)
            self.extrapolate_depth(mediapipe_out)
            print("Calculating body forces...")
            self.calc_body_forces()
        
        self.frame_counter += 1
        
        
        return #?

