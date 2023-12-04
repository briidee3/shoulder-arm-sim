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


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2
import threading
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import extrapolation as ep      # functions from previous iteration of project



# user input variables (treated as constants)
user_height = 1.78      # height of the user
user_weight = 90        # weight of the user

# number of frames to wait between updates of certain data (so as to not bog down the machine each frame)
tick_length = 60        # not used yet
frame_counter = 0       # used for keeping track of current tick/frame



### DEPTH EXTRAPOLATION and BODY FORCE CALCULATIONS

# given 2D motion tracking data for a single frame, return 3D motion tracking data for a single frame
def extrapolate_depth(mediapipe_output):
    # set the data for the current frame
    ep.update_current_frame(mediapipe_output, frame_counter)                       # update mediapipe data
    
    # calculations that don't need to run each frame (hence run every "tick")
    if (frame_counter % tick_length == 0):
        ep.calc_conversion_ratio(real_height_metric = user_height)  # calculate conversion ratio (mediapipe units to meters)

    # calculate depth for given frame
    ep.set_depth(ep.get_axes_set_depth_dict())                      # get depth_dict and calculate y axes values

# calculate forces involved with muscles in the body
def calc_body_forces():
    # force calculations
    ep.set_elbow_angle()                                            # find angle at elbow
    ep.set_spher_coords()                                           # calculate spherical coordinates
    ep.run_formula_calculations()                                   # calculate forces

    # display forces graph
    ep.plot_picep_forces().show()                                   # display a graph depicting calculated bicep forces

    frame_conter += 1                                               # update frame counter


### OPTIONS

# model to be used as "Pose Landmarker"
pose_landmarker = './landmarkers/pose_landmarker_full.task'
WIDTH = 640
HEIGHT = 480

# use opencv VideoCapture to start webcam capture
webcam_stream = cv2.VideoCapture(0)     # make video capture object
#ret, cur_frame = webcam_stream.read()   # called pre-loop for access outside of the loop

# PoseLandmarker task object callback references
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



### CLASS

# set up a class to be used for running the pose detection program
class Pose_detection():

    # initialization
    def __init__(self, model_path) -> None:
        # initialization of image (updated asynchronously)
        self.annotated_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        # options for pose landmarker
        options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = model_path),
            running_mode = VisionRunningMode.LIVE_STREAM,
            result_callback = self.draw_landmarks_on_frame
        )
        self.detector = PoseLandmarker.create_from_options(options) # load landmarker model for use in detection
        
        extrapolate_depth(PoseLandmarkerResult, 0)                  # initialize extrapolation
        ep.update_first_frame()

    # run the program
    def run(self):
        try:
            # display and update video stream
            if webcam_stream.isOpened() == False:
                print("Error opening webcam")
            else:
                # main program loop
                while not ((cv2.waitKey(1) & 0xFF == ord('q'))):# or ret != True):
                    # get current millisecond for use by detector
                    cur_msec = (int)(time.time() * 1000)

                    # capture video for each frame
                    self.ret, self.cur_frame = webcam_stream.read()                       # ret is true if frame available, false otherwise; cur_frame is current frame (image)

                    # show (raw) frame on screen (without skeleton overlay)
                    #cv2.imshow('Live Webcam View (Press "q" to exit)', self.cur_frame)


                    # show current frame with skeleton overlay on screen
                    # 
                    # run detector callback function, updates annotated_image
                    self.detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), cur_msec )
                    # display annotated image on screen
                    cv2.imshow( 'Live view + overlay (Press "q" to exit)', cv2.cvtColor( self.annotated_image, cv2.COLOR_RGB2BGR ) )

                    

                    # perform forces calculations
                    extrapolate_depth(ret)


                    # allow resetting the data to allow others to use without restarting
                    #ep.reset_dist_array()


        finally:
            # release capture object from memory
            webcam_stream.release()
            # get rid of windows still up
            cv2.destroyAllWindows()
            print("Program closed.")



    # detector callback function
    # annotate and display frame with skeleton
    def draw_landmarks_on_frame(self, detection_result: PoseLandmarkerResult, rgb_image: mp.Image, _):  #(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image.numpy_view())

        # loop thru detected poses to visualize
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # draw the pose landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        # set current frame to annotated image
        self.annotated_image = annotated_image  # set object's annotated_image variable to the local (to the function) one
        
        
        # depth and forces calculations
        extrapolate_depth(detection_result)
        calc_body_forces()
        
        
        return #?



# try running everything

# make new object of type Pose_detection (defined above)
program = Pose_detection(pose_landmarker)
# run the object/program
program.run()




