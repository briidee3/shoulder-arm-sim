# BD 2023-24
# This file used for testing mediapipe functionality for use in simulating forces pertaining to the human arm system
# as well as implementing and integrating a custom algorithm for discerning depth from 2D skeleton data

# testing code from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#live-stream to start out with


# TODO: 
#   - ensure depth calculations are atomically locked per frame
#       - work on most recent frame
#       - don't start another until the current one is finished
#   - set up multithreading
#       - main process is running, then one thread for handling live stream, and another one for calculating skeleton
#   - figure out how to smooth the data outputs (for example, by using median of past 5 frames)
#       - if a given vector moves too quickly between two frames, ignore the next frame
#           so as to help ignore junk data
#       - figure out proper method of ignoring garbage data efficiently and effectively
#           (e.g. figuring out which data is reliable or not, for example by using the 
#           "accuracy" data from mediapipe output, and if it is below a certain threshold,
#           then don't send that frame's data to extrapolation)
#   - fix crash when mediapipe detects 0 or more than 1 person


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

# functions for data/physics calculations and manipulations
import extrapolation

# testing input from keyboard
#key = cv2.waitKey(0)

### MEDIAPIPE OPTIONS

# video source (default: 0) default typically denotes built in webcam
video_source = 0

# model to be used as "Pose Landmarker"
#pose_landmarker = '../landmarkers/pose_landmarker_heavy.task'
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

# HandLandmarker task object callback references
HandBaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandVisionRunningMode = mp.tasks.vision.RunningMode



### CLASS

# set up a class to be used for running the pose detection program
class Pose_detection(threading.Thread):

    # initialization
    def __init__(self, pose_model_path, hand_model_path) -> None:
        # initialize thread
        threading.Thread.__init__(self)
        
        # allow model_path to be accessible to functions
        self.pose_model_path = pose_model_path
        self.hand_model_path = hand_model_path

        # allow setting of frame height and width
        self.height = HEIGHT
        self.width = WIDTH

        # make video capture object via webcam
        self.webcam_stream = cv2.VideoCapture(video_source)
        # set height and width accordingly
        #HEIGHT = self.webcam_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #WIDTH = self.webcam_stream.get(cv2.CAP_PROP_FRAME_WIDTH)

        # test webcam
        if self.webcam_stream is None or not self.webcam_stream.isOpened():
            print("Warning: unable to open camera (camera source: %s)" % video_source)
        else:
            print("Info: Initialized webcam (source: %s)" % video_source)
        
        print("Info: Initialized PoseLandmarker")        


        ### SET UP PIPELINE

        # boolean handlers
        self.stop = False
        self.toggle_auto_calibrate = False

        # lock for mediapipe output data
        self.mp_data_lock = threading.Lock()

        # allow use of current frame from external program (GUI)
        self.ret = None
        self.cur_frame = None
        self.cur_msec = 0
        
        # helps with counting frames across functions
        self.frame_counter = 0                                          # used to keep track of which frame is which
        self.tick_length = 60                                           # num of frames between periodic updater functions (e.g. calibration)

        # user input data
        self.user_height = 1.78                                         # in meters
        self.user_weight = 90                                           # in kilograms

        # set up dictionary to read from for gui display of data
        self.calculated_data = {
            "right_bicep_force": "NaN",
            "right_elbow_angle": "NaN",
            "left_bicep_force": "NaN",
            "left_elbow_angle": "NaN",
            "uarm_spher_coords": "NaN",#["NaN", "NaN", "NaN"],
            "farm_spher_coords": "NaN"#["NaN", "NaN", "NaN"]
        }

        # initialize extrapolation and body force calculation object
        #self.right_arm = extrapolation.Extrapolate_forces(is_right = True)  # right arm
        #self.left_arm = extrapolation.Extrapolate_forces()             # left arm
        self.ep = extrapolation.Extrapolate_forces(self.mp_data_lock)                    # both arms
        
        #self.initialize_display()                                       # initialize display input
        # initialization of image (updated asynchronously)
        self.annotated_image = np.zeros((self.height, self.width, 3), np.uint8)

        # options for pose landmarker
        options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path = self.pose_model_path),
            running_mode = VisionRunningMode.LIVE_STREAM,
            result_callback = self.draw_landmarks_on_frame
        )
        self.pose_detector = PoseLandmarker.create_from_options(options)     # load pose landmarker model for use in detection

        # options for hand landmarker
        hand_options = HandLandmarkerOptions(
            base_options = HandBaseOptions(model_asset_path = self.hand_model_path),
            running_mode = HandVisionRunningMode.LIVE_STREAM,
            result_callback = self.hand_draw_landmarks_on_frame
        )
        self.hand_detector = HandLandmarker.create_from_options(hand_options)    # load hand landmarker model for use in detection

        print("Initialized Pose_detection()")

    # run the program
    def run(self):
        #try:
        # display and update video stream
        if self.webcam_stream.isOpened() == False:
            print("ERROR opening webcam")       # make it so it doesnt crash when there's no webcam
        else:
            # main program loop
            while not self.stop:    #((cv2.waitKey(1) & 0xFF == ord('q'))):    # or ret != True):'normal' == self.root.state():     # run while gui root is running     
                #if cv2.waitKey(1) == 27:   # trying to get keyboard input to work. doesnt wanna lol
                #    print("ESC pressed")
                
                # get current millisecond for use by detector
                cur_msec = (int)(time.time() * 1000)
                self.cur_msec = cur_msec    # object-wide version of cur_msec (currently in testing)

                # capture video for each frame
                ret, cur_frame = self.webcam_stream.read()                       # ret is true if frame available, false otherwise; cur_frame is current frame (image)
                # set object variables
                self.ret = ret
                self.cur_frame = cur_frame

                # run detector callback functions, updates annotated_image
                self.pose_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), self.cur_msec )
                #self.hand_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = cur_frame ), cur_msec )
        #except:
        #    print("livestream_mediapipe_class.py: ERROR in `run()`")
        #finally:
        #    print("Info: Stopping mediapipe...")
        self.stop_program()

    # initialize display/camera input
    #def initialize_display(self): 
    #    try:
            
    #    except:
    #        print("livestream_mediapipe_class.py: ERROR in `initialize_display()`")
        
    # set height and width of image
    def set_image_hw(self, height = HEIGHT, width = WIDTH):
        try:
            # set local variables
            self.height = int(height)
            self.width = int(width)

            # reset annotated_image to new size
            #self.annotated_image = np.zeros((self.height, self.width, 3), np.uint8)

            # set height and width of opencv video stream
            self.webcam_stream.set(3, float(self.width))
            self.webcam_stream.set(4, float(self.height))
        except:
            print("livestream_mediapipe_class.py: ERROR in `set_image_hw()`")


    # helper function for use by GUI, returns current frame
    def get_cur_frame(self):
        return self.ret, self.annotated_image
    
    # return current calculated data
    def get_calculated_data(self):
        return dict(self.calculated_data)
    
    # allow setting of user height via external package/program
    def set_height(self, height):
        self.user_height = height
        return self.user_height
    
    # allow setting of user weight via external package/program
    def set_weight(self, weight):
        self.user_weight = weight
        return self.user_weight
    
    # get function for video height and width
    def get_height_width(self):
        return self.height, self.width
    
    # set stop variable
    def set_stop(self, set_ = True):
        self.stop = set_
    
    # callback function to terminate program
    def stop_program(self):
        self.stop = True    # redundancy
        # release capture object from memory
        self.webcam_stream.release()
        # get rid of windows still up
        cv2.destroyAllWindows()
        print("Program closed.")
        #exit()


    ### DEPTH EXTRAPOLATION and BODY FORCE CALCULATIONS

    # given 2D motion tracking data for a single frame, return 3D motion tracking data for a single frame
    def extrapolate_depth(self, mediapipe_output):
        try:
            # set the data for the current frame
            self.ep.update_current_frame(mediapipe_output, self.frame_counter)    # update mediapipe data
            # calculations that don't need to run each frame (hence run every "tick")
            #if not self.toggle_auto_calibrate and (self.frame_counter % self.tick_length == 0):    # now done in extrapolation.py each frame update
            #    self.ep.calc_conversion_ratio(real_height_metric = self.user_height)  # calculate conversion ratio (mediapipe units to meters)

        except:
            print("livestream_mediapipe_class.py: ERROR in extrapolate_depth()")

        # calculate depth for given frame (now done in `extrapolation.py`)
        #try:
        #    self.ep.set_depth()                      # get depth_dict and calculate y axes values
        #except:
        #    print("livestream_mediapipe_class.py: ERROR with ep.set_depth() in extrapolate_depth()")

    # calculate forces involved with muscles in the body
    def calc_body_forces(self):
        # force calculations
        self.calculated_data = self.ep.get_calculated_data()                     # calculate forces

        # display forces graph
        #self.ep.plot_picep_forces().show()                                   # display a graph depicting calculated bicep forces


    # pose detector callback function
    # annotate and display frame with skeleton
    # TODO:
    #   - once HandLandmarker is all set, stop drawing the hand landmarks from PoseLandmarker
    def draw_landmarks_on_frame(self, detection_result: PoseLandmarkerResult, rgb_image: mp.Image, _):  #(rgb_image, detection_result):
        try:
            pose_landmarks_list = detection_result.pose_landmarks
            annotated_image = np.copy(rgb_image.numpy_view())
            mediapipe_out = np.ndarray((10, 3))

            # loop thru detected poses to visualize
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # draw the pose landmarks
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
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
            try:
                # call hand landmarker callback function after finishing for pose landmarker
                self.hand_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), self.cur_msec )
            except:
                print("livestream_mediapipe_class.py: ERROR handling hand_detector in draw_landmarks_on_frame()")


            # add shoulders, elbows, and wrists to current dataframe
            it = 0                                                      # temp iterator
            for i in range(11, 17):
                #print(pose_landmarks_list[0][i].x)
                mediapipe_out[it, 0] = pose_landmarks_list[0][i].x     # get landmark data (x)
                mediapipe_out[it, 1] = 0              # set y (depth) data
                mediapipe_out[it, 2] = pose_landmarks_list[0][i].y     # get landmark data (z) (using landmark.y due to different coordinate system)
                it += 1
            # also add index fingers
            for i in range (19, 21):
                mediapipe_out[it, 0] = pose_landmarks_list[0][i].x     # get landmark data (x)
                mediapipe_out[it, 1] = 0              # set y (depth) data
                mediapipe_out[it, 2] = pose_landmarks_list[0][i].y     # get landmark data (z) (using landmark.y due to different coordinate system)
                it += 1
            # and hips too
            for i in range(23, 25):
                mediapipe_out[it, 0] = pose_landmarks_list[0][i].x     # get landmark data (x)
                mediapipe_out[it, 1] = 0              # set y (depth) data
                mediapipe_out[it, 2] = pose_landmarks_list[0][i].y     # get landmark data (z) (using landmark.y due to different coordinate system)
                it += 1
        except:
            print("livestream_mediapipe_class.py: ERROR with mediapipe in draw_landmarks_on_frame()")
        
        ### DEPTH AND FORCES CALCULATIONS
        try:
            if (pose_landmarks_list) and not self.stop:   # check if results exist (and that program isn't stopping) before attempting calculations
                #print("Extrapolating depth...")
                #print("DEBUG: pose_landmarks_list: %s" % mediapipe_out)
                with self.mp_data_lock:
                    self.extrapolate_depth(mediapipe_out)
                #print("Calculating body forces...")
                #self.calc_body_forces()
            
            self.frame_counter += 1
        except:
            print("livestream_mediapipe_class.py: ERROR with depth/force calculations in draw_landmarks_on_frame()")
        
        
        return 1

    # hand detector callback function
    # TODO: 
    #   - make it so that this utilizes as many components as possible from `draw_landmarks_on_frame` so as to
    #     not repeat things and waste memory in the process  
    #   - or just see about running hand landmarks in a different thread or multiprocess it
    def hand_draw_landmarks_on_frame(self, detection_result: HandLandmarkerResult, rgb_image: mp.Image, _):
        try:
            hand_landmarks_list = detection_result.hand_landmarks
            # get annotated_image after running draw_landmarks_on_frame for PoseLandmarker, use it as a base
            annotated_image = self.annotated_image #np.copy(rgb_image.numpy_view())
            
            # loop thru detected hand poses to visualize
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # draw the hand landmarks
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark in hand_landmarks:
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) 
                    ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    solutions.hand.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style()
                )
        except:
            print("livestream_mediapipe_class.py: ERROR with mediapipe in hand_draw_landmarks_on_frame()")

        return 1
    

    ### HELPER FUNCTIONS

    # handle toggleable auto calibration/conversion ratio calculation
    def toggle_auto_conversion(self, toggle = True):
        self.toggle_auto_calibrate = toggle
        self.ep.set_calibration_manual(toggle)


#testing = Pose_detection(pose_landmarker)
#testing.run()
