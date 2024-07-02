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

import json

# functions for data/physics calculations and manipulations
import extrapolation

# functions for calibrating the camera
from camera_calibration import *


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

# FaceLandmarker task object callback reference
FaceBaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
FaceVisionRunningMode = mp.tasks.vision.RunningMode



### CLASS

# set up a class to be used for running the pose detection program
class Pose_detection(threading.Thread):

    # initialization
    def __init__(self, pose_model_path, hand_model_path, face_model_path) -> None:
        # initialize thread
        threading.Thread.__init__(self)
        
        # allow model_path to be accessible to functions
        self.pose_model_path = pose_model_path
        self.hand_model_path = hand_model_path
        self.face_model_path = face_model_path

        # allow setting of frame height and width
        self.height = HEIGHT
        self.width = WIDTH

        # make video capture object via webcam
        self.webcam_stream = cv2.VideoCapture(video_source)
        # set height and width accordingly
        #HEIGHT = self.webcam_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #WIDTH = self.webcam_stream.get(cv2.CAP_PROP_FRAME_WIDTH)


        ### CAMERA CALIBRATION

        # make cropping optional (if True -> crop undistorted image, else -> don't crop)
        self.use_crop = True
        # make undistortion optional (if True -> undistort each frame, else -> don't undistort)
        #   if true, sets use_crop to false right before main running loop
        self.use_cam_calibration = True

        # set up camera calibration if enabled
        if self.use_cam_calibration:
            # get calibration data from file
            self.camera_matrix, self.dist_coeffs = self.load_cam_calib_data()
            # get the new camera intrinsic matrix and roi region for camera calibration
            self.camera_matrix_new, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (self.width, self.height), 1, (self.width, self.height))
            # get cropped height and width of image
            self.img_x, self.img_y, self.img_w, self.img_h = self.roi
        # set x y w and h to their default uncropped values, so as to not crop the image (and to avoid redundant check each frame)
        if not self.use_crop or not self.use_cam_calibration:
            self.img_x = 0
            self.img_y = 0
            self.img_w = self.width
            self.img_h = self.height


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

        # for use syncing annotated image
        #self.annotated_image_lock = threading.Lock()
        self.annot_img_finish = True


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

        # output data from hand landmarker
        self.hand_mp_out = np.zeros((2,5,3), dtype = "float32") # four points for each hand

        # output data from face landmarker
        self.face_mp_out = np.zeros((2,2,3), dtype = "float16") # two points for both eyes

        # initialize extrapolation and body force calculation object
        #self.right_arm = extrapolation.Extrapolate_forces(is_right = True)  # right arm
        #self.left_arm = extrapolation.Extrapolate_forces()             # left arm
        self.ep = extrapolation.Extrapolate_forces(self.mp_data_lock)                    # both arms
        
        #self.initialize_display()                                       # initialize display input
        # initialization of image (updated asynchronously)
        self.annotated_image = np.zeros((self.height, self.width, 3), np.uint8)
        # used as hand annotated image for use by gui (after being annotated by both HandLandmarker and PoseLandmarker)
        self.hand_annotated_image = self.annotated_image
        # full annotated image after landmarking from all three models
        self.full_annotated_image = self.hand_annotated_image

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
            num_hands = 2,
            running_mode = HandVisionRunningMode.LIVE_STREAM,
            result_callback = self.hand_draw_landmarks_on_frame
        )
        self.hand_detector = HandLandmarker.create_from_options(hand_options)    # load hand landmarker model for use in detection

        # options for face landmarker
        face_options = FaceLandmarkerOptions(
            base_options = FaceBaseOptions(model_asset_path = self.face_model_path),
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
            num_faces = 1,
            running_mode = FaceVisionRunningMode.LIVE_STREAM,
            result_callback = self.face_draw_landmarks_on_frame
        )
        self.face_detector = FaceLandmarker.create_from_options(face_options)

        print("Initialized Pose_detection()")

    # run the program
    def run(self):
        try:
            # display and update video stream
            if self.webcam_stream.isOpened() == False:
                print("ERROR opening webcam")       # make it so it doesnt crash when there's no webcam
            else:
                # loop using camera calibration
                #   split into two loops to avoid redundant check of self.use_cam_calibration every frame (more computationally efficient this way)
                if self.use_cam_calibration:

                    # main program loop
                    while not self.stop:    #((cv2.waitKey(1) & 0xFF == ord('q'))):    # or ret != True):'normal' == self.root.state():     # run while gui root is running     
                        #if cv2.waitKey(1) == 27:   # trying to get keyboard input to work. doesnt wanna lol
                        #    print("ESC pressed")
                        
                        # get current millisecond for use by detector
                        self.cur_msec = (int)(time.time() * 1000)
                        #self.cur_msec = cur_msec    # object-wide version of cur_msec (currently in testing)

                        # capture video for each frame
                        #self.ret, self.cur_frame = self.webcam_stream.read()                       # ret is true if frame available, false otherwise; cur_frame is current frame (image)
                        self.ret, raw_cur_frame = self.webcam_stream.read()                       # ret is true if frame available, false otherwise; cur_frame is current frame (image)
                        # undistort raw_cur_frame, use as self.cur_frame
                        self.cur_frame = get_undistorted(raw_cur_frame, self.camera_matrix, self.dist_coeffs, self.camera_matrix_new, self.roi)#, crop = True)

                        # run detector callback functions, updates annotated_image
                        self.pose_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), self.cur_msec )
                        
                # loop without camera calibration
                else:
                    # set self.use_crop to false, since there's no need if not using camera calibration
                    self.use_crop = False

                    # main program loop
                    while not self.stop:    #((cv2.waitKey(1) & 0xFF == ord('q'))):    # or ret != True):'normal' == self.root.state():     # run while gui root is running     
                        #if cv2.waitKey(1) == 27:   # trying to get keyboard input to work. doesnt wanna lol
                        #    print("ESC pressed")
                        
                        # get current millisecond for use by detector
                        self.cur_msec = (int)(time.time() * 1000)
                        #self.cur_msec = cur_msec    # object-wide version of cur_msec (currently in testing)

                        # capture video for each frame
                        self.ret, self.cur_frame = self.webcam_stream.read()                       # ret is true if frame available, false otherwise; cur_frame is current frame (image)
                        
                        # run detector callback functions, updates annotated_image
                        self.pose_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), self.cur_msec )
                        #self.hand_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = cur_frame ), cur_msec )
        except Exception as e:
            print("livestream_mediapipe_class.py: Exception in `run()`: \n\t%s" % str(e))
        finally:
            print("Info: Stopping mediapipe...")
        
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

    # load camera calibration data from `calibration/output.txt`
    def load_cam_calib_data(self, file = "output.txt"):
        try:
            path = os.path.join("calibration", file)        # get file path
            data_str = ""                                   # string to hold data read from file

            #ret, mtx, dist, rvecs, tvecs
            
            # read data from file
            with open(path, "r") as calib:
                data_str = calib.read()
                calib.close()

            # read in camera matrix and distortion coefficients for use getting undistorted images
            data = data_str.split(":")
            cam_mat = np.array(json.loads(data[1]))       # camera matrix
            dist = np.array(json.loads(data[2]))          # distortion coefficients

            return cam_mat, dist

        except Exception as e:
            print("livestream_mediapipe_class.py: Exception in `load_cam_calib_data: \n\t%s" % str(e))
        
        


    # helper function for use by GUI, returns current frame
    def get_cur_frame(self):
        return self.ret, self.full_annotated_image[self.img_y:self.img_y + self.img_h, self.img_x:self.img_x + self.img_w]
    
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

    # given 2D motion tracking data for a single frame, run calculations in extrapolation.py
    def extrapolate_depth(self, mediapipe_output):
        try:
            # set the data for the current frame
            self.ep.update_current_frame(mediapipe_output, self.hand_mp_out, self.face_mp_out, self.frame_counter)    # update mediapipe data
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
            mediapipe_out = np.ndarray((10, 3), dtype = "float32")

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
            #with self.annotated_image_lock:
            #if self.annot_img_finish:   # check if hand landmarker is done annotating prev image
            self.annotated_image = annotated_image  # set object's annotated_image variable to the local (to the function) one
            #    self.annot_img_finish = False           # prevents pose landmarker from overwriting annotated image before hand landmarker gets to it
            try:
                # call hand landmarker callback function after finishing for pose landmarker
                self.hand_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), self.cur_msec )
            except:
                print("livestream_mediapipe_class.py: ERROR handling hand_detector in draw_landmarks_on_frame()")

            #print("DEBUG: right elbow mediapipe coord: %s" % str(pose_landmarks_list[0][14].z))
            #print("DEBUG: right wrist mediapipe coord: %s" % str(pose_landmarks_list[0][16].z))
            #print("DEBUG: diff right elbow and wrist mediapipe coord: %s" % str(pose_landmarks_list[0][16].z - pose_landmarks_list[0][14].z))

            # add shoulders, elbows, and wrists to current dataframe
            it = 0                                                      # temp iterator
            for i in range(11, 17):
                #print(pose_landmarks_list[0][i].x)
                mediapipe_out[it, 0] = pose_landmarks_list[0][i].x      # get landmark data (x)
                mediapipe_out[it, 1] = pose_landmarks_list[0][i].z      # pass over mediapipe depth data to use for checking if vertex in front of or behind prev vertex
                mediapipe_out[it, 2] = pose_landmarks_list[0][i].y      # get landmark data (z) (using landmark.y due to different coordinate system)
                it += 1
            # also add index fingers
            for i in range (19, 21):
                mediapipe_out[it, 0] = pose_landmarks_list[0][i].x     # get landmark data (x)
                mediapipe_out[it, 1] = pose_landmarks_list[0][i].z      # pass over mediapipe depth data to use for checking if vertex in front of or behind prev vertex
                mediapipe_out[it, 2] = pose_landmarks_list[0][i].y     # get landmark data (z) (using landmark.y due to different coordinate system)
                it += 1
            # and hips too
            for i in range(23, 25):
                mediapipe_out[it, 0] = pose_landmarks_list[0][i].x     # get landmark data (x)
                mediapipe_out[it, 1] = pose_landmarks_list[0][i].z      # pass over mediapipe depth data to use for checking if vertex in front of or behind prev vertex
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
    # referenced https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
    def hand_draw_landmarks_on_frame(self, detection_result: HandLandmarkerResult, rgb_image: mp.Image, _):
        try:
            hand_landmarks_list = detection_result.hand_landmarks
            handedness_list = detection_result.handedness
            # get annotated_image after running draw_landmarks_on_frame for PoseLandmarker, use it as a base
            annotated_image = self.annotated_image #np.copy(rgb_image.numpy_view())
            # hold data for current hand data frame
            hand_mp_out = np.zeros((2,5,3), dtype = "float32")       # actual landmarker data (which hand, num vertices, num dimensions)

        
            # loop thru detected hand poses to visualize
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # draw the hand landmarks
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark in hand_landmarks:
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) 
                    ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )

                
            #with self.annotated_image_lock:# update object version of annotated_image
            #if not self.annot_img_finish:   # used to prevent sync issues with pose landmarker
            self.hand_annotated_image = annotated_image


            # uncomment these lines to enable FaceLandmarker
            #try:
                # call hand landmarker callback function after finishing for pose landmarker
            #    self.face_detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = self.cur_frame ), self.cur_msec )
            #except:
            #    print("livestream_mediapipe_class.py: ERROR handling face_detector in hand_draw_landmarks_on_frame()")
            self.full_annotated_image = annotated_image  # only use this if not using face_detector


            # put together the hand data we're looking for
            it = 0                      # iterator for iterating thru data frame (i.e. hand_mp_out)
            for i in handedness_list:       # do for each hand
                for j in (0, 5, 17, 13, 9):     # get the parts of the hand we're looking for (wrist, index knuckle, pinky knuckle, ring knuckle, middle knuckle)
                    # get which hand
                    hand = i[0].index       # from mediapipe, left hand is 1, right is 0

                    # swap hand values, so left is 0 and right is 1, in accordance with the rest of this whole program
                    if hand:
                        hand = 0
                        # DEBUG
                        #print("Hand vertex: %s\t(x,y,z): (%s, %s, %s)" %
                        #    (str(j), str(hand_landmarks_list[0][j].x), str(hand_landmarks_list[0][j].y), str(hand_landmarks_list[0][j].z)))
                    else:
                        hand = 1
                    
                    # put hand position data into ndarray for sending to extrapolation.py
                    hand_mp_out[hand, it, 0] = hand_landmarks_list[0][j].x
                    hand_mp_out[hand, it, 1] = hand_landmarks_list[0][j].z  # z and y swapped due to convention of prev used coord system
                    hand_mp_out[hand, it, 2] = hand_landmarks_list[0][j].y
                    it += 1             # iterate
                it = 0                  # reset iterator before moving to next hand (if available)

            # update object hand data
            for i in range(0, 2):   # for each of the hands
                self.hand_mp_out[i] = hand_mp_out[i]

            #print("\n\n(DEBUG) HAND DATA: %s" % str(hand_mp_out))
            

        except Exception as e:
            print("livestream_mediapipe_class.py: ERROR with mediapipe in hand_draw_landmarks_on_frame()")
            print("\tException: %s" % str(e))

        return 1

    # face landmarker callback function
    def face_draw_landmarks_on_frame(self, detection_result: FaceLandmarkerResult, rgb_image: mp.Image, _):
        try:
            face_landmarks_list = detection_result.face_landmarks
            # get annotated image after rubnning prev landmarkers
            annotated_image = self.annotated_image
            # used as temp store for data prior to passing to extrapolation.py
            face_mp_out = np.zeros((2,2,3), dtype = "float16")

            # loop thru face landmarks to visualize
            for idx in range(len(face_landmarks_list)):
                face_landmarks = face_landmarks_list[idx]
                #print("DEBUG: Size of face_landmarks_list: %s\nSize of face_landmarks: %s" % (str(len(face_landmarks_list)), str(len(face_landmarks))))

                # draw face landmarks
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in face_landmarks
                ])

                solutions.drawing_utils.draw_landmarks(
                    image = annotated_image,
                    landmark_list = face_landmarks_proto,
                    connections = mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec = None,
                    # only draw the iris style, since that's all we're looking for (there's also "contours style" and "tesselation style")
                    connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                )

            # set annotated image for use in GUI
            self.full_annotated_image = annotated_image


            # put together the data we're looking for (only four points, to get width of each iris)
            # done using two separate loops for sake of efficiency
            it = 0  # iterator
            # right eye
            for i in (469, 471):    # left point of iris (469), right point of iris (471)
                face_mp_out[1, it, 0] = face_landmarks_list[0][i].x
                face_mp_out[1, it, 1] = face_landmarks_list[0][i].z
                face_mp_out[1, it, 2] = face_landmarks_list[0][i].y
                it += 1
            it = 0  # reset iterator
            # left eye
            for i in (474, 476):    # left point of iris (474), right point of iris (476)
                face_mp_out[0, it, 0] = face_landmarks_list[0][i].x
                face_mp_out[0, it, 1] = face_landmarks_list[0][i].z
                face_mp_out[0, it, 2] = face_landmarks_list[0][i].y

            # update iris data all at once to prevent sync issues
            self.face_mp_out = face_mp_out

        except Exception as e:
            print("livestream_mediapipe_class.py: ERROR in face_draw_landmarks_on_frame():\t%s" % str(e))

    

    ### HELPER FUNCTIONS

    # handle toggleable auto calibration/conversion ratio calculation
    def toggle_auto_conversion(self, toggle = True):
        self.toggle_auto_calibrate = toggle
        self.ep.set_calibration_manual(toggle)


#testing = Pose_detection(pose_landmarker)
#testing.run()
