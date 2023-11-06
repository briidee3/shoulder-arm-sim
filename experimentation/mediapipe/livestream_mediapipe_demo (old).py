# BD 2023
# This file used for testing mediapipe functionality for use in simulating forces pertaining to the human arm system
# as well as implementing and integrating a custom algorithm for discerning depth from 2D skeleton data

# testing code from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#live-stream to start out with


# TODO: 
#   - get basic program working
#       - skeleton overlay over live stream
#   - set up multithreading
#       - main process is running, then one thread for handling live stream, and another one for calculating skeleton


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


### OPTIONS

# model to be used as "Pose Landmarker"
pose_landmarker = './landmarkers/pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



### PREP VIDEO LIVE STREAM

# use opencv VideoCapture to start webcam capture
webcam_stream = cv2.VideoCapture(0)     # make video capture object
ret, cur_frame = webcam_stream.read()   # called pre-loop for access outside of the loop

# initialize annotated_img, current frame data annotated and updated by callback function "draw_landmarks_on_frame"
annotated_img = cur_frame




### CREATE/INITIALIZE "TASK"

# callback function for getting async result detected in livestream mode
#def get_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#    return output_image, result

# detector callback function
# annotate and display frame with skeleton
def draw_landmarks_on_frame(self, detection_result, rgb_image, _):  #(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

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
    
    # run detector callback function, updates annotated_img
    detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = cur_frame ), cur_msec )
    
    # set current frame to annotated image
    #global annotated_img            # reference global variable "annotated_img"
    annotated_img = annotated_image # set variable to value of "annotated_image"


# options for pose landmarker
options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = pose_landmarker),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = draw_landmarks_on_frame
)
detector = PoseLandmarker.create_from_options(options)   # load landmarker model for use in detection



### FUNCTIONS

# detect pose landmarks, get skeleton from input image
#def detect_pose(current_frame, cur_msec):                   # "cur_msec" is for use by "detect_async"
    # convert current frame to mediapipe Image format, detect with detector, and return the result.
    # using "detect_async" instead of "detect" to only do latest frame (since live streaming data)
#    return detector.detect_async( mp.Image( image_format = mp.ImageFormat.SRGB, data = current_frame ), cur_msec )

# detect landmarks, draw on frame, convert from RGB to BGR (as per openCV standard)
#def display_frame_with_skeleton(current_frame, cur_msec):        # "cap" is current cv2.VideoCapture() object, used for getting frame times
    #cv2.imshow( 'Live view + overlay (Press "q" to exit)', 
    #    cv2.cvtColor( draw_landmarks_on_image( current_frame, detect_pose(current_frame, cur_msec) ), cv2.COLOR_RGB2BGR )
    #)
    # run async detect_pose beforehand, then we can use the output from display_result()
#    detect_pose(current_frame, cur_msec)
    # display result
#    cv2.imshow( 'Live view + overlay (Press "q" to exit)', 
#        cv2.cvtColor( draw_landmarks_on_image( get_result() ), cv2.COLOR_RGB2BGR )
#    )



### RUN PROGRAM

# display and update video stream
if webcam_stream.isOpened() == False:
    print("Error opening webcam")
else:
    # main program loop
    while True:
        # get current millisecond for use by detector
        cur_msec = (int)(webcam_stream.get(cv2.CAP_PROP_POS_MSEC))  # done here for synchronization purposes

        # capture video for each frame
        ret, cur_frame = webcam_stream.read()                       # ret is true if frame available, false otherwise; cur_frame is current frame (image)

        # show (raw) frame on screen (without skeleton overlay)
        cv2.imshow('Live Webcam View (Press "q" to exit)', cur_frame)


        # show current frame with skeleton overlay on screen

        # display annotated image on screen
        cv2.imshow( 'Live view + overlay (Press "q" to exit)', cv2.cvtColor( annotated_img, cv2.COLOR_RGB2BGR ) )


        # allow program to be quit when the "q" key is pressed or webcam stream gone
        if (cv2.waitKey(1) & 0xFF == ord('q')) or ret != True:
            break

# release capture object from memory
webcam_stream.release()
# get rid of windows still up
cv2.destroyAllWindows()
