# BD 2023
# This file used for testing mediapipe functionality for use in simulating forces pertaining to the human arm system
# as well as implementing and integrating a custom algorithm for discerning depth from 2D skeleton data

# testing code from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#live-stream to start out with


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


## CREATE/INITIALIZE "TASK"

# options and constants/references
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# create pose landmarker instance with live stream mode
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

# options for pose landmarker
options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = './landmarkers/pose_landmarker_full.task'),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result
)

# initialize and use landmarker
with PoseLandmarker.create_from_options(options) as landmarker:
    # landmarker initialized; use it here
    i = 1   # dummy code



## PREP VIDEO LIVE STREAM

# use opencv VideoCapture to start webcam capture
webcam_stream = cv2.VideoCapture(0)     # make video capture object
ret, cur_frame = webcam_stream.read()   # called pre-loop for access outside of the loop

# display and update video stream
if webcam_stream.isOpened() == False:
    print("Error opening webcam")
else:
    while True:     # infinite loop
        # cap video for each frame
        ret, cur_frame = webcam_stream.read()

        # show frame on screen
        cv2.imshow('Live Webcam View', cur_frame)

        # allow program to be quit when the "q" key is pressed or webcam stream gone
        if (cv2.waitKey(1) & 0xFF == ord('q')) or ret != True:
            break
# release capture object from memory
webcam_stream.release()
# get rid of windows still up
cv2.destroyAllWindows()
