import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


"""
Bri imports
"""


import numpy as np
import math
from matplotlib import pyplot as plt

import threading
import os

import cv2

from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL import ImageTk

import livestream_mediapipe_class as lsmp   # custom class, handles mediapipe

import sqlite3


"""
^^^
"""


import os

#import gui
print("Current Working Directory:", os.getcwd())
base_dir = os.path.dirname(os.path.realpath(__file__))

pose_landmarker = os.path.join(base_dir, 'landmarkers', 'pose', 'pose_landmarker_full.task')
hand_landmarker = os.path.join(base_dir, 'landmarkers', 'hand', 'hand_landmarker.task')


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# Variables for data output (placeholders)
direction_num = 0
direction_facing = "Unknown"
last_update_time = 0  # Variable to track the time of the last update
max_shoulder_size = 0
tickCheck = 0
user_height = 170 #cm
user_depth = 150 #cm
wait_for_update = 0
once = True
once2 = True
once3 = True
left_shoulder_z = 0
left_elbow_z = 0
user_weight = 58.967 #kg
forearm = (user_height*0.01) * 0.216
upperarm = (user_height*0.01) * 0.173
cfg = forearm * 0.432
b = forearm * 0.11
weightForearm = user_weight * 0.023
weightAdded = 0
developer_mode = False  # Developer mode state
force_data = 0
angle_data = 0
leftArmAngle = 0
left_arm_bicep_force = 0
time_to_get_position_var = 0
time_simulation_var = 0
start_time = 0
twoStepDone = False
twoStepCountDown = 10
isGraphOn = True
depth_ratio = 0
time_simulation_active = 60 

developer_mode = True
isGraphOn = True


"""
Enable Start Up Bypass
\/ \/ \/ \/ \/ \/ \/ \/
"""
BypassStartUp = False
"""
/\ /\ /\ /\ /\ /\ /\ /\ 
Enable Start Up Bypass
"""

# Assuming you have an image file named "instruction_image.png" in the same directory as your script
instruction_image_1_path = "pose1.png"  # Wingspan
instruction_image_2_path = "pose2.png"  # Blocking
instruction_image_3_path = "pose3.png"  # Arms By Side

horizontal_line_position_1 = 0
horizontal_line_position_2 = 0
horizontal_line_position_3 = 0
horizontal_line_position_4 = 0
horizontal_line_position_5 = 0
horizontal_line_position_6 = 0
vertical_line_position_1 = 0
vertical_line_position_2 = 0
vertical_line_position_3 = 0
vertical_line_position_4 = 0
vertical_line_position_5 = 0
vertical_line_position_6 = 0

# Global variables for the positions of the circles
initial_circle_positions = {}
current_stage = 'overlay_1'







"""

Inits

"""

# Initialize variables to zero
init_distance_shoulder = 0
init_distance_hip_shoulder = 0
init_left_distance_hip_shoulder = 0
init_height_diff_right_shoulder_to_right_hip = 0
init_head_width = 0
init_nose_eye_ear_angle = 0
init_right_shoulder_to_right_elbow = 0
init_right_elbow_to_right_wrist = 0
init_left_shoulder_to_left_elbow = 0
init_left_elbow_to_left_wrist = 0
init_user_max_mpu = 0
m_to_mpu_ratio = 0
init_distance_shoulder2 = 0
init_distance_hip_shoulder2 = 0
init_left_distance_hip_shoulder2 = 0
init_height_diff_right_shoulder_to_right_hip2 = 0
init_head_width2 = 0
init_nose_eye_ear_angle2 = 0
init_right_shoulder_to_right_elbow2 = 0
init_right_elbow_to_right_wrist2 = 0
init_left_shoulder_to_left_elbow2 = 0
init_left_elbow_to_left_wrist2 = 0
init_user_max_mpu2 = 0
init_distance_shoulder3 = 0
init_distance_hip_shoulder3 = 0
init_left_distance_hip_shoulder3 = 0
init_height_diff_right_shoulder_to_right_hip3 = 0
init_head_width3 = 0
init_nose_eye_ear_angle3 = 0
init_right_shoulder_to_right_elbow3 = 0
init_right_elbow_to_right_wrist3 = 0
init_left_shoulder_to_left_elbow3 = 0
init_left_elbow_to_left_wrist3 = 0
init_user_max_mpu3 = 0
init_distance_shoulder_ratio = 0
init_distance_hip_shoulder_ratio = 0
init_left_distance_hip_shoulder_ratio = 0
init_height_diff_right_shoulder_to_right_hip_ratio = 0
init_head_width_ratio = 0
init_nose_eye_ear_angle_ratio = 0
init_right_shoulder_to_right_elbow_ratio = 0
init_right_elbow_to_right_wrist_ratio = 0
init_left_shoulder_to_left_elbow_ratio = 0
init_left_elbow_to_left_wrist_ratio = 0
init_user_max_mpu_ratio = 0
depth_ratio = 0


"""

^^^ Inits ^^^

"""












def create_database():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('datatransfer.db')
    c = conn.cursor()

    # Create table with all the specified fields
    c.execute('''
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        init_distance_shoulder REAL,
        init_distance_hip_shoulder REAL,
        init_left_distance_hip_shoulder REAL,
        init_height_diff_right_shoulder_to_right_hip REAL,
        init_head_width REAL,
        init_nose_eye_ear_angle REAL,
        init_right_shoulder_to_right_elbow REAL,
        init_right_elbow_to_right_wrist REAL,
        init_left_shoulder_to_left_elbow REAL,
        init_left_elbow_to_left_wrist REAL,
        init_user_max_mpu REAL,
        m_to_mpu_ratio REAL,
        init_distance_shoulder2 REAL,
        init_distance_hip_shoulder2 REAL,
        init_left_distance_hip_shoulder2 REAL,
        init_height_diff_right_shoulder_to_right_hip2 REAL,
        init_head_width2 REAL,
        init_nose_eye_ear_angle2 REAL,
        init_right_shoulder_to_right_elbow2 REAL,
        init_right_elbow_to_right_wrist2 REAL,
        init_left_shoulder_to_left_elbow2 REAL,
        init_left_elbow_to_left_wrist2 REAL,
        init_user_max_mpu2 REAL,
        init_distance_shoulder3 REAL,
        init_distance_hip_shoulder3 REAL,
        init_left_distance_hip_shoulder3 REAL,
        init_height_diff_right_shoulder_to_right_hip3 REAL,
        init_head_width3 REAL,
        init_nose_eye_ear_angle3 REAL,
        init_right_shoulder_to_right_elbow3 REAL,
        init_right_elbow_to_right_wrist3 REAL,
        init_left_shoulder_to_left_elbow3 REAL,
        init_left_elbow_to_left_wrist3 REAL,
        init_user_max_mpu3 REAL,
        init_distance_shoulder_ratio REAL,
        init_distance_hip_shoulder_ratio REAL,
        init_left_distance_hip_shoulder_ratio REAL,
        init_height_diff_right_shoulder_to_right_hip_ratio REAL,
        init_head_width_ratio REAL,
        init_nose_eye_ear_angle_ratio REAL,
        init_right_shoulder_to_right_elbow_ratio REAL,
        init_right_elbow_to_right_wrist_ratio REAL,
        init_left_shoulder_to_left_elbow_ratio REAL,
        init_left_elbow_to_left_wrist_ratio REAL,
        init_user_max_mpu_ratio REAL,
        depth_ratio REAL
    )
    ''')
    conn.commit()
    conn.close()

create_database()




















def calculate_distance(landmark1, landmark2):
    """
    Calculate the Euclidean distance between two landmarks.
    """
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def get_distance_right_eye_outer_to_ear():
    """
    Get the distance between the right eye outer and the right ear using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]

        # Calculate the distance
        distance = calculate_distance(right_eye_outer, right_ear)

    return distance


def get_distance_left_eye_outer_to_ear():
    """
    Get the distance between the left eye outer and the left ear using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
        left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]

        # Calculate the distance
        distance = calculate_distance(left_eye_outer, left_ear)

    return distance


def get_distance_right_hip_to_right_shoulder():
    """
    Get the distance between the right hip and the right shoulder using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate the distance
        distance = calculate_distance(right_hip, right_shoulder)

    return distance

def get_distance_left_hip_to_left_shoulder():
    """
    Get the distance between the left hip and the left shoulder using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate the distance
        distance = left_hip.y - left_shoulder.y

    return distance


def get_distance_right_shoulder_to_left_shoulder():
    """
    Get the distance between the right shoulder and the left shoulder using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate the distance
        distance = calculate_distance(right_shoulder, left_shoulder)

    return distance


def get_distance_right_hip_to_left_hip():
    """
    Get the distance between the right hip and the left hip using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        # Calculate the distance
        distance = calculate_distance(right_hip, left_hip)

    return distance


def get_distance_right_shoulder_to_right_elbow():
    """
    Get the distance between the right shoulder and the right elbow using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(right_shoulder, right_elbow)

    return distance


def get_distance_left_shoulder_to_left_elbow():
    """
    Get the distance between the left shoulder and the left elbow using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(left_shoulder, left_elbow)

    return distance


def get_distance_right_elbow_to_right_wrist():
    """
    Get the distance between the right elbow and the right wrist using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    
    if results.pose_landmarks:
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate the distance
        distance = calculate_distance(right_elbow, right_wrist)

    return distance


def get_distance_left_elbow_to_left_wrist():
    """
    Get the distance between the left elbow and the left wrist using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Calculate the distance
        distance = calculate_distance(left_elbow, left_wrist)

    return distance


def get_head_width():
    """
    Get the distance between the right ear and the left ear using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]

        # Calculate the distance
        distance = calculate_distance(right_ear, left_ear)

    return distance


def get_height_diff_right_shoulder_to_right_hip():
    """
    Get the distance between the right hip and the left hip using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate the distance
        distance = right_hip.y - right_shoulder.y

    return distance


def get_distance_fingertip_to_fingertip():
    """
    Get the distance between the left index finger and the right index finger using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        right_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

        # Calculate the distance
        distance = calculate_distance(left_index_finger, right_index_finger)

    return distance

def get_distance_left_fingertip_to_elbow():
    """
    Get the distance between the left index finger and the right index finger using MediaPipe Pose.
    """
    distance = None
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(left_index_finger, left_elbow)

    return distance

def get_distance_right_fingertip_to_elbow():
    """
    Get the distance between the left index finger and the right index finger using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """


    if results.pose_landmarks:
        right_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(right_index_finger, right_elbow)

    return distance


def get_left_hip_x():
    x = 0
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        #left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        #left_hip_z = user_depth
        
        # Calculate the distance
        x = left_hip_x

    return x

def get_left_hip_y():
    y = 0
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        #left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        #left_hip_z = user_depth
        
        # Calculate the distance
        y = left_hip_y

    return y


def get_left_hip_z():
    z = 0
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        #left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        #left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        left_hip_z = user_depth
        
        # Calculate the distance
        z = left_hip_z

    return z

def get_left_hip_x_y_z():
    global left_hip_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        left_hip_z = user_depth
        
        # Calculate the distance
        xyz = [left_hip_x,left_hip_y,left_hip_z]

    return xyz


def get_left_shoulder_x_y_z():
    global left_hip_z, left_shoulder_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    
    #print("inisde left shoulder")


    if results.pose_landmarks:
        if developer_mode:
            print("shoulder")

        left_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * m_to_mpu_ratio
        left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * m_to_mpu_ratio
        left_shoulder_z = calculate_z_angle(left_hip_z, init_left_distance_hip_shoulder, body_pitch)

        # Calculate the distance
        xyz = [left_shoulder_x, left_shoulder_y, left_shoulder_z]
        if developer_mode:
            print(xyz)
    
    return xyz

def get_left_elbow_x_y_z():
    global left_shoulder_z, left_elbow_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    

    #print("inisde left shoulder")


    if results.pose_landmarks:
        if developer_mode:
            print("elbow")

        left_elbow_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * m_to_mpu_ratio
        left_elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * m_to_mpu_ratio
        left_elbow_z = calculate_z(left_shoulder_z, init_left_shoulder_to_left_elbow, init_left_shoulder_to_left_elbow3, get_distance_left_shoulder_to_left_elbow(), 0, body_pitch, hipShoElb)
        
        # Calculate the distance
        xyz = [left_elbow_x, left_elbow_y, left_elbow_z]
        if developer_mode:
            print(xyz)

    return xyz

def get_left_wrist_x_y_z():
    global left_elbow_z, left_wrist_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    #print("inisde left shoulder")


    if results.pose_landmarks:
        if developer_mode:
            print("wrist")

        left_wrist_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * m_to_mpu_ratio
        left_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * m_to_mpu_ratio
        left_wrist_z = calculate_z(left_elbow_z, init_left_elbow_to_left_wrist, init_left_elbow_to_left_wrist3, get_distance_left_elbow_to_left_wrist(), 0, body_pitch, hipShoElb)
        
        # Calculate the distance
        xyz = [left_wrist_x, left_wrist_y, left_wrist_z]
        if developer_mode:
            print(xyz)

    return xyz


# Test the function with an image
# image = cv2.imread("path_to_your_image.jpg")
# distance = get_distance_right_eye_outer_to_ear(image)
# print("Distance:", distance)

"""
def calculate_z(z_init, max_length, max_length3, actual_length, angle, pitch, hipShoElb):
    z = 0
    max_len1 = max_length*m_to_mpu_ratio
    max_len3 = max_length3*m_to_mpu_ratio
    max_len = max_len1 + (depth_ratio*(init_distance_hip_shoulder*abs(pitch/90))) + ((max_len3-max_len1)*abs((90-hipShoElb)/90))
    act_len = actual_length*m_to_mpu_ratio
    act_len_prime = act_len - (depth_ratio*(init_distance_hip_shoulder*abs(pitch/90))) - ((max_len3-max_len1)*abs((90-hipShoElb)/90))

    if angle > 0:
        if act_len >= max_len: 
            act_len = max_len
        if developer_mode:    
            print("z_init: " + str(z_init) + ", max_length: " + str(max_len) + ", actual_length: " + str(act_len) + ", angle: " + str(angle) + ", max mpu: " + str(init_user_max_mpu) + ", z = zinit + " + str(np.sqrt((max_len)**2 - (act_len)**2)) + ", z = " + str(z))
        z = z_init + np.sqrt(abs((max_len)**2 - (act_len)**2))

        return z
    else:
        if act_len >= max_len: 
            act_len = max_len
        z = z_init + np.sqrt((max_len)**2 - (act_len)**2)
        if developer_mode:
            print("z_init: " + str(z_init) + ", max_length1: " + str(max_len1) + ", max_length3: " + str(max_len3) + ", act_max_length: " + str(max_len) + ", actual_length: " + str(act_len) + ", pitch angle: " + str(pitch) + ", shoulder angle: " + str(hipShoElb) + ", max mpu: " + str(init_user_max_mpu) + ", z = zinit + " + str(np.sqrt((max_len)**2 - (act_len)**2)) + ", z = " + str(z))
        return z
"""

def calculate_z(z_init, max_length, max_length3, actual_length, angle, pitch, hipShoElb):
    z = 0






    max_len1 = max_length*m_to_mpu_ratio
    max_len3 = max_length3*m_to_mpu_ratio
    max_len = max_len1
    act_len = actual_length*m_to_mpu_ratio
    act_len_prime = act_len - (depth_ratio*(init_distance_hip_shoulder*abs(pitch/90))) - ((max_len3-max_len1)*abs((90-hipShoElb)/90))

    if angle >= 0:
        if act_len_prime >= max_len: 
            act_len_prime = max_len
        if developer_mode:    
            print("z_init: " + str(z_init) + ", max_length: " + str(max_len) + ", actual_length: " + str(act_len) + ", actual_length_prime: " + str(act_len_prime) + ", angle: " + str(angle) + ", max mpu: " + str(init_user_max_mpu) + ", z = zinit + " + str(np.sqrt((max_len)**2 - (act_len)**2)) + ", z = " + str(z))
        z = z_init + (-depth_ratio*act_len_prime+np.sqrt(-(act_len_prime**2)+(max_len**2)+(depth_ratio**2)*(max_len**2)))/(1+(depth_ratio**2))
        print("\n\n z = z_init + (-k * Lc'' + sqrt(-Lc''2 + Lm2 + k2 * Lm2)) / 1 - k2 \n" + 
              str(z) + " = " + str(z_init) + " + (" + str(-depth_ratio) + " * " + str(act_len_prime) + " + sqrt(" + str(-(act_len_prime**2)) + " + " + str((max_len**2)) + " + " + 
              str((depth_ratio**2)) + " * " + str((max_len**2)) + " )) / 1 - " + str(depth_ratio**2) + "\n\n" +
              "(-k * Lc'' + sqrt(-Lc''2 + Lm2 + k2 * Lm2)) = " + str( (-depth_ratio*act_len_prime+np.sqrt(-(act_len_prime**2)+(max_len**2)+(depth_ratio**2)*(max_len**2)))) +
              "\n 1 - k2 = " + str((1+(depth_ratio**2))))

        return z
    else:
        if act_len >= max_len: 
            act_len = max_len
        z = z_init + np.sqrt((max_len)**2 - (act_len)**2)
        if developer_mode:
            print("z_init: " + str(z_init) + ", max_length1: " + str(max_len1) + ", max_length3: " + str(max_len3) + ", act_max_length: " + str(max_len) + ", actual_length: " + str(act_len) + ", pitch angle: " + str(pitch) + ", shoulder angle: " + str(hipShoElb) + ", max mpu: " + str(init_user_max_mpu) + ", z = zinit + " + str(np.sqrt((max_len)**2 - (act_len)**2)) + ", z = " + str(z))
        return z
    
def calculate_z_angle(z_init, max_length, angle):
    #only used for shoulder

    forward_lean = (angle/90)  
    angle_in_radians = math.asin(forward_lean)
    angle_in_degrees = math.degrees(angle_in_radians)

    z = 0
    if angle > 0: #Backward
        if developer_mode:
            print("z_init: " + str(z_init) + ", max_length: " + str(max_length*m_to_mpu_ratio) + ", angle: " + str(angle/90) + ", z + : " + str(((max_length*m_to_mpu_ratio)*(angle/90))))
        z = z_init + ((max_length*m_to_mpu_ratio)*(angle_in_degrees/90))

        return z
    else: # Forward
        if developer_mode:
            print("z_init: " + str(z_init) + ", max_length: " + str(max_length*m_to_mpu_ratio) + ", angle: " + str(angle/90) + ", z + : " + str(((max_length*m_to_mpu_ratio)*(angle/90))))
        z = z_init + ((max_length*m_to_mpu_ratio)*(angle_in_degrees/90))        
        
        return z


def calculate_direction(distance_right, distance_left):
    """
    Calculate the direction the person is facing based on the distances.
    """
    if distance_right is not None and distance_left is not None and distance_right != 0 and distance_left != 0:
        direction_num = distance_right / distance_left
        direction_facing = "Right" if direction_num <= 1 else "Left"
        return direction_num, direction_facing
    return None, "Unknown"


def calculate_body_yaw(distance_shoulder, distance_hip_shoulder, direction_facing, init_val):
    """
    Calculate the body's yaw based on the distance between the shoulders over the distance between hips and shoulders and the 
    direction facing(left or right) to notate whether the user is turning to the left or right.
    """
    if distance_shoulder is not None and distance_hip_shoulder is not None and distance_hip_shoulder != 0:
        if direction_facing == "Right":
            return round(90-(((distance_shoulder / distance_hip_shoulder)/init_val)*90),4) #init_val = 0.55
        else:
            return round((((distance_shoulder / distance_hip_shoulder)/init_val)*90)-90, 4) #init_val = 0.55
    return 0


def calculate_body_pitch(height_diff_hip_shoulder, eye_ear_angle, init_eye_ear_angle):
    """
    Calculate the body's pitch based on height difference between the right should and the right hip over the width of the head 
    and the direction facing(up or down) to notate whether the user is leaning forward or backward.
    """

    uncertainty_buffer = 10 #10 degrees
    hipShoElb = calculate_left_hip_shoulder_elbow_angle()
    max_height = init_height_diff_right_shoulder_to_right_hip3 + ((init_height_diff_right_shoulder_to_right_hip-init_height_diff_right_shoulder_to_right_hip3)*(hipShoElb)/90)





    #print("height diff: " + str(height_diff_hip_shoulder) + ", max height: " + str(max_height) + ", arms down: " + str(init_height_diff_right_shoulder_to_right_hip3) + ", arms up: " + str(init_height_diff_right_shoulder_to_right_hip))
    if height_diff_hip_shoulder > max_height:
        ratio = 1  
    elif (height_diff_hip_shoulder/max_height) < 0.1:
        ratio = 0.1
    else:
        ratio = (height_diff_hip_shoulder/max_height)  
    angle_in_radians = math.asin(ratio)
    angle_in_degrees = math.degrees(angle_in_radians)

    print("\n\nratio: " + str(ratio) + ", radians: " + str(angle_in_radians) + ", degrees: " + str(angle_in_degrees) + "\n\n")

    if eye_ear_angle <= init_eye_ear_angle:
        if developer_mode:
            print("up")
        
        return_val = round(-(90-(angle_in_degrees)), 4)
        return return_val if return_val < uncertainty_buffer else 0
    else:
        if developer_mode:
            print("down")
        
        return_val = round(90-(angle_in_degrees), 4)
        return return_val if return_val > uncertainty_buffer else 0





def calculate_body_roll():
    """
    Calculate the angle between the line connecting the shoulders and the horizontal line.
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """


    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate the slope and the angle
        delta_x = right_shoulder.x - left_shoulder.x
        delta_y = right_shoulder.y - left_shoulder.y  # Y value decreases upwards in image coordinates
        
        angle_radians = math.atan2(delta_y, delta_x)  # Angle with respect to the horizontal line
        angle_degrees = math.degrees(angle_radians)
        
        # Adjusting the angle to horizontal, 0 degrees means the shoulders are perfectly horizontal
        
        if (angle_degrees > 0):
            shoulder_angle = -(angle_degrees)+180
        else:
            shoulder_angle = -((angle_degrees)+180)
        return shoulder_angle
    return None


def calculate_angle(a,b,c):
    """
    This method is a intermediary function for the rest of the specific angle calculations
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


def calculate_left_arm_angle():
    """
    Basic calculation to find the angle between the left shoulder, left elbow, and left wrist.
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    """
    
    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
 
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        return angle
    return None


def calculate_nose_eyeInR_earR():
    """
    Calculates the angle between the nose, right inner eye, and right ear.
    *Used to calculate whether the user is facing up or down 
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    """


    if results.pose_landmarks:
        # Get landmarks for shoulders
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        angle = calculate_angle(nose, right_eye_inner, right_ear)

        return angle
    return None


def calculate_left_hip_shoulder_elbow_angle():
    """
    Calculates the angle between the nose, right inner eye, and right ear.
    *Used to calculate whether the user is facing up or down 
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    """

    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]

        angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        return angle
    return None


def dot_prod_angle(matrixA, matrixB, matrixC):
    aTimesB = 0
    #vectorA = [matrixB[0] - matrixA[0], matrixB[1] - matrixA[1], matrixB[2] - matrixA[2]]
    #vectorB = [matrixC[0] - matrixB[0], matrixC[1] - matrixB[1], matrixC[2] - matrixB[2]]
    aTimesB = (((matrixB[0]-matrixA[0])*(matrixC[0]-matrixB[0]))+((matrixB[1]-matrixA[1])*(matrixC[1]-matrixB[1]))+((matrixB[2]-matrixA[2])*(matrixC[2]-matrixB[2])))
    aMag = np.sqrt(((matrixB[0]-matrixA[0])**2) + ((matrixB[1]-matrixA[1])**2) + ((matrixB[2]-matrixA[2])**2))
    bMag = np.sqrt(((matrixC[0]-matrixB[0])**2) + ((matrixC[1]-matrixB[1])**2) + ((matrixC[2]-matrixB[2])**2))
    theta = np.arccos(aTimesB/(aMag*bMag))
    
    if developer_mode:
        print(str(theta * (180/np.pi)))
    return theta * (180/np.pi)


def calculate_arm_force(thetaUpper, thetaArm, weightAdded):

    thetaB = 180 - ((b - upperarm * np.cos(thetaUpper))/ (np.sqrt(b**2 + upperarm**2 - 2 * b * upperarm * np.cos(thetaUpper))) )
    leverArmFA = cfg * np.sin(thetaUpper + thetaArm - 90)
    leverArmAdd = forearm * np.sin(thetaUpper + thetaArm - 90)
    leverArmBic = b * np.sin(thetaB)
    if developer_mode:
        print("ThetaB: " + str(thetaB) + ", leverArmFA: " + str(leverArmFA) + "leverArmAdd: " + str(leverArmAdd) + "leverArmBic: " + str(leverArmBic))
    force = abs((weightForearm*9.81 * leverArmFA + weightAdded*9.81 * leverArmAdd) / leverArmBic)
    if developer_mode:
        print("Bicep Force: " + str(force))
    return force



def init_data_update(image):
    """
    This method is called once before the program begins updating calculations so that initial values can be found for the user's specific body ratios
    """
    global init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder, init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle, init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist, init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu, m_to_mpu_ratio, image_rgb, results, landmarks, instruction_image_label, img_instruct_label, main_frame
    
    timesChecked = 0
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark

    # Initialize sums for each measurement
    sum_distance_shoulder = 0
    sum_distance_hip_shoulder = 0
    sum_left_distance_hip_shoulder = 0
    sum_height_diff_right_shoulder_to_right_hip = 0
    sum_head_width = 0
    sum_nose_eye_ear_angle = 0
    sum_right_shoulder_to_right_elbow = 0
    sum_right_elbow_to_right_wrist = 0
    sum_left_shoulder_to_left_elbow = 0
    sum_left_elbow_to_left_wrist = 0
    sum_user_max_mpu = 0


    instruction_image_label.config(image=instruction_image_2_tk)  # Keep a reference, prevent GC
    instruction_image_label.pack(side=tk.TOP, pady=10)
    img_instruct_label.config(text="Please Bring Your Arms Up Like This:")
    img_instruct_label.pack(side=tk.TOP, fill='none', expand=True, padx=10, pady=10)
    





    while timesChecked < 10:
        results = pose.process(image_rgb)
        landmarks = results.pose_landmarks.landmark

        # Accumulate the sums
        sum_distance_shoulder += get_distance_right_shoulder_to_left_shoulder()
        sum_distance_hip_shoulder += get_distance_right_hip_to_right_shoulder()
        sum_left_distance_hip_shoulder += get_distance_left_hip_to_left_shoulder()
        sum_height_diff_right_shoulder_to_right_hip += get_height_diff_right_shoulder_to_right_hip()
        sum_head_width += get_head_width()
        sum_nose_eye_ear_angle += calculate_nose_eyeInR_earR()
        sum_right_shoulder_to_right_elbow += get_distance_right_shoulder_to_right_elbow()
        sum_right_elbow_to_right_wrist += get_distance_right_elbow_to_right_wrist()
        sum_left_shoulder_to_left_elbow += get_distance_left_shoulder_to_left_elbow()
        sum_left_elbow_to_left_wrist += get_distance_left_elbow_to_left_wrist()
        sum_user_max_mpu += get_distance_fingertip_to_fingertip()
        
        timesChecked += 1

    # Calculate averages
    init_distance_shoulder = sum_distance_shoulder / timesChecked
    init_distance_hip_shoulder = sum_distance_hip_shoulder / timesChecked
    init_left_distance_hip_shoulder = sum_left_distance_hip_shoulder / timesChecked
    init_height_diff_right_shoulder_to_right_hip = sum_height_diff_right_shoulder_to_right_hip / timesChecked
    init_head_width = sum_head_width / timesChecked
    init_nose_eye_ear_angle = sum_nose_eye_ear_angle / timesChecked
    init_right_shoulder_to_right_elbow = sum_right_shoulder_to_right_elbow / timesChecked
    init_right_elbow_to_right_wrist = sum_right_elbow_to_right_wrist / timesChecked
    init_left_shoulder_to_left_elbow = sum_left_shoulder_to_left_elbow / timesChecked
    init_left_elbow_to_left_wrist = sum_left_elbow_to_left_wrist / timesChecked
    init_user_max_mpu = sum_user_max_mpu / timesChecked

    m_to_mpu_ratio = user_height / init_user_max_mpu  # cm per mpu

def init2_data_update(image):
    """
    This method is called once before the program begins updating calculations so that initial values can be found for the user's specific body ratios
    """
    global init_distance_shoulder2, init_distance_hip_shoulder2, init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2, init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2, init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2, init_left_elbow_to_left_wrist2, init_user_max_mpu2, m_to_mpu_ratio2, image_rgb, results, landmarks
    timesChecked = 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark

    # Initialize sums for each measurement
    sum_distance_shoulder2 = 0
    sum_distance_hip_shoulder2 = 0
    sum_left_distance_hip_shoulder2 = 0
    sum_height_diff_right_shoulder_to_right_hip2 = 0
    sum_head_width2 = 0
    sum_nose_eye_ear_angle2 = 0
    sum_right_shoulder_to_right_elbow2 = 0
    sum_right_elbow_to_right_wrist2 = 0
    sum_left_shoulder_to_left_elbow2 = 0
    sum_left_elbow_to_left_wrist2 = 0
    sum_user_max_mpu2 = 0

    instruction_image_label.config(image = instruction_image_3_tk)  # Keep a reference, prevent GC
    instruction_image_label.pack(side=tk.TOP, pady=10)
    img_instruct_label.config(text="Please Bring Your Arms Down Like This:")
    img_instruct_label.pack(side=tk.TOP, fill='none', expand=True, padx=10, pady=10)

    while timesChecked < 10:
        results = pose.process(image_rgb)
        landmarks = results.pose_landmarks.landmark

        # Accumulate the sums
        sum_distance_shoulder2 += get_distance_right_shoulder_to_left_shoulder()
        sum_distance_hip_shoulder2 += get_distance_right_hip_to_right_shoulder()
        sum_left_distance_hip_shoulder2 += get_distance_left_hip_to_left_shoulder()
        sum_height_diff_right_shoulder_to_right_hip2 += get_height_diff_right_shoulder_to_right_hip()
        sum_head_width2 += get_head_width()
        sum_nose_eye_ear_angle2 += calculate_nose_eyeInR_earR()
        sum_right_shoulder_to_right_elbow2 += get_distance_right_shoulder_to_right_elbow()
        sum_right_elbow_to_right_wrist2 += get_distance_right_elbow_to_right_wrist()
        sum_left_shoulder_to_left_elbow2 += get_distance_left_shoulder_to_left_elbow()
        sum_left_elbow_to_left_wrist2 += get_distance_left_elbow_to_left_wrist()
        sum_user_max_mpu2 += get_distance_fingertip_to_fingertip()
        
        timesChecked += 1

    # Calculate averages
    init_distance_shoulder2 = sum_distance_shoulder2 / timesChecked
    init_distance_hip_shoulder2 = sum_distance_hip_shoulder2 / timesChecked
    init_left_distance_hip_shoulder2 = sum_left_distance_hip_shoulder2 / timesChecked
    init_height_diff_right_shoulder_to_right_hip2 = sum_height_diff_right_shoulder_to_right_hip2 / timesChecked
    init_head_width2 = sum_head_width2 / timesChecked
    init_nose_eye_ear_angle2 = sum_nose_eye_ear_angle2 / timesChecked
    init_right_shoulder_to_right_elbow2 = sum_right_shoulder_to_right_elbow2 / timesChecked
    init_right_elbow_to_right_wrist2 = sum_right_elbow_to_right_wrist2 / timesChecked
    init_left_shoulder_to_left_elbow2 = sum_left_shoulder_to_left_elbow2 / timesChecked
    init_left_elbow_to_left_wrist2 = sum_left_elbow_to_left_wrist2 / timesChecked
    init_user_max_mpu2 = sum_user_max_mpu2 / timesChecked

    m_to_mpu_ratio2 = user_height / init_user_max_mpu2  # cm per mpu



def init3_data_update(image):
    """
    This method is called once before the program begins updating calculations so that initial values can be found for the user's specific body ratios
    """
    global init_distance_shoulder3, init_distance_hip_shoulder3, init_left_distance_hip_shoulder3, init_height_diff_right_shoulder_to_right_hip3, init_head_width3, init_nose_eye_ear_angle3, init_right_shoulder_to_right_elbow3, init_right_elbow_to_right_wrist3, init_left_shoulder_to_left_elbow3, init_left_elbow_to_left_wrist3, init_user_max_mpu3, m_to_mpu_ratio3, image_rgb, results, landmarks
    timesChecked = 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark

    # Initialize sums for each measurement
    sum_distance_shoulder3 = 0
    sum_distance_hip_shoulder3 = 0
    sum_left_distance_hip_shoulder3 = 0
    sum_height_diff_right_shoulder_to_right_hip3 = 0
    sum_head_width3 = 0
    sum_nose_eye_ear_angle3 = 0
    sum_right_shoulder_to_right_elbow3 = 0
    sum_right_elbow_to_right_wrist3 = 0
    sum_left_shoulder_to_left_elbow3 = 0
    sum_left_elbow_to_left_wrist3 = 0
    sum_user_max_mpu3 = 0

    while timesChecked < 10:
        results = pose.process(image_rgb)
        landmarks = results.pose_landmarks.landmark

        # Accumulate the sums
        sum_distance_shoulder3 += get_distance_right_shoulder_to_left_shoulder()
        sum_distance_hip_shoulder3 += get_distance_right_hip_to_right_shoulder()
        sum_left_distance_hip_shoulder3 += get_distance_left_hip_to_left_shoulder()
        sum_height_diff_right_shoulder_to_right_hip3 += get_height_diff_right_shoulder_to_right_hip()
        sum_head_width3 += get_head_width()
        sum_nose_eye_ear_angle3 += calculate_nose_eyeInR_earR()
        sum_right_shoulder_to_right_elbow3 += get_distance_right_shoulder_to_right_elbow()
        sum_right_elbow_to_right_wrist3 += get_distance_right_elbow_to_right_wrist()
        sum_left_shoulder_to_left_elbow3 += get_distance_left_shoulder_to_left_elbow()
        sum_left_elbow_to_left_wrist3 += get_distance_left_elbow_to_left_wrist()
        sum_user_max_mpu3 += get_distance_fingertip_to_fingertip()
        
        timesChecked += 1

    # Calculate averages
    init_distance_shoulder3 = sum_distance_shoulder3 / timesChecked
    init_distance_hip_shoulder3 = sum_distance_hip_shoulder3 / timesChecked
    init_left_distance_hip_shoulder3 = sum_left_distance_hip_shoulder3 / timesChecked
    init_height_diff_right_shoulder_to_right_hip3 = sum_height_diff_right_shoulder_to_right_hip3 / timesChecked
    init_head_width3 = sum_head_width3 / timesChecked
    init_nose_eye_ear_angle3 = sum_nose_eye_ear_angle3 / timesChecked
    init_right_shoulder_to_right_elbow3 = sum_right_shoulder_to_right_elbow3 / timesChecked
    init_right_elbow_to_right_wrist3 = sum_right_elbow_to_right_wrist3 / timesChecked
    init_left_shoulder_to_left_elbow3 = sum_left_shoulder_to_left_elbow3 / timesChecked
    init_left_elbow_to_left_wrist3 = sum_left_elbow_to_left_wrist3 / timesChecked
    init_user_max_mpu3 = sum_user_max_mpu3 / timesChecked
    m_to_mpu_ratio3 = user_height / init_user_max_mpu3  # cm per mpu




    
def find_depth_ratio():
    global init_distance_shoulder_ratio, init_distance_hip_shoulder_ratio, init_left_distance_hip_shoulder_ratio, init_height_diff_right_shoulder_to_right_hip_ratio, init_head_width_ratio, init_nose_eye_ear_angle_ratio, init_right_shoulder_to_right_elbow_ratio, init_right_elbow_to_right_wrist_ratio, init_left_shoulder_to_left_elbow_ratio, init_left_elbow_to_left_wrist_ratio, init_user_max_mpu_ratio, init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder, init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle, init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist, init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu, m_to_mpu_ratio, init_distance_shoulder2, init_distance_hip_shoulder2, init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2, init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2, init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2, init_left_elbow_to_left_wrist2, init_user_max_mpu2, init_distance_shoulder3, init_distance_hip_shoulder3, init_left_distance_hip_shoulder3, init_height_diff_right_shoulder_to_right_hip3, init_head_width3, init_nose_eye_ear_angle3, init_right_shoulder_to_right_elbow3, init_right_elbow_to_right_wrist3, init_left_shoulder_to_left_elbow3, init_left_elbow_to_left_wrist3, init_user_max_mpu3, depth_ratio


    """
    Method 1 asks the user to step forward 1 foot(rougly 30 centimeters)

    Method 2 asks the user to move arms forward to a guard position where 
      the shoulder and elbow points over lap with the wrists pointing up 
      towards where the shoulders are angled at 90 degrees and the elbows 
      are angled at 90 degrees.
       \/ \/ \/  Example  \/ \/ \/

       Front View          Side View
        *   *                   *
        | O |               O   |
        * _ *               _ __*
        |   |               |
        | _ |               |
        *   *               *
        |   |               |
        |   |               |
        *   *               *
    """

    UseMethod = 2

    if (UseMethod == 1):
        init_distance_shoulder_ratio = (init_distance_shoulder + init_distance_shoulder2) / 30
        init_distance_hip_shoulder_ratio = (init_distance_hip_shoulder + init_distance_hip_shoulder2) / 30
        init_left_distance_hip_shoulder_ratio = (init_left_distance_hip_shoulder + init_left_distance_hip_shoulder2) / 30
        init_height_diff_right_shoulder_to_right_hip_ratio = (init_height_diff_right_shoulder_to_right_hip + init_height_diff_right_shoulder_to_right_hip2) / 30
        init_head_width_ratio = (init_head_width + init_head_width2) / 30
        init_nose_eye_ear_angle_ratio = (init_nose_eye_ear_angle + init_nose_eye_ear_angle2) / 30
        init_right_shoulder_to_right_elbow_ratio = (init_right_shoulder_to_right_elbow + init_right_shoulder_to_right_elbow2) / 30
        init_right_elbow_to_right_wrist_ratio = (init_right_elbow_to_right_wrist + init_right_elbow_to_right_wrist2) / 30
        init_left_shoulder_to_left_elbow_ratio = (init_left_shoulder_to_left_elbow + init_left_shoulder_to_left_elbow2) / 30
        init_left_elbow_to_left_wrist_ratio = (init_left_elbow_to_left_wrist + init_left_elbow_to_left_wrist2) / 30
        init_user_max_mpu_ratio = (init_user_max_mpu + init_user_max_mpu2)   / 30
    else:
        depth_ratio = ((init_right_elbow_to_right_wrist2 + init_left_elbow_to_left_wrist2)/(init_right_elbow_to_right_wrist + init_left_elbow_to_left_wrist))
        print("depth ratio: " + str(depth_ratio))


def data_update(image):
    global user_height, user_depth, left_arm_bicep_force, leftArmAngle
    """
    This method updates all of the input and output data every time its called
    """
    global direction_num, direction_facing, body_yaw, body_roll, body_pitch, test_num, left_hip_x, left_hip_y, left_hip_z, hipShoElb, left_arm_bicep_force, image_rgb, results, landmarks
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    
    distance_right = get_distance_right_eye_outer_to_ear()
    distance_left = get_distance_left_eye_outer_to_ear()
    distance_shoulder = get_distance_right_shoulder_to_left_shoulder()
    distance_hip_shoulder = get_distance_right_hip_to_right_shoulder()
    head_width = get_head_width()
    height_diff_shoulder_hip = get_height_diff_right_shoulder_to_right_hip()
    nose_eye_ear_angle = calculate_nose_eyeInR_earR()
    direction_num, direction_facing = calculate_direction(distance_right, distance_left)
    body_yaw = calculate_body_yaw(distance_shoulder, distance_hip_shoulder, direction_facing, (init_distance_shoulder/init_distance_hip_shoulder))
    body_roll = calculate_body_roll()  # Calculate shoulder angle
    body_pitch = calculate_body_pitch(height_diff_shoulder_hip, nose_eye_ear_angle, init_nose_eye_ear_angle)
    hipShoElb = calculate_left_hip_shoulder_elbow_angle()
    left_hip_x = get_left_hip_x()
    left_hip_y = get_left_hip_y()
    left_hip_z = get_left_hip_z()
    leftShoulderAngle = dot_prod_angle(get_left_elbow_x_y_z(), get_left_shoulder_x_y_z(), get_left_hip_x_y_z())
    leftArmAngle = dot_prod_angle(get_left_wrist_x_y_z(), get_left_elbow_x_y_z(), get_left_shoulder_x_y_z()) 
    print("bicep force input: left shoulder angle: "+str(leftShoulderAngle)+", left arm angle: "+str(leftArmAngle)+", weight added: " + str(weightAdded))
    left_arm_bicep_force = calculate_arm_force(leftShoulderAngle, leftArmAngle, weightAdded)
    test_num = leftArmAngle
    #print("test num updated")


    user_max_mpu = get_distance_fingertip_to_fingertip()
    m_to_mpu_ratio = user_height/user_max_mpu #cm per mpu
    #print("ratio: " + str(m_to_mpu_ratio))


    #(((init_distance_hip_shoulder/init_distance_shoulder))-(((init_distance_hip_shoulder/init_distance_shoulder) * (abs(body_rotation_y-90))/90)))
    

def update_labels():
    """
    This method updates the labels every time it's called
    """
    if twoStepDone:
        if developer_mode:
            direction_facing_label.config(text=f"Direction Facing: {direction_facing}")
            rot_mtx_label.config(text=f"Torso Rotation (Pitch, Yaw, Roll): ({body_pitch}°, {body_yaw if body_yaw else 'N/A'}°, {round(body_roll,4) if body_roll else 'N/A'}°)")
            body_roll_label.config(text=f"Torso Roll: {body_roll:.2f}°" if body_roll is not None else "Torso Roll: N/A")
            body_yaw_label.config(text=f"Torso Yaw: {body_yaw:.2f}°" if body_yaw else "Torso Yaw: N/A")
            body_pitch_label.config(text=f"Torso Pitch: {body_pitch:.2f}°")
        bicep_force_label.config(text=f"Bicep Force: {left_arm_bicep_force if left_arm_bicep_force else 'N/A'}")
        test_num_label.config(text=f"Left Arm Angle: {test_num if test_num else 'N/A'}")
        #test_num_label.config(text=f"Left Hip (X, Y, Z): ({left_hip_x if left_hip_x else 'N/A'}cm, {left_hip_y if left_hip_y else 'N/A'}cm, {left_hip_z if left_hip_z else 'N/A'}cm)")




def draw_guide_overlay_1(frame, results):
    global initial_circle_positions
    radius = 15
    overlay = frame.copy()
    frame_height, frame_width, _ = frame.shape

    if results.pose_landmarks and not initial_circle_positions:
        # Get the necessary landmarks
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_shoulder_x = left_shoulder.x * frame_width
        left_shoulder_y = left_shoulder.y * frame_height
        right_shoulder_x = right_shoulder.x * frame_width
        right_shoulder_y = right_shoulder.y * frame_height
        left_elbow_x = left_shoulder_x - (right_shoulder_x-left_shoulder_x)*0.85
        left_elbow_y = left_shoulder_y
        right_elbow_x = right_shoulder_x + (right_shoulder_x-left_shoulder_x)*0.85
        right_elbow_y = left_shoulder_y
        left_wrist_x = left_shoulder_x - (right_shoulder_x-left_shoulder_x)*1.65
        left_wrist_y = left_shoulder_y
        right_wrist_x = right_shoulder_x + (right_shoulder_x-left_shoulder_x)*1.65
        right_wrist_y = left_shoulder_y




        # Calculate positions for circles
        left_shoulder_pos = np.array([left_shoulder_x, left_shoulder_y])
        right_shoulder_pos = np.array([right_shoulder_x, right_shoulder_y])
        
        avg_shoulder_y = int((left_shoulder_pos[1] + right_shoulder_pos[1]) / 2)

        left_elbow_pos = np.array([left_elbow_x, left_elbow_y])
        right_elbow_pos = np.array([right_elbow_x, right_elbow_y])

        left_wrist_pos = np.array([left_wrist_x, left_wrist_y])
        right_wrist_pos = np.array([right_wrist_x, right_wrist_y])

        # Initialize the circle positions
        initial_circle_positions = {
            'left_shoulder': (int(left_shoulder_pos[0]), avg_shoulder_y),
            'right_shoulder': (int(right_shoulder_pos[0]), avg_shoulder_y),
            'left_elbow': (int(left_elbow_pos[0]), avg_shoulder_y),
            'right_elbow': (int(right_elbow_pos[0]),avg_shoulder_y),
            'left_wrist': (int(left_wrist_pos[0]),avg_shoulder_y),
            'right_wrist': (int(right_wrist_pos[0]),avg_shoulder_y)


        }

    # Draw the circles at the initialized positions
    for pos in initial_circle_positions.values():
        cv2.circle(overlay, pos, radius, (0, 255, 0), 2)

    return overlay

def check_points_in_circles(frame, results):
    global initial_circle_positions
    radius = 15
    points_in_position = {'left_shoulder': False, 'right_shoulder': False,
                          'left_elbow': False, 'right_elbow': False,
                          'left_wrist': False, 'right_wrist': False}

    if results.pose_landmarks:
        frame_height, frame_width, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate positions
        left_shoulder_pos = np.array([left_shoulder.x * frame_width, left_shoulder.y * frame_height])
        right_shoulder_pos = np.array([right_shoulder.x * frame_width, right_shoulder.y * frame_height])
        left_elbow_pos = np.array([left_elbow.x * frame_width, left_elbow.y * frame_height])
        right_elbow_pos = np.array([right_elbow.x * frame_width, right_elbow.y * frame_height])
        left_wrist_pos = np.array([left_wrist.x * frame_width, left_wrist.y * frame_height])
        right_wrist_pos = np.array([right_wrist.x * frame_width, right_wrist.y * frame_height])

        # Check if points are in the circles
        for label, pos, circle_pos in zip(['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
                                          [left_shoulder_pos, right_shoulder_pos, left_elbow_pos, right_elbow_pos, left_wrist_pos, right_wrist_pos],
                                          initial_circle_positions.values()):
            if np.linalg.norm(pos - np.array(circle_pos)) <= radius:
                points_in_position[label] = True
            else:
                points_in_position[label] = False

    return points_in_position










def draw_guide_overlay_2(frame, results):
    global initial_circle_positions
    radius = 15
    overlay = frame.copy()
    frame_height, frame_width, _ = frame.shape

    if results.pose_landmarks and not initial_circle_positions:
        # Get the necessary landmarks
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_shoulder_x = left_shoulder.x * frame_width
        left_shoulder_y = left_shoulder.y * frame_height
        right_shoulder_x = right_shoulder.x * frame_width
        right_shoulder_y = right_shoulder.y * frame_height
        left_elbow_x = left_shoulder_x - (right_shoulder_x-left_shoulder_x)*0.2
        left_elbow_y = left_shoulder_y
        right_elbow_x = right_shoulder_x + (right_shoulder_x-left_shoulder_x)*0.2
        right_elbow_y = left_shoulder_y
        left_wrist_x = left_shoulder_x - (right_shoulder_x-left_shoulder_x)*0.2
        left_wrist_y = left_shoulder_y + (right_shoulder_x-left_shoulder_x)*0.8
        right_wrist_x = right_shoulder_x + (right_shoulder_x-left_shoulder_x)*0.2
        right_wrist_y = left_shoulder_y + (right_shoulder_x-left_shoulder_x)*0.8




        # Calculate positions for circles
        left_shoulder_pos = np.array([left_shoulder_x, left_shoulder_y])
        right_shoulder_pos = np.array([right_shoulder_x, right_shoulder_y])
        
        avg_shoulder_y = int((left_shoulder_pos[1] + right_shoulder_pos[1]) / 2)

        left_elbow_pos = np.array([left_elbow_x, left_elbow_y])
        right_elbow_pos = np.array([right_elbow_x, right_elbow_y])

        left_wrist_pos = np.array([left_wrist_x, left_wrist_y])
        right_wrist_pos = np.array([right_wrist_x, right_wrist_y])

        # Initialize the circle positions
        initial_circle_positions = {
            'left_shoulder': (int(left_shoulder_pos[0]), avg_shoulder_y),
            'right_shoulder': (int(right_shoulder_pos[0]), avg_shoulder_y),
            'left_elbow': (int(left_elbow_pos[0]), int(left_elbow_pos[1])),
            'right_elbow': (int(right_elbow_pos[0]),int(right_elbow_pos[1])),
            'left_wrist': (int(left_wrist_pos[0]),int(left_wrist_pos[1])),
            'right_wrist': (int(right_wrist_pos[0]),int(right_wrist_pos[1]))


        }

    # Draw the circles at the initialized positions
    for pos in initial_circle_positions.values():
        cv2.circle(overlay, pos, radius, (0, 255, 0), 2)

    return overlay



def draw_guide_overlay_3(frame, results):
    global initial_circle_positions
    radius = 15
    overlay = frame.copy()
    frame_height, frame_width, _ = frame.shape

    if results.pose_landmarks and not initial_circle_positions:
        # Get the necessary landmarks
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_shoulder_x = left_shoulder.x * frame_width
        left_shoulder_y = left_shoulder.y * frame_height
        right_shoulder_x = right_shoulder.x * frame_width
        right_shoulder_y = right_shoulder.y * frame_height
        left_elbow_x = left_shoulder_x - (right_shoulder_x-left_shoulder_x)*0.2
        left_elbow_y = left_shoulder_y - (right_shoulder_x-left_shoulder_x)*0.9
        right_elbow_x = right_shoulder_x + (right_shoulder_x-left_shoulder_x)*0.15
        right_elbow_y = left_shoulder_y - (right_shoulder_x-left_shoulder_x)*0.9
        left_wrist_x = left_shoulder_x - (right_shoulder_x-left_shoulder_x)*0.15
        left_wrist_y = left_shoulder_y - (right_shoulder_x-left_shoulder_x)*1.75
        right_wrist_x = right_shoulder_x + (right_shoulder_x-left_shoulder_x)*0.15
        right_wrist_y = left_shoulder_y - (right_shoulder_x-left_shoulder_x)*1.75




        # Calculate positions for circles
        left_shoulder_pos = np.array([left_shoulder_x, left_shoulder_y])
        right_shoulder_pos = np.array([right_shoulder_x, right_shoulder_y])
        
        avg_shoulder_y = int((left_shoulder_pos[1] + right_shoulder_pos[1]) / 2)

        left_elbow_pos = np.array([left_elbow_x, left_elbow_y])
        right_elbow_pos = np.array([right_elbow_x, right_elbow_y])

        left_wrist_pos = np.array([left_wrist_x, left_wrist_y])
        right_wrist_pos = np.array([right_wrist_x, right_wrist_y])

        # Initialize the circle positions
        initial_circle_positions = {
            'left_shoulder': (int(left_shoulder_pos[0]), avg_shoulder_y),
            'right_shoulder': (int(right_shoulder_pos[0]), avg_shoulder_y),
            'left_elbow': (int(left_elbow_pos[0]), int(left_elbow_pos[1])),
            'right_elbow': (int(right_elbow_pos[0]),int(right_elbow_pos[1])),
            'left_wrist': (int(left_wrist_pos[0]),int(left_wrist_pos[1])),
            'right_wrist': (int(right_wrist_pos[0]),int(right_wrist_pos[1]))


        }

    # Draw the circles at the initialized positions
    for pos in initial_circle_positions.values():
        cv2.circle(overlay, pos, radius, (0, 255, 0), 2)

    return overlay


def format_point_name(point):
    """
    Convert point name from 'left_shoulder' to 'Left Shoulder'.
    """
    return ' '.join(word.capitalize() for word in point.split('_'))





def data_collection():
    global init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder, init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle, init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist, init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu, m_to_mpu_ratio, init_distance_shoulder2, init_distance_hip_shoulder2, init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2, init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2, init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2, init_left_elbow_to_left_wrist2, init_user_max_mpu2, init_distance_shoulder3, init_distance_hip_shoulder3, init_left_distance_hip_shoulder3, init_height_diff_right_shoulder_to_right_hip3, init_head_width3, init_nose_eye_ear_angle3, init_right_shoulder_to_right_elbow3, init_right_elbow_to_right_wrist3, init_left_shoulder_to_left_elbow3, init_left_elbow_to_left_wrist3, init_user_max_mpu3, init_distance_shoulder_ratio, init_distance_hip_shoulder_ratio, init_left_distance_hip_shoulder_ratio, init_height_diff_right_shoulder_to_right_hip_ratio, init_head_width_ratio, init_nose_eye_ear_angle_ratio, init_right_shoulder_to_right_elbow_ratio, init_right_elbow_to_right_wrist_ratio, init_left_shoulder_to_left_elbow_ratio, init_left_elbow_to_left_wrist_ratio, init_user_max_mpu_ratio, depth_ratio
    
    # Prepare data for writing
    data = (
        init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder, 
        init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle, 
        init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist, 
        init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu, 
        m_to_mpu_ratio, init_distance_shoulder2, init_distance_hip_shoulder2, 
        init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2, 
        init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2, 
        init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2, 
        init_left_elbow_to_left_wrist2, init_user_max_mpu2, init_distance_shoulder3, 
        init_distance_hip_shoulder3, init_left_distance_hip_shoulder3, 
        init_height_diff_right_shoulder_to_right_hip3, init_head_width3, 
        init_nose_eye_ear_angle3, init_right_shoulder_to_right_elbow3, 
        init_right_elbow_to_right_wrist3, init_left_shoulder_to_left_elbow3, 
        init_left_elbow_to_left_wrist3, init_user_max_mpu3, init_distance_shoulder_ratio, 
        init_distance_hip_shoulder_ratio, init_left_distance_hip_shoulder_ratio, 
        init_height_diff_right_shoulder_to_right_hip_ratio, init_head_width_ratio, 
        init_nose_eye_ear_angle_ratio, init_right_shoulder_to_right_elbow_ratio, 
        init_right_elbow_to_right_wrist_ratio, init_left_shoulder_to_left_elbow_ratio, 
        init_left_elbow_to_left_wrist_ratio, init_user_max_mpu_ratio, depth_ratio
    )

    # Convert data tuple to a comma-separated string
    data_str = ",".join(map(str, data))

    # Write to file
    with open("measurements_data.txt", "w") as file:
        file.write(data_str)

"""

def data_collection():
    global init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder, init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle, init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist, init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu, m_to_mpu_ratio, image_rgb, results, landmarks, instruction_image_label, img_instruct_label, main_frame, init_distance_shoulder2, init_distance_hip_shoulder2, init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2, init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2, init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2, init_left_elbow_to_left_wrist2, init_user_max_mpu2, init_distance_shoulder3, init_distance_hip_shoulder3, init_left_distance_hip_shoulder3, init_height_diff_right_shoulder_to_right_hip3, init_head_width3, init_nose_eye_ear_angle3, init_right_shoulder_to_right_elbow3, init_right_elbow_to_right_wrist3, init_left_shoulder_to_left_elbow3, init_left_elbow_to_left_wrist3, init_user_max_mpu3, init_distance_shoulder_ratio, init_distance_hip_shoulder_ratio, init_left_distance_hip_shoulder_ratio, init_height_diff_right_shoulder_to_right_hip_ratio, init_head_width_ratio, init_nose_eye_ear_angle_ratio, init_right_shoulder_to_right_elbow_ratio, init_right_elbow_to_right_wrist_ratio, init_left_shoulder_to_left_elbow_ratio, init_left_elbow_to_left_wrist_ratio, init_user_max_mpu_ratio, depth_ratio
    # Connect to the SQLite database
    conn = sqlite3.connect('datatransfer.db')
    c = conn.cursor()
    
    # Prepare data for insertion
    data = (
        init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder, 
        init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle, 
        init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist, 
        init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu, 
        m_to_mpu_ratio, init_distance_shoulder2, init_distance_hip_shoulder2, 
        init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2, 
        init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2, 
        init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2, 
        init_left_elbow_to_left_wrist2, init_user_max_mpu2, init_distance_shoulder3, 
        init_distance_hip_shoulder3, init_left_distance_hip_shoulder3, 
        init_height_diff_right_shoulder_to_right_hip3, init_head_width3, 
        init_nose_eye_ear_angle3, init_right_shoulder_to_right_elbow3, 
        init_right_elbow_to_right_wrist3, init_left_shoulder_to_left_elbow3, 
        init_left_elbow_to_left_wrist3, init_user_max_mpu3, init_distance_shoulder_ratio, 
        init_distance_hip_shoulder_ratio, init_left_distance_hip_shoulder_ratio, 
        init_height_diff_right_shoulder_to_right_hip_ratio, init_head_width_ratio, 
        init_nose_eye_ear_angle_ratio, init_right_shoulder_to_right_elbow_ratio, 
        init_right_elbow_to_right_wrist_ratio, init_left_shoulder_to_left_elbow_ratio, 
        init_left_elbow_to_left_wrist_ratio, init_user_max_mpu_ratio, depth_ratio
    )

    # SQL query for inserting data
    query = '''
    INSERT INTO measurements (
        init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder,
        init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle,
        init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist,
        init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu,
        m_to_mpu_ratio, init_distance_shoulder2, init_distance_hip_shoulder2,
        init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2,
        init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2,
        init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2,
        init_left_elbow_to_left_wrist2, init_user_max_mpu2, init_distance_shoulder3,
        init_distance_hip_shoulder3, init_left_distance_hip_shoulder3,
        init_height_diff_right_shoulder_to_right_hip3, init_head_width3,
        init_nose_eye_ear_angle3, init_right_shoulder_to_right_elbow3,
        init_right_elbow_to_right_wrist3, init_left_shoulder_to_left_elbow3,
        init_left_elbow_to_left_wrist3, init_user_max_mpu3, init_distance_shoulder_ratio,
        init_distance_hip_shoulder_ratio, init_left_distance_hip_shoulder_ratio,
        init_height_diff_right_shoulder_to_right_hip_ratio, init_head_width_ratio,
        init_nose_eye_ear_angle_ratio, init_right_shoulder_to_right_elbow_ratio,
        init_right_elbow_to_right_wrist_ratio, init_left_shoulder_to_left_elbow_ratio,
        init_left_elbow_to_left_wrist_ratio, init_user_max_mpu_ratio, depth_ratio
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''



    c.execute(query, data)
    conn.commit()
    conn.close()

# Ensure that all variables used in data_collection are defined before calling the function.

"""





def update_image():
    global last_update_time, twoStepDone, current_stage, initial_circle_positions, img_instruct_label, instruction_image_label, start_time




    ret, frame = cap.read()
    if not ret:
        return  # Exit if video capture has failed

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)





    """
    cap.release()
    root.destroy()
    cv2.destroyAllWindows()
    time.sleep(10)
    simgui1 = gui.SimGUI()
    simgui1.start()
    """

    #data_collection()
    if current_stage == 'overlay_1':
        overlay = draw_guide_overlay_1(frame, results)
        points_in_position = check_points_in_circles(frame, results)

        # Clear the existing text
        vid_instruct_text.config(state=tk.NORMAL)
        vid_instruct_text.delete('1.0', tk.END)

         # Insert new status text with colors
        for point, status in points_in_position.items():
            formatted_point = format_point_name(point)
            color = 'green' if status else 'purple' if 'Wrist' in formatted_point else 'red' if 'Elbow' in formatted_point else 'blue'
            vid_instruct_text.insert(tk.END, f"{formatted_point} is in Position: {status}\n", color)

        vid_instruct_text.config(state=tk.DISABLED)



        if all(points_in_position.values()):
            init_data_update(image)
            initial_circle_positions = {}  # Reset for the next overlay
            current_stage = 'overlay_2'
            
            

    elif current_stage == 'overlay_2':
        overlay = draw_guide_overlay_2(frame, results)
        points_in_position = check_points_in_circles(frame, results)
        
        
        # Clear the existing text
        vid_instruct_text.config(state=tk.NORMAL)
        vid_instruct_text.delete('1.0', tk.END)

         # Insert new status text with colors
        for point, status in points_in_position.items():
            formatted_point = format_point_name(point)
            color = 'green' if status else 'purple' if 'Wrist' in formatted_point else 'red' if 'Elbow' in formatted_point else 'blue'
            vid_instruct_text.insert(tk.END, f"{formatted_point} is in Position: {status}\n", color)

        vid_instruct_text.config(state=tk.DISABLED)






        if all(points_in_position.values()):
            init2_data_update(image)
            find_depth_ratio()
            initial_circle_positions = {}  # Reset for the next overlay
            current_stage = 'overlay_3'
            

    elif current_stage == 'overlay_3':
        overlay = draw_guide_overlay_3(frame, results)
        points_in_position = check_points_in_circles(frame, results)

        
        # Clear the existing text
        vid_instruct_text.config(state=tk.NORMAL)
        vid_instruct_text.delete('1.0', tk.END)

         # Insert new status text with colors
        for point, status in points_in_position.items():
            formatted_point = format_point_name(point)
            color = 'green' if status else 'purple' if 'Wrist' in formatted_point else 'red' if 'Elbow' in formatted_point else 'blue'
            vid_instruct_text.insert(tk.END, f"{formatted_point} is in Position: {status}\n", color)

        vid_instruct_text.config(state=tk.DISABLED)




        if all(points_in_position.values()):
            init3_data_update(image)
            instruct_label.config(text="Simulation Started")
            vid_instruct_text.config(state=tk.NORMAL)
            vid_instruct_text.delete('1.0', tk.END)
            vid_instruct_text.insert(tk.END, "Simulation in Progress", 'black')
            vid_instruct_text.config(state=tk.DISABLED,height=1)
            print("im being ran&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            twoStepDone = True
            packTwoSteps()
            current_stage = 'data_update'

    elif current_stage == 'data_update' and twoStepDone:
        overlay = None  # Clear overlay for the data update phase
        data_collection()


        if start_time == 0:
            # Clear the existing text

            start_time = time.time()

        # Calculate the elapsed time since the start
        elapsed_time = time.time() - start_time



        if True: #elapsed_time > time_simulation_active:
            print(str(time_simulation_active) + " seconds have elapsed, stopping the update.")
            cap.release()
            root.destroy()
            cv2.destroyAllWindows()
            time.sleep(10)
            gui.SimGUI()
            gui.start()
            return # Exit the function to stop the loop



    if overlay is not None:
        image = cv2.addWeighted(overlay, 0.6, frame, 1 - 0.6, 0)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    image_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    video_label.config(image=image_tk)
    video_label.image = image_tk
    if twoStepDone:
        if isGraphOn:
            plot_graph()

    


    root.after(10, update_image)










"""
Start up sequence
\/ \/ \/
"""

# Main window setup
main_window = tk.Tk()
main_window.title("Muscle Force Calculator")
main_window.geometry("450x350")



def show_data_input():
    # Hide the start frame and show the data input frame
    start_frame.pack_forget()
    data_input_frame.pack(fill='both', expand=True)

def show_settings():
    # Hide the data input frame and show the settings frame
    data_input_frame.pack_forget()
    settings_frame.pack(fill='both', expand=True)

# Countdown frame setup
countdown_frame = ttk.Frame(main_window)
countdown_label = ttk.Label(countdown_frame, text="Starting in 0 seconds", font=("Helvetica", 16))
countdown_label.pack(padx=20, pady=20)


def on_settings_submit():
    global time_to_get_in_position, time_simulation_active

    time_to_get_in_position = int(time_to_get_position_var.get())
    time_simulation_active = int(time_simulation_var.get())
    print(f"Time to Get in Position: {time_to_get_in_position} seconds, Time Simulation Active: {time_simulation_active} seconds")

    # Hide the current settings frame and show the countdown frame
    settings_frame.pack_forget()
    countdown_frame.pack(fill='both', expand=True)
    
    # Start the countdown
    countdown(time_to_get_in_position)

def countdown(time_left):
    if time_left > 0:
        countdown_label.config(text=f"Starting in {time_left} seconds")
        countdown_frame.after(1000, countdown, time_left - 1)
    else:
        countdown_label.config(text="Starting now!")
        main_window.after(1000, main_window.destroy)  # Close the window after 1 second


# Default value for time_to_get_in_position
# Add a new frame for additional settings (initially hidden)
settings_frame = ttk.Frame(main_window)

# Time to Get in Position Dropdown
time_to_get_position_var = tk.StringVar()
time_to_get_position_label = ttk.Label(settings_frame, text="Time To Get In Position(sec):")
time_to_get_position_label.pack(padx=10, pady=5)
time_to_get_position_dropdown = ttk.Combobox(settings_frame, textvariable=time_to_get_position_var, 
                                             values=[5, 10, 20, 30, 60])
time_to_get_position_dropdown.pack(padx=10, pady=5)

# Time Simulation Active Dropdown
time_simulation_var = tk.StringVar()
time_simulation_label = ttk.Label(settings_frame, text="Time Simulation Active(sec):")
time_simulation_label.pack(padx=10, pady=5)
time_simulation_dropdown = ttk.Combobox(settings_frame, textvariable=time_simulation_var,
                                        values=[30, 60, 120, 300, 1200])
time_simulation_dropdown.pack(padx=10, pady=5)

# Settings Submit Button
settings_submit_button = ttk.Button(settings_frame, text="Submit", command=on_settings_submit)
settings_submit_button.pack(padx=10, pady=15)


# Modify the on_submit function to show the settings frame
def on_submit():
    try:
        global user_weight, user_height, user_depth, weightAdded, developer_mode, isGraphOn, forearm, upperarm, cfg, b, weightForearm
        user_weight = float(weight_entry.get())
        user_height = float(height_entry.get()) * 0.9588 #Mediapipe measures from kunckle to kunckle not fingertip to fingertip
        user_depth = float(depth_entry.get())
        weightAdded = float(weight_holding_entry.get())
        developer_mode = dev_mode_var.get() == 1
        isGraphOn = graph_on_var.get() == 1
        print(f"User Weight: {user_weight} kg, User Height: {user_height} cm, User Depth: {user_depth} cm, Weight Holding: {weightAdded} kg")
        show_settings()  # Show the settings frame instead of destroying the window

        #Updating initial values
        forearm = (user_height*0.01) * 0.216
        upperarm = (user_height*0.01) * 0.173
        cfg = forearm * 0.432
        b = forearm * 0.11
        weightForearm = user_weight * 0.023

    except ValueError:
        print("Please enter valid numbers for weight, height, and depth.")

# Start frame
start_frame = ttk.Frame(main_window)
start_frame.pack(fill='both', expand=True)

title_label = ttk.Label(start_frame, text="Muscle Force Calculator", font=("Helvetica", 16))
title_label.pack(padx=20, pady=20)

start_button = ttk.Button(start_frame, text="Start", command=show_data_input)
start_button.pack(side=tk.BOTTOM, padx=10, pady=10)


# Data input frame setup
data_input_frame = ttk.Frame(main_window)

# Left column for height and depth
left_column = ttk.Frame(data_input_frame)
left_column.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)

ttk.Label(left_column, text="Enter Your Height (cm):").pack(padx=10, pady=5)
height_entry = ttk.Entry(left_column)
height_entry.pack(padx=10, pady=5)

ttk.Label(left_column, text="Enter Your Distance from Camera (cm):").pack(padx=10, pady=5)
depth_entry = ttk.Entry(left_column)
depth_entry.pack(padx=10, pady=5)

# Right column for user weight, weight holding, and developer mode
right_column = ttk.Frame(data_input_frame)
right_column.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)

ttk.Label(right_column, text="Enter Your Weight (kg):").pack(padx=10, pady=5)
weight_entry = ttk.Entry(right_column)
weight_entry.pack(padx=10, pady=5)

ttk.Label(right_column, text="Enter The Weight You're Holding (kg):").pack(padx=10, pady=5)
weight_holding_entry = ttk.Entry(right_column)
weight_holding_entry.pack(padx=10, pady=5)

dev_mode_var = tk.IntVar()
dev_mode_check = ttk.Checkbutton(right_column, text="Developer Mode", variable=dev_mode_var)
dev_mode_check.pack(padx=10, pady=5)

graph_on_var = tk.IntVar()
graph_on_check = ttk.Checkbutton(right_column, text="Enable Graph", variable=graph_on_var)
graph_on_check.pack(padx=10, pady=5)

submit_button = ttk.Button(right_column, text="Next", command=on_submit)
submit_button.pack(side=tk.BOTTOM, padx=10, pady=10)

if BypassStartUp is True:
    main_window.destroy()


# Start the Tkinter main loop
main_window.mainloop()




"""
/\ /\ /\ 
Start Up Sequence
"""











root = tk.Tk()


root.title("Pose Detection with Data Output")

# Create the main frame
main_frame = ttk.Frame(root)
main_frame.pack(padx=10, pady=0, fill='both', expand=True)

instruct_label = ttk.Label(main_frame, text="Please Align Your Body to Fit In The Circles", font=("Helvetica", 20))
instruct_label.pack(side=tk.TOP, fill='y', expand=True, padx=10, pady=10)

video_frame = ttk.LabelFrame(main_frame, text="Video Output")
video_frame.pack(side=tk.LEFT, fill='both', expand=False, padx=20, pady=10)  # Apply padx and pady here
video_frame.pack_propagate(True)  # Prevent the frame from resizing to fit its content
video_frame.config(width=600, height=600)  # Set the width and height of the frame
"""
vid_instruct_label = ttk.Label(video_frame, text="Please Align Your Body to Fit In The Circles", font=("Helvetica", 16))
vid_instruct_label.pack(side=tk.TOP, fill='both', expand=True, padx=10, pady=10)
"""
vid_instruct_text = tk.Text(video_frame, width=40, height=10, font=("Helvetica", 14))
vid_instruct_text.pack(side=tk.TOP, fill='both', expand=True, padx=10, pady=0)
vid_instruct_text.config(state=tk.DISABLED, height=6)  # Make it read-only


vid_instruct_text.tag_configure('blue', foreground='blue')
vid_instruct_text.tag_configure('red', foreground='red')
vid_instruct_text.tag_configure('purple', foreground='purple')
vid_instruct_text.tag_configure('green', foreground='green')
vid_instruct_text.tag_configure('black', foreground='black')





# Create a label in the main frame for video feed
video_label = ttk.Label(video_frame)
video_label.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)


instruction_frame = ttk.LabelFrame(main_frame, text="Instruction Frame")
instruction_frame.pack(side=tk.RIGHT, fill='x', expand=False, padx=20, pady=10)  # Apply padx and pady here
instruction_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its content
instruction_frame.config(width=650, height=600)  # Set the width and height of the frame




img_instruct_label = ttk.Label(instruction_frame, text="Please Spread Your Arms Out Like This:", font=("Helvetica", 24))
img_instruct_label.pack(side=tk.TOP, fill='none', expand=True, padx=10, pady=10)


# Load the image
instruction_image_1 = Image.open(instruction_image_1_path)

# Resize the image if necessary
instruction_image_1 = instruction_image_1.resize((615, 400), Image.Resampling.LANCZOS)

# Convert the image to a format suitable for Tkinter
instruction_image_1_tk = ImageTk.PhotoImage(instruction_image_1)

# Load the image
instruction_image_2 = Image.open(instruction_image_2_path)

# Resize the image if necessary
instruction_image_2 = instruction_image_2.resize((615, 400), Image.Resampling.LANCZOS)

# Convert the image to a format suitable for Tkinter
instruction_image_2_tk = ImageTk.PhotoImage(instruction_image_2)

# Load the image
instruction_image_3 = Image.open(instruction_image_3_path)

# Resize the image if necessary
instruction_image_3 = instruction_image_3.resize((615, 400), Image.Resampling.LANCZOS)

# Convert the image to a format suitable for Tkinter
instruction_image_3_tk = ImageTk.PhotoImage(instruction_image_3)


# Create a label in the instruction_frame to display the image
instruction_image_label = ttk.Label(instruction_frame, image=instruction_image_1_tk)
instruction_image_label.image = instruction_image_1_tk  # Keep a reference, prevent GC
instruction_image_label.pack(side=tk.TOP, pady=10)

"""
# Create a label in the instruction_frame to display the image
instruction_image_label = ttk.Label(instruction_frame, image=instruction_image_2_tk)
instruction_image_label.image = instruction_image_2_tk  # Keep a reference, prevent GC
instruction_image_label.pack(side=tk.TOP, pady=10)
"""

"""
# Create a label in the instruction_frame to display the image
instruction_image_label = ttk.Label(instruction_frame, image=instruction_image_2_tk)
instruction_image_label.image = instruction_image_2_tk  # Keep a reference, prevent GC
instruction_image_label.pack(side=tk.TOP, pady=10)
"""





def packTwoSteps():
    if developer_mode:
        print("&&&&   PHASE CHANGE : INIT -> RUNNING   &&&&")
    instruction_frame.destroy()
    data_frame.pack(side=tk.RIGHT, fill='both', expand=False, padx=20, pady=10)  # Apply padx and pady here
    data_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its content
    data_frame.config(width=400, height=600)  # Set the width and height of the frame
    if isGraphOn:
        graph_frame.pack(side=tk.BOTTOM, fill='both', expand=False, padx=10, pady=10)
        graph_frame.config(width=400, height=300)  # Set the width and height of the frame
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    if developer_mode:
        direction_facing_label.pack(anchor=tk.W)
        rot_mtx_label.pack(anchor=tk.W)
        body_pitch_label.pack(anchor=tk.W)
        body_roll_label.pack(anchor=tk.W)
        body_yaw_label.pack(anchor=tk.W)
    bicep_force_label.pack(anchor=tk.W)
    test_num_label.pack(anchor=tk.W)
    user_height_frame.pack(fill='x', expand=True, pady=5)





# Create a frame for data output
data_frame = ttk.LabelFrame(main_frame, text="Data Output")


graph_frame = ttk.Frame(data_frame)




# Create a matplotlib figure and a canvas
fig = plt.Figure(figsize=(5, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=graph_frame)

# Initialize the plot data
I_values = np.arange(0, 180)  # Elbow Angle values from -90 to 90
current_index = 0  # Start with the first point
plotted_points = []  # List to store the points that have been plotted

plotted_points = []  # Initialize outside of the plot_graph function to store all points

def plot_graph():
    global leftArmAngle, left_arm_bicep_force

    # Check if there's new data to plot and add it to the list
    if leftArmAngle != 0 and left_arm_bicep_force != 0:
        plotted_points.append((leftArmAngle, left_arm_bicep_force))

    # Clear the previous figure
    fig.clear()

    # Create a new plot
    ax = fig.add_subplot(111)

    # Plot all points in the list
    for point in plotted_points:
        ax.scatter(*point, color='blue')

    ax.set_title("Elbow Angle vs Bicep Force")
    ax.set_xlabel("Elbow Angle")
    ax.set_ylabel("Bicep Force")
    ax.set_xlim(0, 180)  # Set the x-axis limits based on your data
    ax.set_ylim(0, max([8100] + [force for _, force in plotted_points]))  # Set the y-axis limit based on the maximum force

    # Draw the updated graph
    canvas.draw()

    # Reset the current data point
    leftArmAngle = 0
    left_arm_bicep_force = 0
    # Plot the graph as soon as the program runs





# Labels for displaying data
if developer_mode:





    direction_facing_label = ttk.Label(data_frame, text="Direction Facing: N/A")

    rot_mtx_label = ttk.Label(data_frame, text="Rotation Matrix: (x,y,z)")

    body_pitch_label = ttk.Label(data_frame, text="Torso Pitch: N/A")

    body_roll_label = ttk.Label(data_frame, text="Body Rotation (Z-Axis): N/A")

    body_yaw_label = ttk.Label(data_frame, text="Body Yaw: N/A")

    





bicep_force_label = ttk.Label(data_frame, text="Force: N/A")

test_num_label = ttk.Label(data_frame, text="Test Num: N/A")


# Add User Height Input UI Elements
user_height_frame = ttk.Frame(main_frame)





"""
def on_confirm_height():
    global user_height, user_depth
    try:
        user_height = float(user_height_entry.get())
        user_depth = float(user_depth_entry.get())
        print(f"User Height: {user_height} cm")
        # You can now use `user_height` for further calculations or display
    except ValueError:
        print("Please enter a valid number for height.")

confirm_height_button = ttk.Button(user_height_frame, text="Confirm", command=on_confirm_height)
confirm_height_button.pack(side=tk.BOTTOM, padx=5, pady=15)

user_height_label = ttk.Label(user_height_frame, text="Enter User Height (cm):")
user_height_label.pack(side=tk.BOTTOM, padx=5, pady=5)

user_height_entry = ttk.Entry(user_height_frame)
user_height_entry.pack(side=tk.BOTTOM, padx=5, pady=5)

user_depth_label = ttk.Label(user_height_frame, text="Enter User Depth (cm):")
user_depth_label.pack(side=tk.BOTTOM, padx=5, pady=5)

user_depth_entry = ttk.Entry(user_height_frame)
user_depth_entry.pack(side=tk.BOTTOM, padx=5, pady=5)
"""


# Open the webcam
cap = cv2.VideoCapture(0)

# Start the periodic update of the image and data
update_image()


# Start the Tkinter main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()