# BD 2023-24
# This program is designed to reimplement code from a previous program for use in a new environment
# in order to extrapolate 3D motion tracking data from 2D motion tracking data and user input.
# This version has been edited for use directly with MediaPipe, as opposed to with FreeMoCap data output.


# NOTES:
#   - X is left/right, Z is up/down, Y is depth


# TODO: 

# IDEAS:
#   - during initial calibration, have user make all points in body planar with the camera, and check differences in ratios, use as offsets for rest of program


import numpy as np
import math
from matplotlib import pyplot as plt

import threading


# set to not display in scientific notation
np.set_printoptions(suppress = True, precision = 5)

#### CONSTANTS (for use with indexing)
## INDEXING FOR VERTEX ARRAYS
L_SHOULDER = 0#11
R_SHOULDER = 1#12
L_ELBOW = 2#13
R_ELBOW = 3#14
L_WRIST = 4#15
R_WRIST = 5#16
L_INDEX = 6#19
R_INDEX = 7#20
L_HIP = 8#23
R_HIP = 9#24
## HAND INDEXING
WRIST = 0
INDEX = 1
PINKY = 2
THUMB = 3

## INDEXING FOR SEGMENT ARRAYS
SHOULDER_WIDTH = 0
UPPERARM_LENGTH = 1
FOREARM_LENGTH = 2
SHOULDER_TO_HIP = 3
HIP_WIDTH = 4
## FOR HAND SEGMENT ARRAYS
W_TO_I = 0  # wrist to index
W_TO_P = 1  # wrist to pinky
W_TO_T = 2  # wrist to thumb
I_TO_P = 3  # index to pinky

## RATIOS
# these are the average ratios for each body segment/part to height
ARM_TO_HEIGHT = 0.39
FOREARM_TO_HEIGHT = 0.216
UPPERARM_TO_HEIGHT = ARM_TO_HEIGHT - FOREARM_TO_HEIGHT
SHOULDER_TO_HIP_TO_HEIGHT = 1       # temporarily set to 1, until the actual ratio is added
HIP_WIDTH_TO_HEIGHT = 1             # temporarily set to 1, until the actual ratio is added
# ndarray for indexing ratios for use below. 
#RATIOS_NDARRAY = np.zeros((10, 10), dtype = "float32")
# TODO: find a better way of doing this. this is just a temp fix for testing.
#RATIOS_NDARRAY[L_SHOULDER][L_ELBOW] = UPPERARM_TO_HEIGHT
#RATIOS_NDARRAY[L_ELBOW][L_WRIST] = FOREARM_TO_HEIGHT
#RATIOS_NDARRAY[R_SHOULDER][R_ELBOW] = UPPERARM_TO_HEIGHT
#RATIOS_NDARRAY[R_ELBOW][R_WRIST] = FOREARM_TO_HEIGHT

# hand ratios (rough measurement/estimation, should probably be updated at some point)
# all relative to WRIST_TO_INDEX
# (acquired by measuring this picture: https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png on screen using a measuring tape)
WRIST_TO_INDEX = 0.5        # base used for getting ratios in comparison
WRIST_TO_PINKY = 0.435
INDEX_TO_PINKY = 0.361
WRIST_TO_THUMB = 0.213


## INDEX DICTIONARIES
# for use converting vertex indices to segment index
VERTEX_TO_SEGMENT = {
    L_SHOULDER : {
        R_SHOULDER : SHOULDER_WIDTH,
        L_ELBOW : UPPERARM_LENGTH,
        L_HIP : SHOULDER_TO_HIP
    },
    R_SHOULDER : {
        L_SHOULDER : SHOULDER_WIDTH,
        R_ELBOW : UPPERARM_LENGTH,
        R_HIP : SHOULDER_TO_HIP
    },
    L_ELBOW : {
        L_WRIST : FOREARM_LENGTH
    },
    R_ELBOW : {
        R_WRIST : FOREARM_LENGTH
    }
}
# for use converting segment index to vertex indices
SEGMENT_TO_VERTEX = {
    SHOULDER_WIDTH : (L_SHOULDER, R_SHOULDER),
    UPPERARM_LENGTH : (
        (L_SHOULDER, L_ELBOW),
        (R_SHOULDER, R_ELBOW)
    ),
    FOREARM_LENGTH : (
        (L_ELBOW, L_WRIST),
        (R_ELBOW, R_WRIST)
    ),
    SHOULDER_TO_HIP : (
        (L_SHOULDER, L_HIP),
        (R_SHOULDER, R_HIP)
    ),
    HIP_WIDTH : (L_HIP, R_HIP)
}
# for use getting ratios given hand index pairs
HAND_VERTICES_TO_RATIOS = {
    WRIST : {
        INDEX : WRIST_TO_INDEX,
        PINKY : WRIST_TO_PINKY,
        THUMB : WRIST_TO_THUMB
    },
    INDEX : {
        PINKY : INDEX_TO_PINKY
    }
}


#### OBJECT FOR EASE OF MANAGEMENT OF EXTRAPOLATION OF DEPTH AND CALCULATION OF BODY FORCES
class Extrapolate_forces():
        
    # initialization
    def __init__(self, right = False, one_arm = False, mp_data_lock = threading.Lock()) -> None:
        ### USER INPUT DATA

        self.user_height = 1.78     # user height (meters)
        self.user_weight = 90       # user weight (kilograms)
        self.ball_mass = 3          # mass of ball (kilograms)

        #self.forearm_length = self.user_height * FOREARM_TO_HEIGHT      # estimated length of user's forearm (based on statistical average measurements)
        #self.upperarm_length = self.user_height * UPPERARM_TO_HEIGHT    # estimated length of user's upperarm (based on statistical average measurements)
        
        # toggle for calculating left arm or right arm
        self.is_right = right
        self.is_one_arm = one_arm

        # calibration settings
        self.manual_calibration = False
        self.sim_to_real_conversion_factor = 1  # convert mediapipe units to real world units (meters)
        self.use_full_wingspan = False
        self.use_biacromial = True              # use new calibration method
        self.biacromial_scale = 0.23            # temporarily set to middle of male (0.234) to female (0.227) range for testing

        # ndarray to store mediapipe data output, even if from other process(es)
        self.mediapipe_data_output = np.zeros((10, 3), dtype = "float64")
        # ndarray to store mediapipe hand data output
        self.hand_mp_out = np.zeros((2,3,3), dtype = "float32")
        self.hand_check = np.zeros((2), dtype = "float32")              # used to check if hand data updated
        self.hand_orientation = np.zeros((2, 2), dtype = "float32")     # theta: hand normal and forearm - 90 deg, phi: hand normal and screen normal

        # lock for mediapipe data
        self.mp_data_lock = mp_data_lock

        # used for storing distance data (to prevent unnecessary recalculations)
        # consider changing to float32 or float16
        self.dist_array = np.zeros((10, 10), dtype = "float64")         # indexed by two body part names/indices
        self.max_array = np.zeros((10, 10), dtype = "float64")          # used for storing max distance data
        self.avg_ratio_array = np.ones((10, 10), dtype = "float32")    # used for storing avg ratio distance between segments
        # store elbow angle in memory so it can be calculated right after depth for the given frame is calculated (to prevent syncing issues)
        self.elbow_angles = np.zeros((2), dtype = "float32")

        # bodypart_lengths intended to store baseline lengths of bodyparts
        self.bodypart_lengths = np.ones((6), dtype = "float32")         # stores body part lengths, assuming symmetry between sides (so, only one value for forearm length as opposed to 2, for example. may be changed later)
        # biases for bodypart lengths (calculated in countdown_calibrate), default to 1 for no bias
        self.bodypart_ratio_bias_array = np.ones((np.shape(self.bodypart_lengths)[0]), dtype = "float32")

        # stores calculated data by frame
        self.calculated_data = {
                "right_bicep_force": "NaN",
                "right_elbow_angle": "NaN",
                "left_bicep_force": "NaN",
                "left_elbow_angle": "NaN",
                "uarm_spher_coords": "NaN",
                "farm_spher_coords": "NaN"
            }

        self.cur_frame = 0   # used to keep track of current frame

        # put together pairs for each of the vertices
        # ordered in a particular manner which uses the shoulders as anchors for the elbows, and elbows as anchors for the wrists
        if not self.is_one_arm:                 # only track what needs to be tracked
            self.vertex_order = [
                [
                    L_SHOULDER,
                    R_SHOULDER
                ],
                [
                    L_SHOULDER,
                    L_ELBOW,
                    L_WRIST,
                ],
                [
                    R_SHOULDER,
                    R_ELBOW,
                    R_WRIST
                ]
            ]
        elif self.is_right:                     # set vertex order for only right arm
            self.vertex_order = [
                [
                    L_SHOULDER,
                    R_SHOULDER
                ],
                [
                    R_SHOULDER,
                    R_ELBOW,
                    R_WRIST,
                ]
            ]
        else:                                   # set vertex order for only left arm
            self.vertex_order = [
                [
                    L_SHOULDER,
                    R_SHOULDER
                ],
                [
                    L_SHOULDER,
                    L_ELBOW,
                    L_WRIST,
                ]
            ]
        
        # run initialization functions
        try:
            self.set_all_bodypart_lengths()
        except:
            print("extrapolation.py: ERROR initializing bodypart lengths")

        print("extrapolation.py: Info: Initialized extrapolation.py")


    # IMPORTANT: set mediapipe_data_output for the current frame
    def update_current_frame(self, mp_data_out, hand_mp_out, current_frame):
        try:
            # set data of current frame dataset
            self.mediapipe_data_output = mp_data_out
            self.hand_mp_out = hand_mp_out
            
            # reset dist_array
            self.dist_array = np.zeros(np.shape(self.dist_array))

            # update current frame number
            self.cur_frame = current_frame

            # update calibration settings (old)
            #try:
            #    if self.use_full_wingspan and not self.is_one_arm:
            #        self.calc_wingspan()                            # keep track of max distance between index fingers
            #    else:
            #        self.calc_shoulder_width()
            #        #self.calc_half_wingspan()                       # keep track of max length of given arm
            #        self.calc_avg_ratio_shoulders()
            #except:
            #    print("extrapolation.py: Error updating calibration in update_current_frame()")

            # calculate calibration coefficient/metric to sim units conversion ratio
            self.calc_conversion_ratio()
            
            # set depth (pose landmarker data)
            self.set_depth()

            # calculate bicep forces
            self.calc_bicep_force()

        except:
            print("extrapolation.py: ERROR in update_current_frame(%s)" % current_frame)

    # IMPORTANT: temporary bandaid fix for calibration
    #def calc_wingspan(self):
    #    self.calc_dist_between_vertices(L_INDEX, R_INDEX)

    # track max dist between half wingspan for calibration (automatically done via calc_dist_between_vertices, updating max_dist)
    #def calc_half_wingspan(self):
    #    try:
    #        # keep track of arm length
    #        self.calc_dist_between_vertices((L_INDEX + (int)(self.is_right)), (L_SHOULDER + (int)(self.is_right)))
    #    except:
    #        print("extrapolation.py: ERROR in `calc_half_wingspan()`")

    # track shoulder width (needed for shoulder-based calibration)
    def calc_shoulder_width(self):
        try:
            # keep track of shoulder width
            self.dist_array[L_SHOULDER][R_SHOULDER] = self.calc_dist_between_vertices(L_SHOULDER, R_SHOULDER)
        except:
            print("extrapolation.py: ERROR in `calc_shoulder_width()`")



    ### HELPER FUNCTIONS:

    # set height and weight and ball mass externally
    def set_hwb(self, height, weight, ball):
        self.user_height = float(height)
        self.user_weight = float(weight)
        self.ball_mass = float(ball)

    # set biacromial ratio externally
    def set_biacromial(self, new_biacromial = 0.23):
        # change local biacromial scale value
        self.biacromial_scale = new_biacromial
        # set shoulder width true length
        self.bodypart_lengths[0] = self.user_height * self.biacromial_scale

    # get sim units from estimated real length of part
    def real_to_sim_units(self, length):
        try:# convert real (length) units to sim (length) units
            return length / self.sim_to_real_conversion_factor
        except:
            print("extrapolation.py: ERROR converting real units to sim units")

    # return calculated data (for use by other classes)
    def get_calculated_data(self):
        # returns the currently stored calculated data
        return self.calculated_data



    ### DISTANCE FUNCTIONS:

    # get distance between vertices for current frame
    def calc_dist_between_vertices(self, first_part = 0, second_part = 0, is_hand = False):
        try:
            # calculate distance for parts in current frame
            #dist = np.linalg.norm(
            #                self.mediapipe_data_output[second_part, :] - 
            #                self.mediapipe_data_output[first_part, :]
            #            )
            
            if not is_hand: # check if using hand data
                first = self.mediapipe_data_output[first_part]
                second = self.mediapipe_data_output[second_part]
            else:           # work with hand data
                first = self.hand_mp_out[first_part]
                second = self.hand_mp_out[second_part]

            dist = np.sqrt( (second[0] - first[0])**2 + (second[2] - first[2])**2 )

            # update max distance between these parts, if necessary
            if (not is_hand) and dist > self.max_array[first_part][second_part]:
                self.max_array[first_part][second_part] = dist

            # what this chunk of code does is basically act as a workaround for not having a way to properly work with segments instead of just vertices
            #try:
            #    # update avg_ratio_array between these parts (this is a TEMP FIX for testing. TODO: make a better thing later)
            #    cur_ratio = 0.0                             # used as temp var to hold ratio to use below
            #    first_vertex = int(first_part / 2) * 2         # used to make left and right vertices effectively the same
            #
            #    match first_vertex:                         # check which vertex it is to find the assumed segment
            #        case 0:
            #            cur_ratio = UPPERARM_TO_HEIGHT      # assume it's the upper arm
            #        case 2:
            #            cur_ratio = FOREARM_TO_HEIGHT       # assume it's the forearm
            #        case _:
            #            cur_ratio = 1.0
            #    if second_part == 1:                        # assume it's shoulder to shoulder if the second part is right shoulder
            #        cur_ratio = self.biacromial_scale
            #        self.calc_avg_ratio_shoulders()
            #    else:
                    # get sim units from real units
            #        self.avg_ratio_array[first_part][first_part + 2] = self.real_to_sim_units(self.user_height * cur_ratio)
            #except:
            #    print("extrapolation.py: Error handling avg_ratio_array[][]")

            return dist
        except:
            print("extrapolation.py: ERROR in `calc_dist_between_vertices()`")

        
    # get avg ratio for shoulder distance
    #def calc_avg_ratio_shoulders(self):
    #    try:
    #        self.avg_ratio_array[0][1] = self.real_to_sim_units(self.user_height * self.biacromial_scale)
    #    except:
    #        print("extrapolation.py: ERROR in `calc_avg_ratio_shoulders()`")


    # retrieve the max distance between body parts found thus far
    def get_max_dist(self, first_part, second_part):
        return float(self.max_array[first_part][second_part])



    ### NEW CALIBRATION: AVG RATIOS AND OFFSETS

    # calculate true length of segments (e.g. upper arm) via use of avg ratios and estimated deviation from avg ratios
    # 
    # this function is not intended to run every frame; this is calculated before any conversions to or from sim units,
    #   thus it doesn't need to be updated when a new calibration coefficient is calculated either.
    # these calculations are intended to represent the true lengths of the user's body parts in metric units.
    # this function should be run whenever user_height is changed or countdown_calibrate() run (as well as at program start)
    def set_all_bodypart_lengths(self):
        # assuming symmetry between left and right side
        try:
            # shoulder width
            self.bodypart_lengths[SHOULDER_WIDTH] = self.user_height * self.biacromial_scale
            # upper arms
            self.bodypart_lengths[UPPERARM_LENGTH] = self.user_height * UPPERARM_TO_HEIGHT * self.bodypart_ratio_bias_array[UPPERARM_LENGTH]
            # forearms
            self.bodypart_lengths[FOREARM_LENGTH] = self.user_height * FOREARM_TO_HEIGHT * self.bodypart_ratio_bias_array[FOREARM_LENGTH]
            # shoulder to hip
            self.bodypart_lengths[SHOULDER_TO_HIP] = self.user_height * SHOULDER_TO_HIP_TO_HEIGHT * self.bodypart_ratio_bias_array[SHOULDER_TO_HIP]
            # hip width
            self.bodypart_lengths[HIP_WIDTH] = self.user_height * HIP_WIDTH_TO_HEIGHT * self.bodypart_ratio_bias_array[HIP_WIDTH]
        except:
            print("extrapolation.py: ERROR calculating in `set_bodypart_lengths()`")

    # run countdown to snapshot to calibration step to determine avg ratio offsets/std dev for current user
    # TODO:
    #   - whenever the user's height is adjusted, recalculate avg ratios based on the calibration step data
    #   - draw countdown numbers on screen until snapshot taken
    #   - once button is clicked, instruct user to get to a position where they can show their full wingspan, upper body, and hips, then click button to confirm to start countdown
    #def countdown_calibrate(self):



    ### CALIBRATION CONVERSION FACTOR:

    # calculate conversion ratio using the new shoulder width method
    def calc_conversion_ratio(self, real_height_metric = 1.78):
        try:
            # get maximum distance between shoulders
            #sim_biacromial = self.get_max_dist(L_SHOULDER, R_SHOULDER)      # still using max distance for calibration here
            #self.avg_ratio_array[L_SHOULDER][R_SHOULDER] = self.real_to_sim_units(self.user_height * cur_ratio)   # calc avg ratio between shoulders (redundancy)
            #sim_biacromial = self.avg_ratio_array[L_SHOULDER][R_SHOULDER]   # now using avg ratios instead of max distance

            # calculate avg ratio for shoulder to shoulder distance for use in depth approximations
            self.calc_shoulder_width()

            # getting current simulation distance between shoulders
            sim_biacromial = self.dist_array[L_SHOULDER][R_SHOULDER]

            # use true shoulder width and current distance between shoulders in sim units to get conversion factor
            self.sim_to_real_conversion_factor = self.bodypart_lengths[SHOULDER_WIDTH] / sim_biacromial
        except:
            print("extrapolation.py: ERROR calculating conversion ratio")

    # calculate ratio for conversion of simulated units to metric units (meters) using wingspan and input real height
    # using the wingspan method
    # (unused)
    #def calc_conversion_ratio_wingspan(self, real_height_metric = 1.78):        # DEPRECATED
    #    # get ratio to real distance in meters using max distance between wrists via mediapipe output data
    #    if self.use_full_wingspan:
    #        sim_wingspan = self.get_max_dist(L_INDEX, R_INDEX)
    #
    #    # new calibration method
    #    # uses half wingspan, for ease of use
    #
    #    # set global conversion factor
    #    #global sim_to_real_conversion_factor
    #    if not self.manual_calibration:
    #        # calculate for full wingspan
    #        if self.use_full_wingspan: 
    #            sim_wingspan = self.get_max_dist(L_INDEX, R_INDEX)
    #            self.sim_to_real_conversion_factor = real_height_metric / sim_wingspan
    #        # calculate for half wingspan
    #        else:
    #            half_wingspan = self.get_max_dist((L_INDEX + (int)(self.is_right)), (L_SHOULDER + (int)(self.is_right))) + (self.get_max_dist(L_SHOULDER, R_SHOULDER) / 2)
    #            self.sim_to_real_conversion_factor = real_height_metric / (half_wingspan * 2)
    #
    #    return self.sim_to_real_conversion_factor
    
    # get conversion ratio (so it doesn't need to be calculated for each of these calls)
    def get_conversion_ratio(self):
        return self.sim_to_real_conversion_factor
    
    # set the conversion factor/ratio manually
    def set_conversion_ratio(self, conv_ratio):
        self.sim_to_real_conversion_factor = conv_ratio
    
    # set calibration to manual
    #def set_calibration_manual(self, is_manual = True):
    #    self.manual_calibration = is_manual
    
    # reset max_array data to essentially reset the application (no longer used in calibration)
    #def reset_calibration(self):
    #    self.max_array = np.zeros((8,8), dtype = "float64") # reset max distances
    #    self.sim_to_real_conversion_factor = 1
        


    ### DEPTH:

    # calculate the angle of the segment (body part) from the normal (of the screen/camera) (where it is longest)
    def angle_from_normal(self, cur_dist, max_dist):
        try:
            # angle should be always between 0 and 90 degrees (sorta like phi in spherical coordinates)
            return np.arccos(np.clip(cur_dist / max_dist, -1, 1))
        except:
            print("extrapolation.py: ERROR in `angle_from_normal()")

    # get depth for body part in most recent frame
    def get_depth(self, vertex_one = 0, vertex_two = 1, is_hand = False):
        try:
            cur_dist = self.calc_dist_between_vertices(vertex_one, vertex_two, is_hand)      # current distance between given parts
            
            if is_hand:     # check if calculating for hand data, if so, calculate for hand
                max_dist = HAND_VERTICES_TO_RATIOS[vertex_one][vertex_two]
            else:               # use pose landmarker data
                segment_index = VERTEX_TO_SEGMENT[vertex_one][vertex_two]               # get segment index for getting bodypart length 
                max_dist = self.bodypart_lengths[segment_index]                         # set max_dist to true length of given bodypart/segment

            #print("v1 %s, v2 %s, si %s" % (vertex_one, vertex_two, segment_index))  # DEBUG
            angle = self.angle_from_normal(((self.sim_to_real_conversion_factor * int(not is_hand)) * cur_dist), max_dist)   # calculate difference between max distance and current distance

            r = np.sin(angle) * max_dist                                            # calculate depth
            #print(r)

            return r
        except:
            print("extrapolation.py: ERROR in `get_depth()`")

    # get y axes/depths by order of body parts for the current frame
    def set_depth(self):
        try:
            # go thru 2d array of vertex order
            for i in range(0, len(self.vertex_order)):  # for each set of vertices denoting body segments
                #print("vertices: " + str(i))
                for j in range(0, (len(self.vertex_order[i]) - 1)):   # for each vertex in set of vertices (except the last one)
                        #print(self.vertex_order[i][j])
                        #print(self.vertex_order[i][j + 1])
                        #if self.vertex_order[i][j] != self.vertex_order[i][-1]:  # if current vertex isn't the last in the set
                        # calculate depth for vertex pair
                        y_dist_between_vertices = self.get_depth(self.vertex_order[i][j], self.vertex_order[i][j + 1])          # calculate depth
                        
                        # check if "nan" value
                        if math.isnan(y_dist_between_vertices):
                            y_dist_between_vertices = 0                             # set all nan values to 0

                        # add previous anchor vertex (if not first in set)
                        if self.vertex_order[i][j] > 0:   # if not L_SHOULDER
                            vertex_y = self.mediapipe_data_output[self.vertex_order[i][j]][1] +  y_dist_between_vertices      # add y depth of anchor (previous node) to current
                        else:
                            vertex_y = y_dist_between_vertices

                        # set depth in current frame of mediapipe data
                        self.mediapipe_data_output[self.vertex_order[i][j + 1], 1] = vertex_y
            
            # calculate elbow angle for both arms
            self.calc_elbow_angle(right_side = False)    # left
            self.calc_elbow_angle(right_side = True)     # right
        except:
            print("extrapolation.py: ERROR in set_depth()")



    ### HAND CALCULATIONS

    # calculate orientation of hand (called from calc_elbow_angle)
    def calc_hand_orientation(self, is_right = False, forearm = np.zeros((3), dtype = "float32"), cross_ua_fa = np.zeros((3), dtype = "float32")):
        try: 
            i = int(is_right)
            hand_check = self.hand_mp_out[i, 0, 0] + self.hand_mp_out[i, 1, 0] + self.hand_mp_out[i, 2, 0]  # used for checking for changes in hand data (prevent redundant calculations)
            # get depth of hand parts
            # check if hand data is present by checking the sum of the x component of each vertex, which should be different if change occurred
            if not (hand_check == self.hand_check[i]):
                w_to_i = self.hand_mp_out[i, 1, :] - self.hand_mp_out[i, 0, :]  # wrist to index vector
                w_to_p = self.hand_mp_out[i, 2, :] - self.hand_mp_out[i, 0, :]  # wrist to pinky vector
                #p_to_i = self.hand_mp_out[i, 1, :] - self.hand_mp_out[i, 2, :]  # pinky to index vector
                #screen_normal = np.zeros((3), dtype = "float32")                # normal of screen
                #screen_normal[:] = (0, (-(-1)**int(is_right)), 0)

                # get normal of hand data (cross product between 0,5 vector and 0,17 vector) unit vector
                hand_normal = np.cross(w_to_i, w_to_p)
                hand_normal /= np.linalg.norm(hand_normal)

                # normal between forearm and normal between forearm and upper arm
                normal_fa_ua_fa = np.cross(cross_ua_fa, forearm)

                # get angle between using atan2
                # phi (hand normal to forearm - 90 degrees)
                phi = np.arctan2(np.linalg.norm(np.cross(hand_normal, forearm)), np.dot(hand_normal, forearm)) - (np.pi/2)
                if not (phi == np.nan):
                    self.hand_orientation[i, 1] = phi
                # theta (hand normal to screen normal)
                theta = np.arctan2(np.linalg.norm(np.cross(normal_fa_ua_fa, hand_normal)), np.dot(normal_fa_ua_fa, hand_normal))
                if not (theta == np.nan):
                    self.hand_orientation[i, 0] = theta
            
                #if not is_right:
                    #DEBUG
                #    print("\nAngle between hand and forearm: \tTheta: %s\t Phi: %s\n" % (np.rad2deg(self.hand_orientation[i, 0]), np.rad2deg(self.hand_orientation[i, 1])))

            self.hand_check[i] = hand_check     # update hand check for use next timestep/frame

        except Exception as e:
            print("extrapolation.py: ERROR in `calc_hand_orientation()`: %s\n" % str(e))

    # get depth for hand parts (for one hand)
    # TODO: combine this with set_depth()
    #def set_hand_depth(self, is_right = False):
    #    try:
            # go thru all vertices for hand
    #        for i in range(0, 4):#len(self.hand_mp_out)):
                
    #            wrist_loc = self.hand_mp_out[is_right, 0]
    #            index_loc = self.hand_mp_out[is_right, 1]
    #            pinky_loc = self.hand_mp_out[is_right, 2]
    #            thumb_loc = self.hand_mp_out[is_right, 3]

                


    #    except Exception as e:
    #        print("extrapolation.py: ERROR in `set_hand_depth()`: %s\n" % str(e))



    ### FORCES CALCULATIONS

    # calculate elbow angle
    def calc_elbow_angle(self, right_side = False):
        try:
            # coordinate data for shoulder, elbow, and wrist
            x = self.mediapipe_data_output[(0 + (int)(right_side)):(5 + (int)(right_side)):2, 0]
            y = self.mediapipe_data_output[(0 + (int)(right_side)):(5 + (int)(right_side)):2, 1]
            z = self.mediapipe_data_output[(0 + (int)(right_side)):(5 + (int)(right_side)):2, 2]

            # DEBUG
            #if not right_side:
                #print("extrapolation.py: DEPTH OF LEFT ELBOW: " + str(y[1]))

            #shoulder = self.mediapipe_data_output[(0 + (int)(right_side))]
            #elbow = self.mediapipe_data_output[(2 + (int)(right_side))]
            #wrist = self.mediapipe_data_output[(4 + (int)(right_side))]

            # get unit vectors representing upper and lower arm
            vector_a = [(x[0] - x[1]), (y[0] - y[1]), (z[0] - z[1])]    # upper arm
            vector_b = [(x[2] - x[1]), (y[2] - y[1]), (z[2] - z[1])]    # lower arm
            vector_a = vector_a / np.linalg.norm(vector_a)  # turn into unit vector
            vector_b = vector_b / np.linalg.norm(vector_b)  # turn into unit vector

            #print(vector_a)
            #print(vector_b)

            # get magnitude of vectors squared (i.e. not using sqrt yet, for use in quaternion solution of elbow angle)
            #vector_a_mag = (vector_a[0] ** 2) + (vector_a[1] ** 2) + (vector_a[2] ** 2)
            #vector_b_mag = (vector_b[0] ** 2) + (vector_b[1] ** 2) + (vector_b[2] ** 2)

            # convert vectors to unit vectors
            #vector_a = vector_a / np.sqrt(vector_a_mag)
            #vector_b = vector_b / np.sqrt(vector_b_mag)

            #dot_ab = np.dot(vector_a, vector_b)

            # get the norm of the cross product of the two vectors
            #cross_ab = np.cross(vector_a, vector_b)
            # using atan2
            #norm_cross = np.sqrt(cross_ab[0]**2 + cross_ab[1]**2 + cross_ab[2]**2)

            # calculate angle at elbow
            #elbow_angle = np.arccos( np.clip( ( ((vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1]) + (vector_a[2] * vector_b[2])) / (vector_a_mag * vector_b_mag) ), -1, 1) )#[0] )
            
            # using arctan2
            cross_ua_fa = np.cross(vector_b, vector_a)
            self.elbow_angles[(int)(right_side)] = np.pi - np.arctan2(np.linalg.norm(cross_ua_fa), np.dot(vector_b, vector_a))


            # trying with quaternion stuff instead
            # the following should give us the "shortest arc", which is basically what we're looking for, and is one of a couple of options using quaternion math for this application
            # helped by the following two sources:
            # https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another#1171995
            # https://github.com/toji/gl-matrix/blob/f0583ef53e94bc7e78b78c8a24f09ed5e2f7a20c/src/gl-matrix/quat.js#L54 
            
            # quaternion to hold the 
            #quaternion = np.array((4))
            # check dot product to account for potential errors from edge cases:
            # if dot is less than -0.999999 (i.e. opposite parallel), return 180 degrees (pi)
            #elbow_angle = 0
            #if (dot_ab < -0.999999):
            #    elbow_angle = np.pi
            # if dot greater than 0.999999 (i.e. parallel), return 0 degrees (0)
            #elif (dot_ab > 0.999999):
            #    elbow_angle = 0
            #else:
            #    elbow_angle = 2 * np.acos(np.clip((np.sqrt(1 + dot_ab)), -1, 1))  # doing this so we don't waste memory on the quat_w variable if not necessary. the 1 accounts for the i+j+k part, as it always adds up to 1 at the end of it
                
                #quaternion[0:3] = cross_ab  # set first 3 elements of quaternion (x y and z) to the cross product of the two vectors (which represent the upper arm and forearm with the elbow as the origin)
                # set w value of quaternion (np.clip is used to account for if the vectors are parallel, issues w the cross product may occur)
                # assuming the square root of the product of the square of the two vectors = 1, since they're unit vectors
                #quaternion[3] = np.sqrt(1 + dot_ab)
                #quat_w = np.sqrt(1 + dot_ab) # w is all that's needed, since we can extract the angle from it directly
                #elbow_angle = 2 * np.acos(quat_w)  

            #DEBUGGING
            #if right_side:
            #    print(np.rad2deg(self.elbow_angles[(int)(right_side)]))
            #rint("vector A: ", vector_a)
            #print("vector B: ", vector_b)

            #return elbow_angle

            # call calc_hand_orientation from here to prevent need to recalculate vector_b
            self.calc_hand_orientation(right_side, vector_b, cross_ua_fa)

        except:
            print("extrapolation.py: ERROR in `calc_elbow_angle()`")

    # get spherical coordinates for each of the 3 vertices (bodyparts) of interest
    # vertex_one is the anchor point, and vertex_two is calculated based on its anchor
    # NOTE: x is horizontal, z is up and down, y is forward and backwards (in the coordinate system we're using; comes from past version of program)
    #def calc_spher_coords(self, vertex_one, vertex_two):    
    #    try:
            # basically one dimensional vectors
    #        x_diff = self.mediapipe_data_output[vertex_two][0] - self.mediapipe_data_output[vertex_one][0]
    #        y_diff = self.mediapipe_data_output[vertex_two][1] - self.mediapipe_data_output[vertex_one][1]
    #        z_diff = self.mediapipe_data_output[vertex_two][2] - self.mediapipe_data_output[vertex_one][2]

            #rho = np.sqrt((x_diff ** 2) + (y_diff ** 2) + (z_diff ** 2))
            #print("test")
    #        rho = self.bodypart_lengths[VERTEX_TO_SEGMENT[vertex_one][vertex_two]]  # rho = true segment length
            #print("%s", rho)
            # division by zero is fine here for now, since it returns infinity, and np.arctan(infinity) = 90 degrees
    #        theta = np.arctan(x_diff / y_diff)                   # swapped x and y due to equations having different Cartesian coordinate system layout
            #print(theta)
            # NOTE: find better way to do this (preferably without "if" statements)
            # ensure the argument for np.arccos() is always less than or equal to 1
    #        phi = np.arccos(np.clip((z_diff / rho), -1, 1))
            #print(phi)
            
            # DEBUG
    #        if vertex_one == 2: # left elbow
    #            print("Forearm spherical coords: (%s, %s, %s)" % (rho, theta, phi))

    #        return [rho, theta, phi]
    #    except:
    #        print("extrapolation.py: ERROR in `calc_spher_coords()`")#%s, %s)`" % (vertex_one, vertex_two))

    # new version of calc_spher_coords (using up as axis)
    def calc_spher_coords(self, is_right, vertex_one, vertex_two):
        try:
            # get vector from given vertices/points
            vector = self.mediapipe_data_output[vertex_two] - self.mediapipe_data_output[vertex_one]

            # use up vector as polar axis
            z_axis = (0, 0, 1)
            x_axis = ((-1)**(int(is_right)), 0, 0)  # x axis is -1 if is_right is True (i.e. (-1)^(1)), or 1 if False (i.e. (-1)^(0))

            rho = self.bodypart_lengths[VERTEX_TO_SEGMENT[vertex_one][vertex_two]]  # rho = true segment length
            #theta = np.arccos(vector[1] / rho)
            # calculate theta; right now, this only has a range of 180 degrees in front of the anchor (i.e. vertex_two).abs
            #   to fix this, check if vertex is in front of or behind other vertex; if behind, multiply by -1 to get full range (i.e. 0 to -pi and 0 to pi)
            theta = np.abs(np.arctan2(vector[2], (x_axis[0] * vector[0])) / 2)

            vector /= np.linalg.norm(vector)    # turn to unit vector
            # using atan2 to get angle between polar axis and current body segment
            phi = np.arctan2(np.linalg.norm(np.cross(z_axis, vector)), np.dot(z_axis, vector)) - (np.pi/2)
            
            # DEBUG
            #if not is_right: # left elbow anchor => upper arm
            segment = "<segment>"
            match vertex_one:
                case 0:
                    segment = "\nLeft upper arm"
                case 1:
                    segment = "\nRight upper arm"
                case 2:
                    segment = "Left lower arm"
                case 3:
                    segment = "Right lower arm"
                case _:
                    segment = segment
            
            print("%s spherical coords: (%s, %s, %s)" % (segment, rho, theta, phi))

            return [rho, theta, phi]
        except:
            print("extrapolation.py: ERROR in `calc_spher_coords()`")#%s, %s)`" % (vertex_one, vertex_two))



    ### FORMULA CALCULATIONS

    # calculate forces of muscle exertions of the arm
    def calc_bicep_force(self):#, is_right = False):   # allow choosing which arm to use
        try:
            # constants, used for organization and readability
            RHO = 0
            THETA = 1
            PHI = 2
            
            w_bal = self.ball_mass           # kilograms   # weight of ball

            # temp storage for getting both arms in one data frame
            right_elbow_angle = 0
            right_bicep_force = 0
            left_elbow_angle = 0
            left_bicep_force = 0

            # run through once for left, once for right
            for is_right in [0, 1]:
                # only calculate the following if the elbow angle exists
                elbow_angle = self.elbow_angles[int(is_right)]
                if math.isnan(elbow_angle):
                    return math.nan                                         # if elbow_angle == nan, exit function by returning nan

                # convert sim units to metric units
                conv_factor = self.sim_to_real_conversion_factor

                # get spherical coordinate data for arm segments
                try:
                    uarm_spher_coords = self.calc_spher_coords(bool(is_right), (L_SHOULDER + (int)(is_right)), (L_ELBOW + (int)(is_right)))
                    farm_spher_coords = self.calc_spher_coords(bool(is_right), (L_ELBOW + (int)(is_right)), (L_WRIST + (int)(is_right)))
                except:
                    print("extrapolation.py: ERROR calculating spherical coords in `calc_bicep_force()`")

                # get arm segment lengths in metric units (meters)
                try:
                    f = farm_spher_coords[RHO] * conv_factor
                    u = uarm_spher_coords[RHO] * conv_factor
                    #print("%0.2f" % f)
                    b = u * 0.11 #0.636                                         # calculated via algebra using pre-existing average proportions data
                    w_fa = self.user_weight * (f * 0.1065)                      # use ratio of f to weight proportion to get weight with calculated f 
                    cgf = 2 * (f ** 2)                                          # calculated via algebra using pre-existing average proportions data
                except:
                    print("extrapolation.py: ERROR calculating metrics in `calc_bicep_force()`")

                # angles
                try:
                    #theta_arm = (np.pi / 2) - farm_spher_coords[THETA]          # angle at shoulder
                    theta_uarm = (np.pi / 2) + uarm_spher_coords[THETA]         # angle of upper arm
                    theta_u = elbow_angle#theta_arm + theta_uarm                # angle at elbow
                    theta_b = np.pi - ( (b - u * np.sin(theta_u)) / np.sqrt( (b ** 2) + (u ** 2) - 2 * b * u * np.sin(theta_u) ) )      # angle at bicep insertion point
                    theta_la = np.cos(theta_uarm) #theta_u - theta_arm - np.pi) #np.sin(theta_uarm)        # angle used for leverage arms fa and bal
                except:
                    print("extrapolation.py: ERROR calculating angles in `calc_bicep_force()`")

                # lever arms
                try:
                    la_fa = cgf * theta_la                                      # forearm lever arm
                    la_bal = f * theta_la                                       # ball lever arm
                    la_bic = b * np.sin(theta_b)                                # bicep lever arm
                except:
                    print("extrapolation.py: ERROR calculating lever arms in `calc_bicep_force()`")

                # forces
                force_bicep = (w_fa * la_fa + w_bal * la_bal) / la_bic      # force applied by bicep muscle

                # handle which arm is currently calculating
                if is_right:
                    right_elbow_angle = elbow_angle
                    right_bicep_force = force_bicep         # seems to be having issues; may be due to being flipped relative to left, and trig functions don't account for that.
                else:
                    left_elbow_angle = elbow_angle
                    left_bicep_force = force_bicep




            # set/return data in dictionary format
            self.calculated_data = {
                "right_bicep_force": str("%0.2f" % right_bicep_force),
                "right_elbow_angle": str("%0.2f" % np.rad2deg(right_elbow_angle)),
                "left_bicep_force": str("%0.2f" % left_bicep_force),
                "left_elbow_angle": str("%0.2f" % np.rad2deg(left_elbow_angle)),
                "uarm_spher_coords": str(uarm_spher_coords),
                "farm_spher_coords": str(farm_spher_coords)
            }
            #print("%0.2f" % force_bicep)

            return self.calculated_data
        except:
            print("extrapolation.py: ERROR in `calc_bicep_force()`")



    ### BICEP FORCES GRAPH

    # graph bicep forces
    def plot_bicep_forces(self, bicep_force, elbow_angle):
        try:
            # plot bicep force / theta_u
            y = np.abs(bicep_force)            # bicep force
            x = np.rad2deg(elbow_angle)        # angle at elbow

            # plot
            plt.scatter(x,y)
            #plt.scatter(range(0, np.shape(freemocap_3d_body_data[:,0,0])[0] ), y)
            plt.ylim([0, 1000])     # restrict y axis to between 0 and 1000 (Newtons)
            plt.xlim([0, 180])      # restrixt x axis to between 0 and 180 (degrees)

            # plot bicep forces
            return plt
        except:
            print("extrapolation.py: ERROR in `plot_bicep_forces()`")

