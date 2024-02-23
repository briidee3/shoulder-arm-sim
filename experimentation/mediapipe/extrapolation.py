# BD 2023
# This program is designed to reimplement code from a previous program for use in a new environment
# in order to extrapolate 3D motion tracking data from 2D motion tracking data and user input.
# This version has been edited for use directly with MediaPipe, as opposed to with FreeMoCap data output.


# TODO: 
#   - optimize code
#       - minimize reads/writes
#           - try to do in-place manipulations of data
#   - implement multithreading, like how it was done for `livestream_mediapipe_class.py`
#   - take picture of user, notifying user, making em click a button, then counting down, snapping pic
#       - this will be the calibration shot, and function "recalibrate" will do this.
#       - this removes the need for lots of unnecessary calculations and reads/writes.
#       - the output of this will be used to calculate depth and 
#   - figure out if raw mediapipe output is x, y, z, or x, z, y
#   - fix elbow angle inaccuracies (e.g. how 90 degrees isn't seen as 90 degrees due to the calculations
#       being used being parallel to the plane of the screen/webcam)
#   - orientation from shoulder to hip stuffs
#       - send over the hips data from the mediapipe stuff
#   - initialize an array at the start containing vector lengths of segment distances
#   - update the actual lengths of body segments during run time

# IDEAS:
#   - train a machine learning model to tune the weights for the ratios (after implementing ratio-based depth instead of max_dist-based depth)


import numpy as np
import math
from matplotlib import pyplot as plt



# set to not display in scientific notation
np.set_printoptions(suppress = True, precision = 3)

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

## INDEXING FOR SEGMENT ARRAYS
SHOULDER_WIDTH = 0
UPPERARM_LENGTH = 1
FOREARM_LENGTH = 2
SHOULDER_TO_HIP = 3
HIP_WIDTH = 4

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


#### OBJECT FOR EASE OF MANAGEMENT OF EXTRAPOLATION OF DEPTH AND CALCULATION OF BODY FORCES
class Extrapolate_forces():
        
    # initialization
    def __init__(self, right = False, one_arm = False) -> None:
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
        self.mediapipe_data_output = np.ndarray((10, 3), dtype = "float64")

        # used for storing distance data (to prevent unnecessary recalculations)
        # consider changing to float32 or float16
        self.dist_array = np.zeros((10, 10), dtype = "float64")         # indexed by two body part names/indices
        self.max_array = np.zeros((10, 10), dtype = "float64")          # used for storing max distance data
        self.avg_ratio_array = np.ones((10, 10), dtype = "float32")    # used for storing avg ratio distance between segments

        # bodypart_lengths intended to store baseline lengths of bodyparts
        self.bodypart_lengths = np.ones((6), dtype = "float32")         # stores body part lengths, assuming symmetry between sides (so, only one value for forearm length as opposed to 2, for example. may be changed later)
        # biases for bodypart lengths (calculated in countdown_calibrate), default to 1 for no bias
        self.bodypart_ratio_bias_array = np.ones((np.shape(self.bodypart_lengths)[0]), dtype = "float32")

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
    def update_current_frame(self, mp_data_out, current_frame):
        try:
            # set data of current frame dataset
            self.mediapipe_data_output = mp_data_out
            
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
        except:
            print("extrapolation.py: ERROR in update_current_frame(%s)" % current_frame)

    # IMPORTANT: temporary bandaid fix for calibration
    def calc_wingspan(self):
        self.calc_dist_between_vertices(L_INDEX, R_INDEX)

    # track max dist between half wingspan for calibration (automatically done via calc_dist_between_vertices, updating max_dist)
    def calc_half_wingspan(self):
        try:
            # keep track of arm length
            self.calc_dist_between_vertices((L_INDEX + (int)(self.is_right)), (L_SHOULDER + (int)(self.is_right)))
        except:
            print("extrapolation.py: ERROR in `calc_half_wingspan()`")

    # track shoulder width
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



    ### DISTANCE FUNCTIONS:

    # get distance between vertices for current frame
    def calc_dist_between_vertices(self, first_part, second_part):
        try:
            # calculate distance for parts in current frame
            dist = np.linalg.norm(
                            self.mediapipe_data_output[first_part, :] - 
                            self.mediapipe_data_output[second_part, :]
                        )
            
            # update max distance between these parts, if necessary
            if dist > self.max_array[first_part][second_part]:
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
    def calc_avg_ratio_shoulders(self):
        try:
            self.avg_ratio_array[0][1] = self.real_to_sim_units(self.user_height * self.biacromial_scale)
        except:
            print("extrapolation.py: ERROR in `calc_avg_ratio_shoulders()`")


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
    def calc_conversion_ratio_wingspan(self, real_height_metric = 1.78):        # DEPRECATED
        # get ratio to real distance in meters using max distance between wrists via mediapipe output data
        if self.use_full_wingspan:
            sim_wingspan = self.get_max_dist(L_INDEX, R_INDEX)

        # new calibration method
        # uses half wingspan, for ease of use

        # set global conversion factor
        #global sim_to_real_conversion_factor
        if not self.manual_calibration:
            # calculate for full wingspan
            if self.use_full_wingspan: 
                sim_wingspan = self.get_max_dist(L_INDEX, R_INDEX)
                self.sim_to_real_conversion_factor = real_height_metric / sim_wingspan
            # calculate for half wingspan
            else:
                half_wingspan = self.get_max_dist((L_INDEX + (int)(self.is_right)), (L_SHOULDER + (int)(self.is_right))) + (self.get_max_dist(L_SHOULDER, R_SHOULDER) / 2)
                self.sim_to_real_conversion_factor = real_height_metric / (half_wingspan * 2)

        return self.sim_to_real_conversion_factor
    
    # get conversion ratio (so it doesn't need to be calculated for each of these calls)
    def get_conversion_ratio(self):
        return self.sim_to_real_conversion_factor
    
    # set the conversion factor/ratio manually
    def set_conversion_ratio(self, conv_ratio):
        self.sim_to_real_conversion_factor = conv_ratio
    
    # set calibration to manual
    def set_calibration_manual(self, is_manual = True):
        self.manual_calibration = is_manual
    
    # reset max_array data to essentially reset the application
    def reset_calibration(self):
        self.max_array = np.zeros((8,8), dtype = "float64") # reset max distances
        self.sim_to_real_conversion_factor = 1
        


    ### DEPTH:

    # calculate the angle of the segment (body part) from the normal (where it is longest)
    def angle_from_normal(self, cur_dist, max_dist):
        try:
            # angle always between 0 and 90 degrees
            return np.arccos(min(cur_dist / max_dist, 1))
        except:
            print("extrapolation.py: ERROR in `angle_from_normal()")

    # get depth for body part in most recent frame
    def get_depth(self, vertex_one, vertex_two):
        try:
            cur_dist = self.calc_dist_between_vertices(vertex_one, vertex_two)      # current distance between given parts
            
            segment_index = VERTEX_TO_SEGMENT[vertex_one][vertex_two]               # get segment index for getting bodypart length
            max_dist = self.bodypart_lengths[segment_index]                         # set max_dist to true length of given bodypart/segment
            #print("v1 %s, v2 %s, si %s" % (vertex_one, vertex_two, segment_index))  # DEBUG
            angle = self.angle_from_normal(cur_dist, max_dist)                      # calculate difference between max distance and current distance

            r = np.sin(angle) * max_dist                                         # calculate depth
            print(r)

            return r
        except:
            print("extrapolation.py: ERROR in `get_depth()`")

    # get y axes/depths by order of body parts
    def set_depth(self):
        try:
            # go thru 2d array of vertex order
            for i in range(0, len(self.vertex_order)):  # for each set of vertices denoting body segments
                print("vertices: " + str(i))
                for j in range(0, (len(self.vertex_order[i]) - 1)):   # for each vertex in set of vertices (except the last one)
                        print(self.vertex_order[i][j])
                        print(self.vertex_order[i][j + 1])
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
        except:
            print("extrapolation.py: ERROR in set_depth()")



    ### FORCES CALCULATIONS

    # calculate elbow angle
    def calc_elbow_angle(self, right_side = False):
        try:
            # coordinate data for shoulder, elbow, and wrist
            x = self.mediapipe_data_output[(0 + (int)(right_side)):(5 + (int)(right_side)):2, 0]
            y = self.mediapipe_data_output[(0 + (int)(right_side)):(5 + (int)(right_side)):2, 1]
            z = self.mediapipe_data_output[(0 + (int)(right_side)):(5 + (int)(right_side)):2, 2]

            # calculate vectors for getting angle at elbow
            vector_a = [(x[0] - x[1]), (y[0] - y[1]), (z[0] - z[1])]
            vector_b = [(x[2] - x[1]), (y[2] - y[1]), (z[2] - z[1])]

            # calculate length of arm segments via vector math
            forearm_length = np.sqrt( (vector_b[0] ** 2) + (vector_b[1] ** 2) + (vector_b[2] ** 2) )
            upperarm_length = np.sqrt( (vector_a[0] ** 2) + (vector_a[1] ** 2) + (vector_a[2] ** 2) )

            # calculate angle at elbow
            elbow_angle = np.arccos( ( (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1]) + (vector_a[2] * vector_b[2]) ) / (forearm_length * upperarm_length) )

            #print(elbow_angle)
            return elbow_angle
        except:
            print("extrapolation.py: ERROR in `calc_elbow_angle()`")

    # get spherical coordinates for each of the 3 vertices (bodyparts) of interest
    # vertex_one is the anchor point, and vertex_two is calculated based on its anchor
    def calc_spher_coords(self, vertex_one, vertex_two):    
        try:
            # effectively sets origin to cur_anchor
            x_diff = self.mediapipe_data_output[vertex_two][0] - self.mediapipe_data_output[vertex_one][0]
            y_diff = self.mediapipe_data_output[vertex_two][1] - self.mediapipe_data_output[vertex_one][1]
            z_diff = self.mediapipe_data_output[vertex_two][2] - self.mediapipe_data_output[vertex_one][2]

            #rho = np.sqrt((x_diff ** 2) + (y_diff ** 2) + (z_diff ** 2))
            print("test")
            rho = self.bodypart_lengths[VERTEX_TO_SEGMENT[vertex_one][vertex_two]]  # rho = true segment length
            print("%s", rho)
            theta = np.arctan(x_diff / y_diff)                                      # swapped due to equations having different Cartesian coordinate system layout
            print(theta)
            phi = np.arccos(z_diff / rho)
            print(phi)

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
                elbow_angle = self.calc_elbow_angle(is_right)
                if math.isnan(elbow_angle):
                    return math.nan                                         # if elbow_angle == nan, exit function by returning nan

                # convert sim units to metric units
                conv_factor = self.sim_to_real_conversion_factor

                # get spherical coordinate data for arm segments
                try:
                    uarm_spher_coords = self.calc_spher_coords((L_SHOULDER + (int)(is_right)), (L_ELBOW + (int)(is_right)))
                    farm_spher_coords = self.calc_spher_coords((L_ELBOW + (int)(is_right)), (L_WRIST + (int)(is_right)))
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
            calculated_data = {
                "right_bicep_force": str("%0.2f" % right_bicep_force),
                "right_elbow_angle": str("%0.2f" % np.rad2deg(right_elbow_angle)),
                "left_bicep_force": str("%0.2f" % left_bicep_force),
                "left_elbow_angle": str("%0.2f" % np.rad2deg(left_elbow_angle)),
                "uarm_spher_coords": str(uarm_spher_coords),
                "farm_spher_coords": str(farm_spher_coords)
            }
            #print("%0.2f" % force_bicep)

            return calculated_data
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

