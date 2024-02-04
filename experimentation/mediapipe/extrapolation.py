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


import numpy as np
import math
from matplotlib import pyplot as plt



# set to not display in scientific notation
np.set_printoptions(suppress = True, precision = 3)

#### CONSTANTS (for use with indexing)
L_SHOULDER = 0#11
R_SHOULDER = 1#12
L_ELBOW = 2#13
R_ELBOW = 3#14
L_WRIST = 4#15
R_WRIST = 5#16
L_INDEX = 6#19
R_INDEX = 7#20


#### OBJECT FOR EASE OF MANAGEMENT OF EXTRAPOLATION OF DEPTH AND CALCULATION OF BODY FORCES
class Extrapolate_forces():
        
    # initialization
    def __init__(self, right = False, one_arm = False) -> None:
        ### USER INPUT DATA

        self.user_height = 1.78     # user height (meters)
        self.user_weight = 90       # user weight (kilograms)
        self.ball_mass = 3          # mass of ball (kilograms)
        
        # toggle for calculating left arm or right arm
        self.is_right = right
        self.is_one_arm = one_arm

        # calibration settings
        self.manual_calibration = False
        self.sim_to_real_conversion_factor = 1  # convert mediapipe units to real world units (meters)
        self.use_full_wingspan = False
        self.biacromial_scale = 0.23              # temporarily set to middle of male (0.234) to female (0.227) range for testing

        # ndarray to store mediapipe data output, even if from other process(es)
        self.mediapipe_data_output = np.ndarray((8, 3), dtype = "float64")

        # used for storing distance data (to prevent unnecessary recalculations)
        self.dist_array = np.zeros((8, 8), dtype = "float64")         # indexed by two body part names/indices
        self.max_array = np.zeros((8, 8), dtype = "float64")          # used for storing max distance data

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

        print("extrapolation.py: Info: Initialized extrapolation.py")


    # IMPORTANT: set mediapipe_data_output for the current frame
    def update_current_frame(self, mp_data_out, current_frame):
        # set data of current frame dataset
        self.mediapipe_data_output = mp_data_out
        
        # reset dist_array
        self.dist_array = np.zeros(np.shape(self.dist_array))

        # update current frame number
        self.cur_frame = current_frame

        # update calibration settings
        if self.use_full_wingspan and not self.is_one_arm:
            self.calc_wingspan()                            # keep track of max distance between index fingers
        else:
            self.calc_half_wingspan()                       # keep track of max length of given arm

    # IMPORTANT: temporary bandaid fix for calibration
    def calc_wingspan(self):
        self.calc_dist_between_vertices(L_INDEX, R_INDEX)

    # track max dist between half wingspan for calibration (automatically done via calc_dist_between_vertices, updating max_dist)
    def calc_half_wingspan(self):
        # keep track of arm length
        self.calc_dist_between_vertices((L_INDEX + (int)(self.is_right)), (L_SHOULDER + (int)(self.is_right)))
        # keep track of shoulder width
        self.calc_dist_between_vertices(L_SHOULDER, R_SHOULDER)



    ### HELPER FUNCTIONS:

    # set height and weight and ball mass externally
    def set_hwb(self, height, weight, ball):
        self.user_height = float(height)
        self.user_weight = float(weight)
        self.ball_mass = float(ball)

    # set biacromial ratio externally
    def set_biacromial(self, new_biacromial = 0.23):
        self.biacromial_scale = new_biacromial



    ### DISTANCE FUNCTIONS:

    # get distance between vertices for current frame
    def calc_dist_between_vertices(self, first_part, second_part):
        # calculate distance for parts in current frame
        dist = np.linalg.norm(
                        self.mediapipe_data_output[first_part, :] - 
                        self.mediapipe_data_output[second_part, :]
                    )
        
        # update max distance between these parts, if necessary
        if dist > self.max_array[first_part][second_part]:
            self.max_array[first_part][second_part] = dist

        return dist

    # retrieve the max distance between body parts found thus far
    def get_max_dist(self, first_part, second_part):
        return float(self.max_array[first_part][second_part])



    ### CALIBRATION CONVERSION FACTOR:

    # calculate ratio for conversion of simulated units to metric units (meters) using wingspan and input real height
    # using the wingspan method
    def calc_conversion_ratio_wingspan(self, real_height_metric = 1.78):
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

    # calculate conversion ratio using the shoulder width method
    def calc_conversion_ratio(self, real_height_metric = 1.78):
        # get maximum distance between shoulders
        sim_biacromial = self.get_max_dist(L_SHOULDER, R_SHOULDER)
        self.sim_to_real_conversion_factor = (real_height_metric * self.biacromial_scale) / sim_biacromial
    
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
        return np.arccos(cur_dist / max_dist)

    # get depth for body part in most recent frame
    def get_depth(self, vertex_one, vertex_two):
        cur_dist = self.calc_dist_between_vertices(vertex_one, vertex_two)      # current distance between given parts
        max_dist = self.get_max_dist(vertex_one, vertex_two)                    # max distance between given parts
        
        angle = self.angle_from_normal(cur_dist, max_dist)                      # calculate difference between max distance and current distance
        return np.sin(angle) * max_dist                                         # calculate depth

    # get y axes/depths by order of body parts
    def set_depth(self):
        for vertices in self.vertex_order:
            #print(vertices)
            for i in enumerate(vertices):
                #print(i[1])
                if i[1] != (vertices[-1]):
                    y_dist_between_vertices = self.get_depth(i[1], i[1] + 1)          # calculate depth
                    # check if "nan" value
                    if math.isnan(y_dist_between_vertices):
                        y_dist_between_vertices = 0                             # set all nan values to 0
                    # add previous anchor vertex
                    if i[1] > 0:       # if i is not left shoulder
                        vertex_y = self.mediapipe_data_output[i[1] - 1][1] +  y_dist_between_vertices      # add y depth of anchor (previous node) to current
                    else:
                        vertex_y = y_dist_between_vertices
                    self.mediapipe_data_output[i[1], 1] = vertex_y



    ### FORCES CALCULATIONS

    # calculate elbow angle
    def calc_elbow_angle(self, right_side = False):
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

    # get spherical coordinates for each of the 3 vertices (bodyparts) of interest
    # vertex_one is the anchor point, and vertex_two is calculated based on its anchor
    def calc_spher_coords(self, vertex_one, vertex_two):    
        # effectively sets origin to cur_anchor
        x_diff = self.mediapipe_data_output[vertex_two][0] - self.mediapipe_data_output[vertex_one][0]
        y_diff = self.mediapipe_data_output[vertex_two][1] - self.mediapipe_data_output[vertex_one][1]
        z_diff = self.mediapipe_data_output[vertex_two][2] - self.mediapipe_data_output[vertex_one][2]

        rho = np.sqrt((x_diff ** 2) + (y_diff ** 2) + (z_diff ** 2))
        theta = np.arctan(y_diff / x_diff)         # swapped due to equations having different Cartesian coordinate system layout
        phi = np.arccos(z_diff / rho)

        return [rho, theta, phi]



    ### FORMULA CALCULATIONS

    # calculate forces of muscle exertions of the arm
    def calc_bicep_force(self):#, is_right = False):   # allow choosing which arm to use
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
            uarm_spher_coords = self.calc_spher_coords((L_SHOULDER + (int)(is_right)), (L_ELBOW + (int)(is_right)))
            farm_spher_coords = self.calc_spher_coords((L_ELBOW + (int)(is_right)), (L_WRIST + (int)(is_right)))

            # instead of using averages for segment length, use calculated instead
            f = farm_spher_coords[RHO] * conv_factor
            u = uarm_spher_coords[RHO] * conv_factor
            #print("%0.2f" % f)
            b = u * 0.11 #0.636                                         # calculated via algebra using pre-existing average proportions data
            w_fa = self.user_weight * (f * 0.1065)                      # use ratio of f to weight proportion to get weight with calculated f 
            cgf = 2 * (f ** 2)                                          # calculated via algebra using pre-existing average proportions data

            # angles
            #theta_arm = (np.pi / 2) - farm_spher_coords[THETA]          # angle at shoulder
            theta_uarm = (np.pi / 2) + uarm_spher_coords[THETA]         # angle of upper arm
            theta_u = elbow_angle#theta_arm + theta_uarm                # angle at elbow
            theta_b = np.pi - ( (b - u * np.sin(theta_u)) / np.sqrt( (b ** 2) + (u ** 2) - 2 * b * u * np.sin(theta_u) ) )      # angle at bicep insertion point
            theta_la = np.cos(theta_uarm) #theta_u - theta_arm - np.pi) #np.sin(theta_uarm)        # angle used for leverage arms fa and bal

            # lever arms
            la_fa = cgf * theta_la                                      # forearm lever arm
            la_bal = f * theta_la                                       # ball lever arm
            la_bic = b * np.sin(theta_b)                                # bicep lever arm

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



    ### BICEP FORCES GRAPH

    # graph bicep forces
    def plot_bicep_forces(self, bicep_force, elbow_angle):
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

