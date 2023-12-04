# BD 2023
# This program is designed to reimplement code from a previous program for use in a new environment
# in order to extrapolate 3D motion tracking data from 2D motion tracking data and user input.
# This version has been edited for use directly with MediaPipe, as opposed to with FreeMoCap data output.


# TODO: 
#   - refactor code to work with one frame at a time
#       - basically, just turn mediapipe_data_output from mpdo[1][2][3] to mpdo[2][3]
#       - (the first one denotes the current frame)
#       - save frames over time for things like data analysis, BUT 
#           - only calculate one frame at a time
#               - otherwise you'd be doing a lot of unnecessary calculations, bogging down compute time massively
#   - set up for compatibility with mediapipe data
#   - optimize code
#       - minimize reads/writes
#           - try to do in-place manipulations of data


import numpy as np
from matplotlib import pyplot as plt


#### OBJECT FOR EASE OF MANAGEMENT OF EXTRAPOLATION OF DEPTH AND CALCULATION OF BODY FORCES
class Extrapolate_forces():
        
    # initialization
    def __init__(self) -> None:
        ### USER INPUT DATA

        self.user_height = 1.78      # user height (meters)
        self.user_weight = 90        # user weight (kilograms)



        ### IMPORTANT OBJECTS/VARIABLES

        # ndarray to store mediapipe data output, even if from other process(es)
        self.mediapipe_data_output = np.ndarray((1, 33, 3), dtype = "float64")

        # used for storing distance data (to prevent unnecessary recalculations)
        self.dist_array = np.zeros((1, 33, 33), dtype = "float64")         # indexed by two body part names/indices
        self.max_array = np.zeros((1, 33, 33), dtype = "float64")          # used for storing max distance data

        # spherical coordinates initialization
        self.rho = np.zeros((1, 33))
        self.theta = np.zeros((1, 33))
        self.phi = np.zeros((1, 33))


        self.cur_frame = 0   # used to keep track of current frame

        # help with readability and usability and identification of data vertices
        self.mediapipe_indices = ['nose',
        'left_eye_inner',
        'left_eye',
        'left_eye_outer',
        'right_eye_inner',
        'right_eye',
        'right_eye_outer',
        'left_ear',
        'right_ear',
        'mouth_left',
        'mouth_right',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_pinky',
        'right_pinky',
        'left_index',
        'right_index',
        'left_thumb',
        'right_thumb',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        'left_heel',
        'right_heel',
        'left_foot_index',
        'right_foot_index']

        # convert mediapipe units to real world units (meters)
        self.sim_to_real_conversion_factor = 0   # declared here for later use in functions

        # put together pairs for each of the vertices
        # ordered in a particular manner which uses the shoulders as anchors for the elbows, and elbows as anchors for the wrists
        self.vertex_order = [
            [
                'left_shoulder',
                'right_shoulder'
            ],
            [
                'left_shoulder',
                'left_elbow',
                'left_wrist',
            ],
            [
                'right_shoulder',
                'right_elbow',
                'right_wrist'
            ]
        ]


    # IMPORTANT: set mediapipe_data_output for the current frame
    def update_current_frame(self, mp_data_out, current_frame):
        # add data of current frame to dataset
        temp = np.zeros((1, 33, 3))                                                         # used for getting proper shape of ndarray to append
        temp[0, :, :] = mp_data_out                                                         # set only value of ndarray to mp_data_out
        self.mediapipe_data_output = np.append(self.mediapipe_data_output, temp, axis = 0)
        
        # add new frame to dist_array
        self.dist_array = np.append(self.dist_array, np.zeros(np.shape(self.dist_array)), axis = 0)   # temporarily hold previous frame's data as placeholder

        # update current frame number
        self.cur_frame = current_frame

    # used to account for the first frame being empty by default
    def update_first_frame(self):
        # remove first (empty) frame from datasets
        self.mediapipe_data_output = np.delete(self.mediapipe_data_output, 0, axis = 0)
        self.dist_array = np.delete(self.dist_array, 0, axis = 0)



    ### HELPER FUNCTIONS:

    # get indices for body parts
    def get_index(self, body_part = 'left_wrist'):
        return self.mediapipe_indices.index(body_part)

    # function to get data for particular body part
    #def get_bodypart_data(self, bodypart = "left_index"):
    #    return self.mediapipe_indices.index(bodypart)

    # get distance between vertices for current frame
    def cur_dist_between_vertices(self, first_part, second_part):
        # calculate distance for parts in current frame and add to dist_array
        self.dist_array[-1, first_part, second_part] = np.linalg.norm(
                                                            self.mediapipe_data_output[-1, first_part, :] - 
                                                            self.mediapipe_data_output[-1, second_part, :]
                                                        )

        return self.dist_array[-1, first_part, second_part]

    # get median of largest distances between vertices/bodyparts
    def calc_max_dist_between_parts(self, part_one, part_two):
        #num_frames = int(np.shape(freemocap_3d_body_data[:,0,0])[0] * 0.05) # relative to length of capture recording
        #ind = np.argpartition(dist_array, -num_frames)[-num_frames:-5]

        # make sure you're not using garbage data
        #if (num_frames < 50):
            
        #else:
        #    ind = np.argpartition(dist_array, -50)[0:-5]

        # return max distance between given parts
        return np.max(self.dist_array[:, part_one, part_two])

        #global dist_array                                                               # using global variable
        #max_array[1, part_one, part_two] = np.median(dist_array[:, part_one, part_two]) # store data to prevent unnecessary recalculation

    # return max distance between given parts
    #def get_max_dist_between_parts(part_one, part_two):
    #    return max_array[1, part_one, part_two]

    # reset dist array, for use when changing user and/or fixing tracking issues
    def reset_dist_array(self):
        #global dist_array   # using global variable
        self.dist_array = np.ndarray((1, 33, 33), dtype = "float64")



    ### CONVERSION FACTOR:

    # calculate ratio for conversion of simulated units to metric units (meters) using wingspan and input real height
    def calc_conversion_ratio(self, real_height_metric = 1.78):
        # get ratio to real distance in meters using max distance between wrists via mediapipe output data
        sim_wingspan = np.max(self.calc_max_dist_between_parts(
                                self.get_index("left_index"), 
                                self.get_index("right_index")
                            ))

        # set global conversion factor
        #global sim_to_real_conversion_factor
        self.sim_to_real_conversion_factor = real_height_metric / sim_wingspan

    #calc_conversion_ratio()     # unit conversion ratio for use converting sim units to metric

    # get conversion ratio (so it doesn't need to be calculated for each of these calls)
    def get_conversion_ratio(self):
        return self.sim_to_real_conversion_factor
        


    ### DEPTH:

    # calculate the angle of the segment (body part) from the normal (where it is longest)
    def angle_from_normal(self, cur_dist, max_dist):
        return np.arccos(cur_dist / max_dist)

    # get depth for body part in most recent frame
    def get_depth(self, vertex_one, vertex_two):
        cur_dist = self.cur_dist_between_vertices(vertex_one, vertex_two)    # current distance between given parts
        max_dist = self.calc_max_dist_between_parts(self.dist_array)              # max distance between given parts
        
        angle = self.angle_from_normal(cur_dist, max_dist)                   # calculate difference between max distance and current distance
        return np.sin(angle) * max_dist                                 # calculate depth

    # get y axes/depths by order of body parts
    def get_y_axes(self):
        y = list()
        for vertices in self.vertex_order:
            group_y = list()
            num_vertices = len(vertices)
            for i, vertex in enumerate(vertices):
                if i < (num_vertices - 1):
                    y_dist_between_vertices = np.nan_to_num(self.get_depth(self.get_index(vertices[i]), self.get_index(vertices[i + 1])))
                    if i > 0:
                        vertex_y = group_y[i - 1] +  y_dist_between_vertices    # add y depth of anchor to current
                    else:
                        vertex_y = y_dist_between_vertices
                    group_y.append(vertex_y)
            y.append(group_y)
        return y

    # get indices for body parts used
    #def get_indices():
    #    indices = list()
    #    for vertices in depth_dict['vertex_order']:
    #        for vertex in vertices:
    #            indices.append(get_index(vertex))
    #    return indices

    #indices = get_indices()

    # approximate depth for the current frame
    def set_depth(self, depth_dict):
        print("a")
        # get y axes
        y_axes = self.get_y_axes()
        print("b")
        # account for difference between shoulder y-axes
        y_axes[2] += y_axes[0]  # by adding it to the branch off the right shoulder

        print("c")
        # put together dictionary to coordinate vertex pairs and y-axes coordinates (calculated depth)
        depth_dict = {
            'vertex_order': self.vertex_order,       # pairs of body parts/segments
            'y_axes': y_axes,                   # approximated depth
        }
        #global mediapipe_data_output                                    # allow use of global variable
        # go through and set y-axes values accordingly
        print("d")
        for i, order_group in enumerate(depth_dict['y_axes']):
            cur_length = len(depth_dict['vertex_order'][i])
            # go thru all vertices in current group
            for j, vertex in enumerate(order_group):
                if j < (cur_length - 1):
                    print("e")
                    # set y axis for each vertex in the order group
                    cur_vertex = depth_dict['vertex_order'][i][j + 1]   # + 1 so that it applies to the non-anchor vertex
                    print("f")
                    vertex_index = self.mediapipe_indices.index(cur_vertex)
                    print("g")
                    self.mediapipe_data_output[-1, vertex_index, 1] = vertex # set depth directly in dataset
                    print("h")



    ### FORCES CALCULATIONS

    # calculate elbow angle
    def set_elbow_angle(self):
        #global mediapipe_data_output                                # allow manipulation of global variable
        # coordinate data for left shoulder, elbow, and wrist
        x = self.mediapipe_data_output[-1, 11:17:2, 0] 
        y = self.mediapipe_data_output[-1, 11:17:2, 1]
        z = self.mediapipe_data_output[-1, 11:17:2, 2]

        # calculate vectors for getting angle at elbow
        vector_a = [(x[-1, 0] - x[-1, 1]), (y[-1, 0] - y[-1, 1]), (z[-1, 0] - z[-1, 1])]
        vector_b = [(x[-1, 2] - x[-1, 1]), (y[-1, 2] - y[-1, 1]), (z[-1, 2] - z[-1, 1])]

        # calculate length of arm segments via vector math
        forearm_length = np.sqrt( (vector_b[0] ** 2) + (vector_b[1] ** 2) + (vector_b[2] ** 2) )
        upperarm_length = np.sqrt( (vector_a[0] ** 2) + (vector_a[1] ** 2) + (vector_a[2] ** 2) )

        # calculate angle at elbow
        elbow_angle = np.arccos( ( (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1]) + (vector_a[2] * vector_b[2]) ) / (forearm_length * upperarm_length) )

        # Now put it in the data matrix for display by plotly

        self.mediapipe_data_output[-1, 30, :] = np.swapaxes(vector_a, 0, 1)  # swapped axes due to shape of 2D array
        self.mediapipe_data_output[-1, 31, :] = np.swapaxes(vector_b, 0, 1)

        self.mediapipe_data_output[-1, 32, 0] = forearm_length
        self.mediapipe_data_output[-1, 32, 1] = upperarm_length
        self.mediapipe_data_output[-1, 32, 2] = elbow_angle

        return elbow_angle

    # get spherical coordinates for each of the 3 vertices (bodyparts) of interest
    def set_spher_coords(self):
        # append new empty data for current frame
        self.rho = np.append(self.rho, np.zeros((1, 33)), axis = 0)
        self.theta = np.append(self.theta, np.zeros((1, 33)), axis = 0)
        self.phi = np.append(self.phi, np.zeros((1, 33)), axis = 0)

        # set for [elbow, wrist]    using difference between current point and shoulder point
        for vertex in [1, 2]:       # using shoulder as origin, running for elbow (1) and wrist (2)
            if vertex == 1:         # if elbow (from shoulder to elbow)
                cur_shoulder = self.mediapipe_data_output[-1, 11, :]    # shoulder
            elif vertex == 2:       # if wrist (from elbow to wrist)
                cur_shoulder = self.mediapipe_data_output[-1, 13, :]    # elbow
            cur_bodypart = self.mediapipe_data_output[-1, (11 + (vertex * 2) ), :]
            
            # effectively sets origin to cur_shoulder
            x_diff = cur_bodypart[-1, 0] - cur_shoulder[-1, 0]
            y_diff = cur_bodypart[-1, 1] - cur_shoulder[-1, 1]
            z_diff = cur_bodypart[-1, 2] - cur_shoulder[-1, 2]

            self.rho[-1, vertex] = np.sqrt((x_diff ** 2) + (y_diff ** 2) + (z_diff ** 2))
            self.theta[-1, vertex] = np.arctan(y_diff / x_diff)         # swapped due to equations having different Cartesian coordinate system layout
            self.phi[-1, vertex] = np.arccos(z_diff / self.rho[-1, vertex])

    ### FORMULA CALCULATIONS
    # calculate forces of muscle exertions of the left arm
    def run_formula_calculations(self):
        #h_p = self.user_height   # meters      # height of person
        #w_p = self.user_weight   # kilograms   # weight of person
        w_bal = 3           # kilograms   # weight of ball

        # convert sim units to metric units
        conv_factor = self.get_conversion_ratio()

        # convert rho from simulated units to metric units
        #metric_rho = self.rho * conv_factor

        # equations used primarily from the paper labeled "shoulderarm3.pdf" in the dropbox, as well as some info from emails between Dr. Liu and I (Bri)
        #w_fa = w_p * 0.023                      # weight of forearm
        #cgf = h_p * 0.432 * 0.216               # center of gravity of forearm
        #f = h_p * 0.216                         # length of forearm
        #b = f * 0.11                            # dist between elbow and bicep insertion point
        #u = h_p * 0.173                         # length of upper arm

        # instead of using averages for segment length, use calculated instead
        f = self.rho[-1, 2] * conv_factor
        u = self.rho[-1, 1] * conv_factor
        b = u * 0.11 #0.636                   # calculated via algebra using pre-existing average proportions data
        w_fa = self.user_weight * (f * 0.1065)       # use ratio of f to weight proportion to get weight with calculated f 
        cgf = 2 * (f ** 2)                     # calculated via algebra using pre-existing average proportions data

        # angles
        theta_arm = (np.pi / 2) - self.phi[-1, 1]                         # angle at shoulder
        theta_uarm = (np.pi / 2) + self.phi[-1, 2]                        # angle of upper arm
        theta_u = self.set_elbow_angle()  #theta_arm + theta_uarm                            # angle at elbow
        theta_b = np.pi - ( (b - u * np.sin(theta_u)) / np.sqrt( (b ** 2) + (u ** 2) - 2 * b * u * np.sin(theta_u) ) )      # angle at bicep insertion point
        theta_la = np.cos(theta_uarm)   #theta_u - theta_arm - np.pi) #np.sin(theta_uarm)        # angle used for leverage arms fa and bal

        # lever arms
        la_fa = cgf * theta_la                                      # forearm lever arm
        la_bal = f * theta_la                                       # ball lever arm
        la_bic = b * np.sin(theta_b)                                # bicep lever arm

        # forces
        force_bicep = (w_fa * la_fa + w_bal * la_bal) / la_bic      # force applied by bicep muscle


        # save calculations in a way to help prep for visual in plotly output

        # theta-arm, bicep force, theta-u
        self.mediapipe_data_output[-1, 26, 0] = np.rad2deg(theta_arm)
        self.mediapipe_data_output[-1, 26, 1] = force_bicep 
        self.mediapipe_data_output[-1, 26, 2] = np.rad2deg(theta_u)

        # rho, theta, phi
        self.mediapipe_data_output[-1, 27, :] = self.rho
        self.mediapipe_data_output[-1, 28, :] = np.rad2deg(self.theta) 
        self.mediapipe_data_output[-1, 29, :] = np.rad2deg(self.phi)

    #run_formula_calculations()



    ### BICEP FORCES GRAPH

    # graph bicep forces
    def plot_bicep_forces(self):#body_data = mediapipe_data_output):
        # plot bicep force / theta_u
        y = np.abs(self.mediapipe_data_output[:, 26, 1])             # bicep force
        x = np.rad2deg(self.mediapipe_data_output[:, 32, 2])#freemocap_3d_body_data[:, 26, 2]))                     # angle at elbow

        # plot
        plt.scatter(x,y)
        #plt.scatter(range(0, np.shape(freemocap_3d_body_data[:,0,0])[0] ), y)
        plt.ylim([0, 1000])     # restrict y axis to between 0 and 1000 (Newtons)
        plt.xlim([0, 180])      # restrixt x axis to between 0 and 180 (degrees)

        # plot bicep forces
        return plt

    # plot forces graph
    #plot_bicep_forces().show()

