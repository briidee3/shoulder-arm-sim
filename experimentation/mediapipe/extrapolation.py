# BD 2023
# This program is designed to reimplement code from a previous program for use in a new environment
# in order to extrapolate 3D motion tracking data from 2D motion tracking data and user input.


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


# placeholder ndarray to represent mediapipe data output
mediapipe_data_output = np.ndarray((1, 33, 3), dtype = "float64")  #freemocap_3d_body_data  # "freemocap_3d_body_data" is from previous iteration; used here for temporary clarification


# IMPORTANT: set mediapipe_data_output for the current frame
def update_current_frame(mp_data_out):
    mediapipe_data_output = mp_data_out


# help with readability and usability and identification of data vertices
mediapipe_indices = ['nose',
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


### HELPER FUNCTIONS:

# get indices for body parts
def get_index(body_part = 'left_wrist'):
    return mediapipe_indices.index(body_part)

# function to get data for particular body part
def get_bodypart_data(bodypart = "left_index"):
    
    joint_to_plot_index = mediapipe_indices.index(bodypart)

    return mediapipe_data_output[:,joint_to_plot_index,:]

# get distance between two body parts
def dist_between_vertices(first_part, second_part):     # parameters are the data arrays for the body parts
    cur_dist = np.ndarray(shape = first_part.shape, dtype = first_part.dtype)
    for i in range(0, len(first_part) - 1):     # get the distance for each frame of the video (excluding last frame)
        cur_dist[i] = np.linalg.norm(first_part[i] - second_part[i])
        #print(cur_dist[i])]
    return cur_dist[:-1, 0]     # returns array of distances between parts

# get median of largest distances between vertices/bodyparts
def max_dist_between_parts(dist_array):
    #num_frames = int(np.shape(freemocap_3d_body_data[:,0,0])[0] * 0.05) # relative to length of capture recording
    #ind = np.argpartition(dist_array, -num_frames)[-num_frames:-5]
    ind = np.argpartition(dist_array, -50)[-50:-5]
    return np.median(dist_array[ind])



### CONVERSION FACTOR:

# calculate ratio for conversion of simulated units to metric units (meters) using wingspan and input real height
def calc_conversion_ratio(real_height_metric = 1.78):
    # get ratio to real distance in meters
    sim_wingspan = np.max(dist_between_vertices(get_bodypart_data("left_index"), get_bodypart_data("right_index")))     # max distance between wrists via `freemocap`

    return real_height_metric / sim_wingspan

#sim_to_real_conversion_factor = calc_conversion_ratio()     # unit conversion ratio for use converting sim units to metric



### DEPTH:

# put together pairs for each of the vertices
# ordered in a particular manner which uses the shoulders as anchors for the elbows, and elbows as anchors for the wrists
vertex_order = [
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

# calculate the angle of the segment (body part) from the normal (where it is longest)
def angle_from_normal(cur_dist, max_dist):
    return np.arccos(cur_dist / max_dist)

# get depth
def get_depth(vertex_one, vertex_two):
    dist_array = dist_between_vertices(vertex_one, vertex_two)
    max_dist = max_dist_between_parts(dist_array)
    depths = list()
    for frame in dist_array:
        angle = angle_from_normal(frame, max_dist)
        depths.append(np.sin(angle) * max_dist)
    return depths


# get y axes by order of body parts
def get_y_axes(vertex_order = vertex_order):
    y = list()
    for vertices in vertex_order:
        group_y = list()
        num_vertices = len(vertices)
        for i, vertex in enumerate(vertices):
            if i < (num_vertices - 1):
                y_dist_between_vertices = np.nan_to_num(get_depth(get_bodypart_data(vertices[i]), get_bodypart_data(vertices[i + 1])))
                if i > 0:
                    vertex_y = group_y[i - 1] +  y_dist_between_vertices    # add y depth of anchor to current
                else:
                    vertex_y = y_dist_between_vertices
                group_y.append(vertex_y)
        y.append(group_y)
    return y


# get y axes
y_axes = get_y_axes()
# account for difference between shoulder y-axes
y_axes[2] += y_axes[0]  # by adding it to the branch off the right shoulder

# put together dictionary to coordinate vertex pairs and y-axes coordinates (calculated depth)
depth_dict = {
    'vertex_order': vertex_order,       # pairs of body parts/segments
    'y_axes': y_axes,                   # approximated depth
}

# function to work with depth_dict externally
def set_depth_dict(y_ax = y_axes):
    depth_dict['y_axes'] = y_ax

# get indices for body parts used
def get_indices():
    indices = list()
    for vertices in depth_dict['vertex_order']:
        for vertex in vertices:
            indices.append(get_index(vertex))
    return indices

#indices = get_indices()


# approximate depth for the video recording
def set_depth(depth_dict = depth_dict, body_data = mediapipe_data_output):
    depth_body_data = body_data[:, :, 1]    # y axis of each part on each frame

    # go through and set y-axes values accordingly
    for i, order_group in enumerate(depth_dict['y_axes']):
        cur_length = len(depth_dict['vertex_order'][i])
        # go thru all vertices in current group
        for j, vertex in enumerate(order_group):
            if j < (cur_length - 1):
                # set y axis for each vertex in the order group
                cur_vertex = depth_dict['vertex_order'][i][j + 1]   # + 1 so that it applies to the non-anchor vertex
                vertex_index = mediapipe_indices.index(cur_vertex)
                body_data[:, vertex_index, 1] = np.append(vertex, 0)

    #body_data[:, :, 1] = depth_body_data
    mediapipe_data_output[:, :, 1] = depth_body_data
    #return body_data

# set the depth for all body parts:
mediapipe_data_output = set_depth(body_data = mediapipe_data_output)

x_shape = mediapipe_data_output[:, 11:17:2, 0]  # tmp used for getting shape

### ELBOW ANGLE:
def set_elbow_angle():
    # coordinate data for left shoulder, elbow, and wrist
    x = mediapipe_data_output[:, 11:17:2, 0] 
    y = mediapipe_data_output[:, 11:17:2, 1]
    z = mediapipe_data_output[:, 11:17:2, 2]

    # calculate vectors for getting angle at elbow
    vector_a = [(x[:, 0] - x[:, 1]), (y[:, 0] - y[:, 1]), (z[:, 0] - z[:, 1])]
    vector_b = [(x[:, 2] - x[:, 1]), (y[:, 2] - y[:, 1]), (z[:, 2] - z[:, 1])]

    # calculate length of arm segments via vector math
    forearm_length = np.sqrt( (vector_b[0] ** 2) + (vector_b[1] ** 2) + (vector_b[2] ** 2) )
    upperarm_length = np.sqrt( (vector_a[0] ** 2) + (vector_a[1] ** 2) + (vector_a[2] ** 2) )

    # calculate angle at elbow
    elbow_angle = np.arccos( ( (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1]) + (vector_a[2] * vector_b[2]) ) / (forearm_length * upperarm_length) )

    # Now put it in the data matrix for display by plotly

    mediapipe_data_output[:, 30, :] = np.swapaxes(vector_a, 0, 1)  # swapped axes due to shape of 2D array
    mediapipe_data_output[:, 31, :] = np.swapaxes(vector_b, 0, 1)

    mediapipe_data_output[:, 32, 0] = forearm_length
    mediapipe_data_output[:, 32, 1] = upperarm_length
    mediapipe_data_output[:, 32, 2] = elbow_angle



### CONVERSION TO SPHERICAL COORDINATE SYSTEM

# get spherical coordinates for each of the 3 vertices (bodyparts) of interest, 
#   and set them to overwrite parts 27-29 of freemocap_3d_body_data for displaying with `plotly`
def set_spher_coords():#mp_data_out = mediapipe_data_output):

    # initialize to x.shaped ndarray since x is conveniently the same shape we want
    rho = np.zeros(x_shape)
    theta = np.zeros(x_shape)
    phi = np.zeros(x_shape)

    # set for [elbow, wrist]    using difference between current point and shoulder point
    for vertex in [1, 2]:       # using shoulder as origin, running for elbow (1) and wrist (2)
        if vertex == 1:     # if elbow (from shoulder to elbow)
            cur_shoulder = mediapipe_data_output[:, 11, :]     # shoulder
        elif vertex == 2:   # if wrist (from elbow to wrist)
            cur_shoulder = mediapipe_data_output[:, 13, :]     # elbow
        cur_bodypart = mediapipe_data_output[:, (11 + (vertex * 2) ), :]
        
        # effectively sets origin to cur_shoulder
        x_diff = cur_bodypart[:, 0] - cur_shoulder[:, 0]
        y_diff = cur_bodypart[:, 1] - cur_shoulder[:, 1]
        z_diff = cur_bodypart[:, 2] - cur_shoulder[:, 2]

        rho[:, vertex] = np.sqrt((x_diff ** 2) + (y_diff ** 2) + (z_diff ** 2))
        theta[:, vertex] = np.arctan(y_diff / x_diff)   # swapped due to equations having different Cartesian coordinate system layout
        phi[:, vertex] = np.arccos(z_diff / rho[:, vertex])

    # put spherical coords in bodydata matrix for displaying in the model
    mediapipe_data_output[:, 27, :] = rho
    mediapipe_data_output[:, 28, :] = theta    # 
    mediapipe_data_output[:, 29, :] = phi      # z / rho^2
        # the data at the parts of the freemocap_3d_body_data tensor used here are not actual for body parts (anymore), but places to hold information
        # for simplification for displaying via plotly, which is used by freemocap, hence its continued use here

    #return rho, theta, phi

# get spherical coordinates for use later on
#rho, theta, phi = get_spher_coords()



### FORCES CALCULATIONS

# calculate forces of muscle exertions of the left arm
def run_formula_calculations():
    h_p = 1.75    # meters      # height of person
    w_p = 90      # kilograms   # weight of person
    w_bal = 3     # kilograms   # weight of ball

    # convert sim units to metric units
    conv_factor = calc_conversion_ratio(h_p)

    # convert rho from simulated units to metric units
    rho = mediapipe_data_output[:, 27, :] * conv_factor

    # equations used primarily from the paper labeled "shoulderarm3.pdf" in the dropbox, as well as some info from emails between Dr. Liu and I (Bri)
    #w_fa = w_p * 0.023                      # weight of forearm
    #cgf = h_p * 0.432 * 0.216               # center of gravity of forearm
    #f = h_p * 0.216                         # length of forearm
    #b = f * 0.11                          # dist between elbow and bicep insertion point
    #u = h_p * 0.173                         # length of upper arm

    # instead of using averages for segment length, use calculated instead
    f = rho[:, 2] * conv_factor
    u = rho[:, 1] * conv_factor
    b = u * 0.11 #0.636                   # calculated via algebra using pre-existing average proportions data
    w_fa = w_p * (f * 0.1065)       # use ratio of f to weight proportion to get weight with calculated f 
    cgf = 2 * (f ** 2)                     # calculated via algebra using pre-existing average proportions data

    # angles
    theta_arm = (np.pi / 2) - phi[:, 1]                         # angle at shoulder
    theta_uarm = (np.pi / 2) + phi[:, 2]                        # angle of upper arm
    theta_u = elbow_angle  #theta_arm + theta_uarm                            # angle at elbow
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
    mediapipe_data_output[:, 26, 0] = np.rad2deg(theta_arm)
    mediapipe_data_output[:, 26, 1] = force_bicep 
    mediapipe_data_output[:, 26, 2] = np.rad2deg(theta_u)

    # rho, theta, phi
    mediapipe_data_output[:, 27, :] = rho
    mediapipe_data_output[:, 28, :] = np.rad2deg(theta) 
    mediapipe_data_output[:, 29, :] = np.rad2deg(phi)

run_formula_calculations()



### BICEP FORCES GRAPH

# graph bicep forces
def plot_bicep_forces():#body_data = mediapipe_data_output):
    # plot bicep force / theta_u
    y = np.abs(mediapipe_data_output[:, 26, 1])             # bicep force
    x = np.rad2deg(elbow_angle)#freemocap_3d_body_data[:, 26, 2]))                     # angle at elbow

    # plot
    plt.scatter(x,y)
    #plt.scatter(range(0, np.shape(freemocap_3d_body_data[:,0,0])[0] ), y)
    plt.ylim([0, 1000])     # restrict y axis to between 0 and 1000 (Newtons)
    plt.xlim([0, 180])      # restrixt x axis to between 0 and 180 (degrees)

    # plot bicep forces
    return plt

# plot forces graph
plot_bicep_forces().show()

