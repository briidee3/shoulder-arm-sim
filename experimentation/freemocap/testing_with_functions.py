# Brianna D'Urso 2023
# 3 Dimensional motion capture from webcam video based on the default output from Freemocap motion capture software
# Created for use simulating body forces for elementary physics demonstrations research project with Dr. Dan Liu and Dylan StJames


import numpy as np
import os
from matplotlib import pyplot as plt


# freemocap default imports
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pathlib import Path


# folder containing Freemocap recording to be read from
data_folder = "recording_19_56_30_gmt-4"    # recording_19_51_41_gmt-4  


# initialize dataset and helper data arrays
# the following 40-50 (or so) lines taken from the default output .ipynb from Freemocap (mostly without custom changes)
# local folder with freemocap recording files located within

path_to_recording = os.path.join(os.getcwd(), data_folder)

bodypart = 'left_index'
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

joint_to_plot_index = mediapipe_indices.index(bodypart)


path_to_recording = Path(path_to_recording)
path_to_center_of_mass_npy = path_to_recording/'output_data'/'center_of_mass'/'total_body_center_of_mass_xyz.npy'
path_to_freemocap_3d_body_data_npy = path_to_recording/'output_data'/'mediapipe_body_3d_xyz.npy'

freemocap_3d_body_data = np.load(path_to_freemocap_3d_body_data_npy)
total_body_com_data = np.load(path_to_center_of_mass_npy)



### HELPER FUNCTIONS:

# get indices for body parts
def get_index(body_part = 'left_wrist'):
    return mediapipe_indices.index(body_part)

# function to get data for particular body part
def get_bodypart_data(bodypart = "left_index"):
    
    joint_to_plot_index = mediapipe_indices.index(bodypart)

    return freemocap_3d_body_data[:,joint_to_plot_index,:]

# get distance between two body parts
def dist_between_vertices(first_part, second_part):     # parameters are the data arrays for the body parts
    cur_dist = np.ndarray(shape = first_part.shape, dtype = first_part.dtype)
    for i in range(0, len(first_part) - 1):     # get the distance for each frame of the video (excluding last frame)
        cur_dist[i] = np.linalg.norm(first_part[i] - second_part[i])
        #print(cur_dist[i])]
    return cur_dist[:-1, 0]     # returns array of distances between parts

# get median of largest distances between vertices/bodyparts
def max_dist_between_parts(dist_array):
    ind = np.argpartition(dist_array, -30)[-30:]
    return np.median(dist_array[ind])



### CONVERSION FACTOR:

# calculate ratio for conversion of simulated units to metric units (meters) using wingspan and input real height
def calc_conversion_ratio(real_height_metric = 1.78):
    # get ratio to real distance in meters
    real_height_metric = 1.78     # meters
    sim_wingspan = np.max(dist_between_vertices(get_bodypart_data("left_index"), get_bodypart_data("right_index")))     # max distance between wrists via `freemocap`

    return real_height_metric / sim_wingspan

sim_to_real_conversion_factor = calc_conversion_ratio()     # unit conversion ratio for use converting sim units to metric



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

# get indices for body parts used
indices = list()
for vertices in depth_dict['vertex_order']:
    for vertex in vertices:
        indices.append(get_index(vertex))


# approximate depth for the video recording
def set_depth(depth_dict = depth_dict, body_data = freemocap_3d_body_data):
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

    body_data[:, :, 1] = depth_body_data
    return body_data

# set the depth for all body parts:
freemocap_3d_body_data = set_depth(body_data = freemocap_3d_body_data)



### ELBOW ANGLE:

# coordinate data for left shoulder, elbow, and wrist
x = freemocap_3d_body_data[:, 11:17:2, 0] 
y = freemocap_3d_body_data[:, 11:17:2, 1]
z = freemocap_3d_body_data[:, 11:17:2, 2]

# calculate vectors for getting angle at elbow
vector_a = [(x[:, 0] - x[:, 1]), (y[:, 0] - y[:, 1]), (z[:, 0] - z[:, 1])]
vector_b = [(x[:, 2] - x[:, 1]), (y[:, 2] - y[:, 1]), (z[:, 2] - z[:, 1])]

# calculate length of arm segments via vector math
forearm_length = np.sqrt( (vector_b[0] ** 2) + (vector_b[1] ** 2) + (vector_b[2] ** 2) )
upperarm_length = np.sqrt( (vector_a[0] ** 2) + (vector_a[1] ** 2) + (vector_a[2] ** 2) )

# calculate angle at elbow
elbow_angle = np.arccos( ( (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1]) + (vector_a[2] * vector_b[2]) ) / (forearm_length * upperarm_length) )

# Now put it in the data matrix for display by plotly

freemocap_3d_body_data[:, 30, :] = np.swapaxes(vector_a, 0, 1)  # swapped axes due to shape of 2D array
freemocap_3d_body_data[:, 31, :] = np.swapaxes(vector_b, 0, 1)

freemocap_3d_body_data[:, 32, 0] = forearm_length
freemocap_3d_body_data[:, 32, 1] = upperarm_length
freemocap_3d_body_data[:, 32, 2] = elbow_angle



### CONVERSION TO SPHERICAL COORDINATE SYSTEM

# get spherical coordinates for each of the 3 vertices (bodyparts) of interest, 
#   and set them to overwrite parts 27-29 of freemocap_3d_body_data for displaying with `plotly`
def get_spher_coords(freemocap_data_out = freemocap_3d_body_data):

    # initialize to x.shaped ndarray since x is conveniently the same shape we want
    rho = np.zeros(x.shape)
    theta = np.zeros(x.shape)
    phi = np.zeros(x.shape)

    # set for [elbow, wrist]    using difference between current point and shoulder point
    for vertex in [1, 2]:       # using shoulder as origin, running for elbow (1) and wrist (2)
        if vertex == 1:     # if elbow (from shoulder to elbow)
            cur_shoulder = freemocap_data_out[:, 11, :]     # shoulder
        elif vertex == 2:   # if wrist (from elbow to wrist)
            cur_shoulder = freemocap_data_out[:, 13, :]     # elbow
        cur_bodypart = freemocap_data_out[:, (11 + (vertex * 2) ), :]
        
        # effectively sets origin to cur_shoulder
        x_diff = cur_bodypart[:, 0] - cur_shoulder[:, 0]
        y_diff = cur_bodypart[:, 1] - cur_shoulder[:, 1]
        z_diff = cur_bodypart[:, 2] - cur_shoulder[:, 2]

        rho[:, vertex] = np.sqrt((x_diff ** 2) + (y_diff ** 2) + (z_diff ** 2))
        theta[:, vertex] = np.arctan(y_diff / x_diff)   # swapped due to equations having different Cartesian coordinate system layout
        phi[:, vertex] = np.arccos(z_diff / rho[:, vertex])

    # put spherical coords in bodydata matrix for displaying in the model
    freemocap_data_out[:, 27, :] = rho
    freemocap_data_out[:, 28, :] = theta    # 
    freemocap_data_out[:, 29, :] = phi      # z / rho^2
        # the data at the parts of the freemocap_3d_body_data tensor used here are not actual for body parts (anymore), but places to hold information
        # for simplification for displaying via plotly, which is used by freemocap, hence its continued use here

    return rho, theta, phi

# get spherical coordinates for use later on
rho, theta, phi = get_spher_coords()



### FORCES CALCULATIONS

# calculate forces of muscle exertions of the left arm
def run_formula_calculations():
    h_p = 1.75    # meters      # height of person
    w_p = 90      # kilograms   # weight of person
    w_bal = 3     # kilograms   # weight of ball

    # convert sim units to metric units
    conv_factor = calc_conversion_ratio(h_p)

    # convert rho from simulated units to metric units
    rho = freemocap_3d_body_data[:, 27, :] * conv_factor

    # equations used primarily from the paper labeled "shoulderarm3.pdf" in the dropbox, as well as some info from emails between Dr. Liu and I (Bri)
    #w_fa = w_p * 0.023                      # weight of forearm
    #cgf = h_p * 0.432 * 0.216               # center of gravity of forearm
    #f = h_p * 0.216                         # length of forearm
    #b = h_p * 0.11                          # dist between elbow and bicep insertion point
    #u = h_p * 0.173                         # length of upper arm

    # instead of using averages for segment length, use calculated instead
    f = rho[:, 2] * conv_factor
    u = rho[:, 1] * conv_factor
    b = u * 0.636                   # calculated via algebra using pre-existing average proportions data
    w_fa = w_p * (f * 0.1065)       # use ratio of f to weight proportion to get weight with calculated f 
    cgf = 2 * (f ** 2)                     # calculated via algebra using pre-existing average proportions data

    # angles
    theta_arm = phi[:, 1] - (np.pi / 2)                         # angle at shoulder
    theta_u = elbow_angle                                       # angle at elbow
    theta_b = np.pi - ( (b - u * np.cos(theta_u)) / np.sqrt( (b ** 2) + (u ** 2) - 2 * b * u * np.cos(theta_u) ) )      # angle at bicep insertion point
    theta_la = np.sin(theta_u + theta_arm - (np.pi / 2))        # angle used for leverage arms fa and bal

    # lever arms
    la_fa = cgf * theta_la                                      # forearm lever arm
    la_bal = f * theta_la                                       # ball lever arm
    la_bic = b * np.sin(theta_b)                                # bicep lever arm

    # forces
    force_bicep = (w_fa * la_fa + w_bal * la_bal) / la_bic      # force applied by bicep muscle


    # save calculations in a way to help prep for visual in plotly output

    # theta-arm, bicep force, theta-u
    freemocap_3d_body_data[:, 26, 0] = np.rad2deg(theta_arm)
    freemocap_3d_body_data[:, 26, 1] = force_bicep 
    freemocap_3d_body_data[:, 26, 2] = np.rad2deg(elbow_angle)

    # rho, theta, phi
    freemocap_3d_body_data[:, 27, :] = rho
    freemocap_3d_body_data[:, 28, :] = np.rad2deg(theta) 
    freemocap_3d_body_data[:, 29, :] = np.rad2deg(phi)

run_formula_calculations()


### 3D PLOTTING

# create and display 3D plot of the previously manipulated data
def plot_body_data(body_data = freemocap_3d_body_data):
    # set to not display in scientific notation
    np.set_printoptions(suppress = True, precision = 3)

    # the structure of the following (displaying with plotly) was put together in part by analyzing the code used to do the same in the default Freemocap output

    def calculate_axes_means(skeleton_3d_data):
        mx_skel = np.nanmean(skeleton_3d_data[:,0:33,0])
        my_skel = np.nanmean(skeleton_3d_data[:,0:33,1])
        mz_skel = np.nanmean(skeleton_3d_data[:,0:33,2])

        return mx_skel, my_skel, mz_skel

    ax_range = 1500

    mx_skel, my_skel, mz_skel = calculate_axes_means(body_data)

    # Create a list of frames
    frames = [go.Frame(data=[go.Scatter3d(
        x=body_data[i, [11, 13, 15], 0],
        y=body_data[i, [11, 13, 15], 1],
        z=body_data[i, [11, 13, 15], 2],
        mode='markers',#+text',
        marker=dict(
            size=5,  # Adjust marker size as needed
            color=body_data[i, 11:17:2, 1],
            colorscale='Jet',
            opacity=0.8
        )
    )], name=str(i),
    # add spherical coord annotations -BD
    layout = go.Layout(annotations = [
        dict(
            text = "( Theta-arm : Bicep force : Theta-u (elbow angle) ): ",
            x = 0.1, y = 1,
            showarrow = False
        ),
        dict(
            text = "\t" + str(body_data[i, 26, :]),
            x = 0.1, y = 0.9,
            showarrow = False
        ),
        dict(
            text = "Rho: " + str(body_data[i, 27, :]),
            x = 0.1, y = 0.8,
            showarrow = False
        ),
        dict(
            text = "Theta: " + str(body_data[i, 28, :]),
            x = 0.1, y = 0.7,
            showarrow = False
        ),
        dict(
            text = "Phi: " + str(body_data[i, 29, :]),
            x = 0.1, y = 0.6,
            showarrow = False
        ),
    ])
    ) for i in range(body_data.shape[0])]

    # Define axis properties
    axis = dict(
        showbackground=True,
        backgroundcolor="rgb(230, 230,230)",
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
    )

    # Create a figure
    fig = go.Figure(
        data=[go.Scatter3d(
            x=body_data[0, [11, 13, 15], 0],
            y=body_data[0, [11, 13, 15], 1],
            z=body_data[0, [11, 13, 15], 2],
            mode='markers',
            marker=dict(
                size=5,  # Adjust marker size as needed
                color=body_data[0, 11:17:2, 1],
                colorscale='Jet',
                opacity=0.8
            )
        )],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(axis, range = [-1400, 1400]),            #range=[mx_skel-ax_range, mx_skel+ax_range]), # Adjust range as needed
                yaxis=dict(axis, range = [-1000, 1000]),            #range=[my_skel-ax_range, my_skel+ax_range]), # Adjust range as needed
                zaxis=dict(axis, range = [-1000, 0]),               #range=[mz_skel-ax_range, mz_skel+ax_range]),  # Adjust range as needed
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=True,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, {"frame": {"duration": 30}}]
                ),
                # add "pause" button
                dict(
                    args = [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    label = 'Stop',
                    method = 'animate'
                )]
            )]
        ),
        frames=frames
    )


    return fig

# plot the body data 
plot_body_data().show()



### BICEP FORCES GRAPH

# graph bicep forces
def plot_bicep_forces(body_data = freemocap_3d_body_data):
    # plot bicep force / theta_u
    y = np.abs(freemocap_3d_body_data[:, 26, 1])             # bicep force
    x = np.rad2deg(elbow_angle)                     # angle at elbow

    # plot
    plt.scatter(x, y)
    plt.ylim([0, 1000])     # restrict y axis to between 0 and 1000 (Newtons)
    plt.xlim([0, 180])      # restrixt x axis to between 0 and 180 (degrees)

    # plot bicep forces
    return plt

# plot forces graph
plot_bicep_forces().show()
