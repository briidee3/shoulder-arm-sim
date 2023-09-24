# This program was designed for use simulating approximate force distributions on the human arm
#   for physical therapy and physics research applications.
# BD 2023


import numpy as np

# the following 65 lines taken from FreeMoCap default file output
from pathlib import Path

try:
    import numpy as np
except Exception as e:
    print(e)
    #%pip install numpy
    import numpy as np


try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except Exception as e:
    print(e)
    #%pip install plotly
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go


bodypart = 'left_index'
path_to_recording = "C:\\Users\\briid\\Documents\\Research\\Arm-Simulation-with-Forces\\experimentation\\freemocap\\recording_19_56_30_gmt-4"
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


# get indices for body parts
def get_index(body_part = 'left_wrist'):
    return mediapipe_indices.index(body_part)

# function to get data for particular body part
def get_bodypart_data(bodypart = "left_index"):
    
    joint_to_plot_index = mediapipe_indices.index(bodypart)

    return freemocap_3d_body_data[:,joint_to_plot_index,:]

# function to get distance between two body parts
# parameters are the data arrays for the body parts
# returns array of distances between parts
def dist_between_vertices(first_part, second_part):
    cur_dist = np.ndarray(shape = first_part.shape, dtype = first_part.dtype)
    for i in range(0, len(first_part) - 1):     # get the distance for each frame of the video (excluding last frame)
        cur_dist[i] = np.linalg.norm(first_part[i] - second_part[i])
        #print(cur_dist[i])]
    return cur_dist[:-1, 0]

# get median of 5 largest distances between hand and 
def max_dist_between_parts(dist_array):
    ind = (np.argpartition(dist_array, -15)[-15:])
    return np.median(dist_array[ind])

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

# put together dictionary coordinating everything
depth_dict = {
    'vertex_order': vertex_order,                # pairs of body parts/segments
    'y_axes': y_axes,   # approximated depth
}


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


# set depth for the body data
freemocap_3d_body_data = set_depth(body_data = freemocap_3d_body_data)


# the following 70 lines taken from the default FreeMoCap output .ipynb
def calculate_axes_means(skeleton_3d_data):
    mx_skel = np.nanmean(skeleton_3d_data[:,0:33,0])
    my_skel = np.nanmean(skeleton_3d_data[:,0:33,1])
    mz_skel = np.nanmean(skeleton_3d_data[:,0:33,2])

    return mx_skel, my_skel, mz_skel

ax_range = 1500

mx_skel, my_skel, mz_skel = calculate_axes_means(freemocap_3d_body_data)

# Create a list of frames
frames = [go.Frame(data=[go.Scatter3d(
    x=freemocap_3d_body_data[i, 11:17:2, 0],
    y=freemocap_3d_body_data[i, 11:17:2, 1],
    z=freemocap_3d_body_data[i, 11:17:2, 2],
    mode='markers',
        marker=dict(    # marker modified -BD
            size=5,  # Adjust marker size as needed
            color=freemocap_3d_body_data[i, 11:17, 1],
            colorscale='Jet',
            opacity=0.8
        )
)], name=str(i)) for i in range(freemocap_3d_body_data.shape[0])]

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
        x=freemocap_3d_body_data[0, 11:17:2, 0],
        y=freemocap_3d_body_data[0, 11:17:2, 1],
        z=freemocap_3d_body_data[0, 11:17:2, 2],
        mode='markers',
        marker=dict(    # marker modified -BD
            size=5,  # Adjust marker size as needed
            color=freemocap_3d_body_data[0, 11:17, 1],
            colorscale='Jet',
            opacity=0.8
        )
    )],
    layout=go.Layout(
        scene=dict(
            xaxis=dict(axis, range=[mx_skel-ax_range, mx_skel+ax_range]), # Adjust range as needed
            yaxis=dict(axis, range=[my_skel-ax_range, my_skel+ax_range]), # Adjust range as needed
            zaxis=dict(axis, range=[mz_skel-ax_range, mz_skel+ax_range]),  # Adjust range as needed
            aspectmode='cube'
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, {"frame": {"duration": 30}}]
            )]
        )]
    ),
    frames=frames
)

fig.show()


