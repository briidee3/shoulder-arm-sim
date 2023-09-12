
import numpy as np

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
    

# taken from the default output .ipynb
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

# save pre-modification for testing purposes
body_data_pre_mod = freemocap_3d_body_data

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

# get body part data
left_index_data = get_bodypart_data("left_index")
left_elbow_data = get_bodypart_data("left_elbow")

# ratio between current length and zero length:
left_hand_to_elbow_array = dist_between_vertices(left_index_data, left_elbow_data)

# get median of largest distances between vertices/bodyparts
def max_dist_between_parts(dist_array):
    ind = (np.argpartition(dist_array, -20)[-20:-5])
    return np.median(dist_array[ind])

# get approximate distance between vertices/body parts
left_hand_to_elbow_dist = max_dist_between_parts(left_hand_to_elbow_array)
print(left_hand_to_elbow_dist)

# calculate the angle of the segment (body part) from the normal (where it is longest)
def angle_from_normal(cur_dist, max_dist):
    return np.arccos(cur_dist / max_dist)

angle_from_normal(0, left_hand_to_elbow_dist)

# Nice! The output, `1.570796`, is half of pi, or pi/2. This is exactly what we were looking for when the cur_dist is 0, telling us that it works as expected.

# Now, use this to calculate the distance the (body part) vertex is away from the norm with the angle, and there's your depth.

test_vertex_one = get_bodypart_data('left_index')
test_vertex_two = get_bodypart_data('left_elbow')
test_dist_array = dist_between_vertices(test_vertex_one, test_vertex_two)

test_dist_array = test_dist_array
test_max_dist = max_dist_between_parts(test_dist_array)
test_angle = angle_from_normal(test_dist_array[13], test_max_dist)
test_depth = np.sin(test_angle) * test_max_dist
print(test_angle)
print(test_depth)


# get depth
def get_depth(vertex_one, vertex_two):
    dist_array = dist_between_vertices(vertex_one, vertex_two)
    max_dist = max_dist_between_parts(dist_array)
    depths = list()
    for frame in dist_array:
        angle = angle_from_normal(frame, max_dist)
        depths.append(np.sin(angle) * max_dist)
    return depths

# put together pairs for each of the vertices
# the first one is the point which will be moved on the y-axis, the second one for calculations
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

freemocap_3d_body_data[20, :, 1]    # the '1' here indicates the y-axis

# get indices for body parts used
indices = list()
for vertices in depth_dict['vertex_order']:
    for vertex in vertices:
        indices.append(get_index(vertex))

print(indices)

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

# testing...
test_body_data = freemocap_3d_body_data

test_body_data = set_depth(body_data = test_body_data)

print(test_body_data[:, 13, 1])

# set the depth:
freemocap_3d_body_data = set_depth(body_data = freemocap_3d_body_data)


# get height ratio
left_index_data = get_bodypart_data("left_index")
right_index_data = get_bodypart_data("right_index")
dist = dist_between_vertices(left_index_data, right_index_data)
# get ratio to real distance in meters
real_height_metric = 1.78     # meters
sim_wingspan = np.max(dist)     # max distance between wrists via `freemocap`

sim_to_real_conversion_factor = real_height_metric / sim_wingspan
print(sim_to_real_conversion_factor)


# get max length of segments/body parts
upper_arm_max_length = max_dist_between_parts(dist_between_vertices(freemocap_3d_body_data[:, 11, :], freemocap_3d_body_data[:, 13, :]))
lower_arm_max_length = max_dist_between_parts(dist_between_vertices(freemocap_3d_body_data[:, 13, :], freemocap_3d_body_data[:, 15, :]))
print(upper_arm_max_length * sim_to_real_conversion_factor)
print(lower_arm_max_length * sim_to_real_conversion_factor)


# not scientific notation
np.set_printoptions(suppress = True, precision = 3)


# get x y and z for spherical coordinate transformations/angle getting
x = freemocap_3d_body_data[:, 11:17:2, 0]
y = freemocap_3d_body_data[:, 11:17:2, 1]
z = freemocap_3d_body_data[:, 11:17:2, 2]

# calculate vectors for getting angle
vector_a = [(x[:, 0] - x[:, 1]), (y[:, 0] - y[:, 1]), (z[:, 0] - z[:, 1])]
vector_b = [(x[:, 2] - x[:, 1]), (y[:, 2] - y[:, 1]), (z[:, 2] - z[:, 1])]

forearm_length = np.sqrt( (vector_b[0] ** 2) + (vector_b[1] ** 2) + (vector_b[2] ** 2) )
upperarm_length = np.sqrt( (vector_a[0] ** 2) + (vector_a[1] ** 2) + (vector_a[2] ** 2) )

angle_between = np.arccos( ( (vector_a[0] * vector_b[0]) + (vector_a[1] * vector_b[1]) + (vector_a[2] * vector_b[2]) ) / (forearm_length * upperarm_length) )

# Now put it in the data matrix for display by plotly

freemocap_3d_body_data[:, 30, :] = np.swapaxes(vector_a, 0, 1)
freemocap_3d_body_data[:, 31, :] = np.swapaxes(vector_b, 0, 1)

freemocap_3d_body_data[:, 32, 0] = forearm_length * sim_to_real_conversion_factor
freemocap_3d_body_data[:, 32, 1] = upperarm_length * sim_to_real_conversion_factor
freemocap_3d_body_data[:, 32, 2] = angle_between


# get spherical coordinates for each of the 3 vertices (bodyparts) of interest, 
#   and set them to overwrite parts 27-29 of freemocap_3d_body_data for displaying with `plotly`

# initialize to x.shaped ndarray since x is conveniently the same shape we want
rho = np.zeros(x.shape)
theta = np.zeros(x.shape)
phi = np.zeros(x.shape)

# set for [elbow, wrist]    using difference between current point and shoulder point
for i in range(0, len(rho)):
    for vertex in [1, 2]:       # using shoulder just returns 0 bc its 0 from the point of origin which is the shoulder (hence it's missing here)
        if vertex == 1:     # if elbow (from shoulder to elbow)
            cur_shoulder = freemocap_3d_body_data[i, 11, :]
        elif vertex == 2:   # if wrist (from elbow to wrist)
            cur_shoulder = freemocap_3d_body_data[i, 13, :]
        cur_bodypart = freemocap_3d_body_data[i, (11 + vertex * 2), :]
        x_diff = cur_bodypart[0] - cur_shoulder[0]
        y_diff = cur_bodypart[1] - cur_shoulder[1]
        z_diff = cur_bodypart[2] - cur_shoulder[2]
        rho[i, vertex] = np.abs((x_diff ** 2) + (y_diff ** 2) + (z_diff ** 2))
        theta[i, vertex] = np.arctan2(y_diff, x_diff)
        phi[i, vertex] = np.arccos( (z_diff / np.sqrt(rho[i, vertex])) )    # needs to be normalized between -1 and 1! use max z (arcsin(max/z), or something like that) and max rho for this

# put spherical coords in bodydata matrix for displaying in the model 
# for (shoulder to elbow) and (elbow to wrist) ([:,:,0] == shoulder (empty), [:,:,1] == elbow, [:,:,2] == wrist)
freemocap_3d_body_data[:, 27, :] = rho
freemocap_3d_body_data[:, 28, :] = theta    # 
freemocap_3d_body_data[:, 29, :] = phi      # z / rho^2



## forces equations:
forearm_len = freemocap_3d_body_data[:, 32, 0]
upperarm_length = freemocap_3d_body_data[:, 32, 1]
elbow_angle = freemocap_3d_body_data[:, 32, 2]

rho = freemocap_3d_body_data[:, 27, :] * sim_to_real_conversion_factor
theta = freemocap_3d_body_data[:, 28, :]
phi = freemocap_3d_body_data[:, 29, :]

h_p = 1.78    # meters      # height of person
w_p = 90      # kilograms   # weight of person
w_bal = 3     # kilograms   # weight of ball


# from paper
phi = np.pi - phi   # equivalent to `180 - phi`

# from email
theta_arm = phi[:, 1] - (np.pi / 2)  # gonna use the elbow for this, since that's representing the shoulder to elbow, or the base of the arm 
w_fa = w_p * 0.023
cgf = h_p * 0.432 * 0.216     # maybe go back and change these ratios, instead of averages use the approximations from earlier? (just a thought)
f = h_p * 0.216
b = h_p * 0.11
u = h_p * 0.173

# referencing paper
theta_fa = theta_arm + elbow_angle
theta_fa_coef = np.sin(theta_fa)    # "sin(theta_u + theta_arm - 90(deg))" from the paper line 4 and 5
theta_u = elbow_angle
theta_b = np.pi - ( (b - u * np.cos(theta_u)) / np.sqrt( (b ** 2) + (u ** 2) - (2 * b * u * np.cos(theta_u)) ) )

la_fa = cgf * theta_fa_coef
la_bal = f * theta_fa_coef
la_bic = b * np.sin(theta_b)

force_bicep = ( (w_fa * la_fa + w_bal * la_bal) / la_bic )

# prep for visual in plotly output
freemocap_3d_body_data[:, 32, 0] = theta_fa_coef
freemocap_3d_body_data[:, 32, 1] = force_bicep 
freemocap_3d_body_data[:, 32, 2] = np.rad2deg(elbow_angle)

freemocap_3d_body_data[:, 27, :] = rho
freemocap_3d_body_data[:, 28, :] = np.rad2deg(theta) 
freemocap_3d_body_data[:, 29, :] = np.rad2deg(phi)




# this cell was copied and pasted straight from the FreeMoCap output .ipynb file
# and then edited
def calculate_axes_means(skeleton_3d_data):
    mx_skel = np.nanmean(skeleton_3d_data[:,0:33,0])
    my_skel = np.nanmean(skeleton_3d_data[:,0:33,1])
    mz_skel = np.nanmean(skeleton_3d_data[:,0:33,2])

    return mx_skel, my_skel, mz_skel

ax_range = 1500

mx_skel, my_skel, mz_skel = calculate_axes_means(freemocap_3d_body_data)

# Create a list of frames
frames = [go.Frame(data=[go.Scatter3d(
    x=freemocap_3d_body_data[i, [11, 13, 15], 0],
    y=freemocap_3d_body_data[i, [11, 13, 15], 1],
    z=freemocap_3d_body_data[i, [11, 13, 15], 2],
    mode='markers',#+text',
    marker=dict(
        size=5,  # Adjust marker size as needed
        color=freemocap_3d_body_data[i, 11:17:2, 1],
        colorscale='Jet',
        opacity=0.8
    )
)], name=str(i),
# add spherical coord annotations -BD
layout = go.Layout(annotations = [
    dict(
        text = "sin(theta_u + theta_arm - 90(deg)), Bicep force, Elbow angle (between upper and fore-arm): ",
        x = 0.1, y = 1,
        showarrow = False
    ),
    dict(
        text = "\t" + str(freemocap_3d_body_data[i, 32, :]),
        x = 0.1, y = 0.9,
        showarrow = False
    ),
    dict(
        text = "Rho: " + str(freemocap_3d_body_data[i, 27, :]),
        x = 0.1, y = 0.8,
        showarrow = False
    ),
    dict(
        text = "Theta: " + str(freemocap_3d_body_data[i, 28, :]),
        x = 0.1, y = 0.7,
        showarrow = False
    ),
    dict(
        text = "Phi: " + str(freemocap_3d_body_data[i, 29, :]),
        x = 0.1, y = 0.6,
        showarrow = False
    ),
])
) for i in range(freemocap_3d_body_data.shape[0])]

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
        x=freemocap_3d_body_data[0, [11, 13, 15], 0],
        y=freemocap_3d_body_data[0, [11, 13, 15], 1],
        z=freemocap_3d_body_data[0, [11, 13, 15], 2],
        mode='markers',
        marker=dict(
            size=5,  # Adjust marker size as needed
            color=freemocap_3d_body_data[0, 11:17:2, 1],
            colorscale='Jet',
            opacity=0.8
        )
    )],
    layout=go.Layout(
        scene=dict(
            xaxis=dict(axis, range = [400, 1400]),          #range=[mx_skel-ax_range, mx_skel+ax_range]), # Adjust range as needed
            yaxis=dict(axis, range = [-500, 1000]),       #range=[my_skel-ax_range, my_skel+ax_range]), # Adjust range as needed
            zaxis=dict(axis, range = [-1000, 0]),       #range=[mz_skel-ax_range, mz_skel+ax_range]),  # Adjust range as needed
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
            # add pause button  -BD
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


fig.show()


