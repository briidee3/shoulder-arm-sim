# BD 2024
# This program is intended to serve as the main process for the biomechanics simulation project.abs


### TODO:
#   - make a new class for doing only the physics calculations


import multiprocessing

import gui              # custom class, handles gui
import livestream       # custom class, handles mediapipe live stream
import extrapolation    # custom class, handles depth extrapolation (and all other calculations for now)


### OPTIONS:

# model to use for mediapipe
pose_landmarker = '../landmarkers/pose_landmarker_full.task'

# load and prep placeholder image for program initialization
no_image_path = './no_image.png'            # placeholder image location
no_image = Image.fromarray(cv2.cvtColor(cv2.imread(no_image_path), cv2.COLOR_BGR2RGB))


### ENTRY POINT
if __name__ == '__main__': 
    # set multiprocessing start method to 'spawn' to prevent any issues running on windows (where only 'spawn' is available) and linux (where default is 'fork')
    multiprocessing.set_start_method('spawn')

    # initialize pipes
    extrap_to_stream_r, extrap_to_stream_w = multiprocessing.Pipe()     # pipe to (live)stream from extrapolation
    stream_to_extrap_r, stream_to_extrap_w = multiprocessing.Pipe()     # pipe to extrap(olation) from livestream
    gui_to_stream_r, gui_to_stream_w = multiprocessing.Pipe()           # pipe to (live)stream from gui
    stream_to_gui_r, stream_to_gui_w = multiprocessing.Pipe()           # pipe to gui from (live)stream
    gui_to_extrap_r, gui_to_extrap_w = multiprocessing.Pipe()           # pipe to extrap(olation) from gui


    ### PROCESS INITIALIZATION
    # initialize gui
    gui = SimGUI()
    gui.start()

    # initialize livestream process
    livestream = livestream.Pose_detection(pose_landmarker, stream_to_extrap_w, extrap_to_stream_r, stream_to_gui_w, gui_to_stream_r)
    livestream.start()
    
    # intitialize extrapolation process
    ep = extrapolation.Extrapolate_forces(pipe_to_stream = pipe_to_stream_w, pipe_to_extrap = pipe_to_extrap_r)
    ep.start()
