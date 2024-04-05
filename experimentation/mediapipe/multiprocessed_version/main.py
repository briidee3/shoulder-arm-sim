# BD 2024
# This program is intended to serve as the main process for the biomechanics simulation project.abs


### TODO:
#   - make a new class for doing only the physics calculations

import cv2
from PIL import Image

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


# function used to stop processes
def stop_processes(stop, extrap, stream, gui_):
    # set stop event
    stop.set()

    # wait for processes to end
    print("Stopping processes...")
    extrap.join()
    stream.join()
    gui_.join()

    print("Processes ended.")


### ENTRY POINT
if __name__ == '__main__': 
    multiprocessing.freeze_support()    # enable freeze support for multiprocessing
    # set multiprocessing start method to 'spawn' to prevent any issues running on windows (where only 'spawn' is available) and linux (where default is 'fork')
    #multiprocessing.set_start_method('spawn')

    print("Starting biomechanics sim program...")

    # initialize pipes
    print("Initializing data pipes...")
    extrap_to_stream_r, extrap_to_stream_w = multiprocessing.Pipe()     # pipe to (live)stream from extrapolation
    stream_to_extrap_r, stream_to_extrap_w = multiprocessing.Pipe()     # pipe to extrap(olation) from livestream
    gui_to_stream_r, gui_to_stream_w = multiprocessing.Pipe()           # pipe to (live)stream from gui
    stream_to_gui_r, stream_to_gui_w = multiprocessing.Pipe()           # pipe to gui from (live)stream
    gui_to_extrap_r, gui_to_extrap_w = multiprocessing.Pipe()           # pipe to extrap(olation) from gui
    extrap_to_gui_r, extrap_to_gui_w = multiprocessing.Pipe()           # pipe to gui from extrap(olation)

    # initialize lock(s)
    mp_data_lock = multiprocessing.Lock()

    # initialize stop event
    stop = multiprocessing.Event()
    stop.clear()    # redundancy, make sure not set


    ### PROCESS INITIALIZATION
    
    # intitialize extrapolation process
    print("Starting `extrapolation.py`...")
    ep = extrapolation.Extrapolate_forces(stop, False, False, extrap_to_stream_w, stream_to_extrap_r, extrap_to_gui_w, gui_to_extrap_r, mp_data_lock)
    ep.start()

    # initialize livestream process
    print("Starting `livestream.py`...")
    livestream = livestream.Pose_detection(stop, pose_landmarker, stream_to_extrap_w, extrap_to_stream_r, stream_to_gui_w, gui_to_stream_r)
    livestream.start()

    # initialize gui
    print("Starting `gui.py`...")
    gui = gui.Sim_GUI(stop, extrap_to_gui_r, gui_to_extrap_w, stream_to_gui_r, gui_to_stream_w)
    gui.start()

    print("Started processes.")

    # initialize object to hold current key press
    k = cv2.waitKey(33)

    # wait until 'Esc' button pressed, then exit program
    while True:
        k = cv2.waitKey(33)                 # waits for 33 ms iirc
        if k == 27:                         # checks if key is 'Esc' (keycode 27)
            print("Stopping...")
            stop_processes(stop, ep, livestream, gui)
    
    print("Program terminated.")


