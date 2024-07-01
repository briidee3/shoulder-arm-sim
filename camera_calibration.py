# BD 2024
# This program is designed to implement camera calibration using Zhang's method of camera calibration as it is implemented in OpenCV.
# This program also integrates a picture taking feature, for help expediting the calibration process.
# This program may be run standalone to run through the process of taking several pictures for use
#   by `calibrate_camera`, or the functions can be used on their own.
#
# Resources used to help development: 
#   https://learnopencv.com/camera-calibration-using-opencv/
#   https://pythonwife.com/camera-calibration-in-python-with-opencv/



import cv2
import numpy as np
import os
import glob
import time
from datetime import datetime
from tkinter import *
from PIL import Image
from PIL import ImageTk
from functools import partial



# iterator denoting current picture
#cur_pic = 0

# checkerboard dimensions
checkerboard = (10,14)                                                              # height x width of checkerboard (num of boxes)
criteria_setting = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria for stopping point of cornerSubPix

# delay between frames (15 for 60 frames per sec)
delay = 15
# countdown time in secs
countdown_len = 5
# directory in "calibration" directory to save calibration pics in
save_dir_name = "calibrate_pics"
# denote how many pics to take
num_pics = 5
# name of data output file
output_filename = "output.txt"



### HELPER FUNCTIONS

# take picture with camera
#def take_picture(camera = cv2.VideoCapture(0), num_pics = 5, countdown = 3, feedback_var = StringVar(), num_var = StringVar()):
"""def take_picture(camera, num_pics, countdown, feedback_var, num_var):
    # wait a few seconds to give user some time to position themselves
    for i in range(0, countdown + 1):
        feedback_var.set("Camera status: Counting down... " + str(countdown - i))
        time.sleep(1)

    # read data from camera to take a picture
    ret, img = camera.read()

    if not ret:
        feedback_var.set("Camera status: Camera could not be read. \n\tPlease identify the issue then try again.")
        print("Camera could not be read. Please identify the issue then try again.")
    else:
        cur_pic = num_pics - num_var.get[-1]   # get current pic number

        cv2.imwrite(str(cur_pic) + ".png", img)         # save pic to local file system
        
        # update info
        num_var.set("Number of pictures left to take: " + str(num_pics - cur_pic))

        print("Picture saved to local folder `calibration_pictures`.")

    # return cur_pic to update cur_pic in main function
    return cur_pic"""

"""
# (alternative to prev) take picture with camera via gui image
#def take_picture_gui(image = Label().photo, num_pics = 5, countdown = 3, feedback_var = StringVar(), num_var = StringVar()):
def take_picture_gui(img, num_pics, countdown, feedback_var, num_var):
    try:
        # read data from gui element to "take a picture"
        img = image.photo       # TODO: ensure this takes the image at time of end of countdown, not at time of function call
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_RGB2BGR)     # un-flip and convert back to BGR from RGB

        cur_pic = num_pics - num_var.get[-1]   # get current pic number from gui element

        cv2.imwrite(str(cur_pic) + ".png", img)         # save pic to local file system
        
        # update info
        num_var.set("Number of pictures left to take: " + str(num_pics - cur_pic))

        print("Picture saved to local folder `calibration_pictures`.")

        # return cur_pic to update cur_pic in main function
        return cur_pic
    except Exception as e:
        print("camera_calibration.py: Exception taking picture in `take_picture_gui`: \n\t%s" % str(e))
"""

# prepare workspace for storing pics
def prep_workspace(dir_name = "calibrate_pics"):
    if "calibration" in os.listdir("."):
        os.chdir("calibration")

    # check if windows; if so, use windows file pathing
    if os.name == 'nt':
        dir_name = dir_name.replace("/", "\\")
    
    # if the directory for holding calibration pics doesn't exist, create it
    if not dir_name in os.listdir("."):
        os.mkdir(dir_name)      # make directory



### GUI FUNCTIONS

# gui main loop
#def update_gui(camera = cv2.VideoCapture(0), tk_image_label = Label(), root = Tk()):
def update_gui(camera, tk_image_label, root, is_countdown, countdown_end_time, is_taking_pic, feedback_var, num_var):
    # read image from camera
    try:
        ret, frame = camera.read()                                      # get image from camera
    except Exception as e:
        print("camera_calibration.py: Exception getting image from camera in `update_gui`: \n\t%s" % str(e))

    # handle frame/image data
    try:
        if ret:
            tk_image_label.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)))   # display image in GUI after flipping horizontally and converting image to RGB
            tk_image_label.configure(image = tk_image_label.photo)    
    except Exception as e:
        print("camera_calibration.py: Exception displaying image in GUI in `update_gui`: \n\t%s" % str(e))                  # configure image in GUI

    try:
        cur_pic = num_pics - int(num_var.get()[-1])         # get current pic number from gui element

        # handle picture taking
        if is_countdown.get():
            countdown_end_time.set(str(datetime.now().timestamp() + countdown_len))
            feedback_var.set("Camera status: Counting down from %s... " % str(countdown_len))
            is_taking_pic.set(True)
            is_countdown.set(False)
        if is_taking_pic.get() and (float(countdown_end_time.get()) <= datetime.now().timestamp()):
            cv2.imwrite(os.path.join(save_dir_name, str(cur_pic) + ".png"), frame)           # save pic to local file system

            # set gui vars
            num_var.set("Number of pictures left to take: " + str(num_pics - (cur_pic + 1)))
            feedback_var.set("Camera status: ")
            is_taking_pic.set(False)

            print("Picture saved to local folder `calibration_pictures`.")

        # if cur_pic <= 0, picture taking part of calibration is over, so gui should be closed
        if num_pics - cur_pic <= 0:
            __del__(root, camera, True)

    except Exception as e:
        print("camera_calibration.py: Exception taking picture in `update_gui`: %s" % str(e))


    # call next cycle
    root.after(delay, update_gui, camera, tk_image_label, root, is_countdown, countdown_end_time, is_taking_pic, feedback_var, num_var)


# set up and run GUI for this step of calibration
def run_cam_calib_gui():
    try:
        # settings
        #cur_pic = 0             # denote current picture (need >3 for calibration      # no longer used, just using pic_num_var for this
        #countdown = 3           # time in seconds from press of "Take picture" button to taking of picture
        element_width = 20      # default width of text in gui elements
        
        # set up workspace
        os.chdir("calibration")         # go into local "calibration" directory
        prep_workspace()
            
        # instructions
        instructions = str("Hold the checkerboard in front of the camera " + 
            "and take {num_of_pics} pictures of it in different orientations ".format(num_of_pics = num_pics) + 
            "at different positions in the camera's view. " + 
            "After clicking the \"Take picture\" button below, you will " + 
            "have 3 seconds to prepare, then the camera will take a picture " + 
            "for use in calibration.")
        

        # initialize camera
        cam = cv2.VideoCapture(0)


        # initialize GUI
        root = Tk()          # set up tkinter root
        root.title("Camera Calibration")


        # create gui frame
        gui = Frame(root)       # set it up inside the root
        gui.grid()              # set up a grid

        # create label for image display
        image_label = Label(gui)
        image_label.grid(row = 0, column = 0)
        image_label.photo = None

        # create frame for displaying info/instructions to user
        info_frame = Frame(gui)
        info_frame.grid(row = 0, column = 1)

        # create label frame for displaying instructions
        instruct_frame = LabelFrame(info_frame, text = "Instructions: ")
        instruct_frame.grid(row = 0, column = 0)

        # create text object for displaying instructions
        instructs_text = Text(instruct_frame, state = DISABLED)
        instructs_text.grid(row = 0, column = 0)
        instructs_text.insert(index = 1.0, chars = instructions)


        # create frame for user input
        input_frame = Frame(info_frame)
        input_frame.grid(row = 1, column = 0)

        # create label to display feedback for taking picture
        pic_fb_label = Frame(input_frame)    
        pic_fb_label.grid(row = 1, column = 0)
        # create variable for showing camera feedback text info
        pic_fb_var = StringVar()
        pic_fb_var.set("Camera status: ")
        pic_fb_entry = Label(pic_fb_label, textvariable = pic_fb_var, height = 1, width = 40)
        pic_fb_entry.grid(row = 0, column = 0)
        # create string variable for showing number of pics left to take
        pic_num_var = StringVar()               # create var to hold num of pics left to take
        pic_num_var.set("Number of pictures left to take: " + str(num_pics))
        pic_num_entry = Label(pic_fb_label, textvariable = pic_num_var, height = 1, width = 40)
        pic_num_entry.grid(row = 1, column = 0)

        # create booleans and string to help handle counting down between functions
        is_countdown = BooleanVar()
        is_countdown.set(False)
        is_taking_pic = BooleanVar()
        is_taking_pic.set(False)
        countdown_end_time = StringVar()
        countdown_end_time.set("0")             # initialize end time to start of UNIX time
        # create button for taking pictures
        #take_pic_command = partial(take_picture_gui, cam, countdown, pic_fb_var, pic_num_var) # put together function call with args = Button(input_frame, text = "Take picture", command = take_pic_command)
        #take_pic_button = Button(input_frame, text = "Take picture", command = take_pic_command)
        take_pic_button = Button(input_frame, text = "Take picture", command = lambda : is_countdown.set(True))
        take_pic_button.grid(row = 0, column = 0)


        try:
            # start the GUI
            root.bind("<Escape>", lambda e : __del__(root, cam, False))       # use escape key to exit program
            root.after(delay, update_gui, cam, image_label, root, is_countdown, countdown_end_time, is_taking_pic, pic_fb_var, pic_num_var)
            root.mainloop()

            # if the number of pics left to take is 0, exit gui, save data to file
            #if int(pic_num_var.get()[-1]) <= 0:
            #    root.destroy()

        except Exception as e:
            print("camera_calibration.py: Exception starting GUI: \n\t%s" % str(e))

    except Exception as e:
        print("camera_calibration.py: Exception in `run_cam_calib_gui`: \n\t%s" % str(e))

# run calibration and end gui
def __del__(root, cam, is_calibrate):

    # end gui
    root.destroy()

    # run calibration on images
    if is_calibrate:
        calibrate_camera(save_dir_name, cam, checkerboard, criteria)
    
    # release camera
    cam.release()



### CALIBRATION FUNCTIONS

# calibrate the camera as described in https://learnopencv.com/camera-calibration-using-opencv/
#   much of the code of this function is used directly as exemplified in the above link
#def calibrate_camera(dir_name = "calibrate_pics", camera = cv2.VideoCapture(1), checkerboard = (6,9), criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), calc_reproj_err = True, save_to_file = True, w_h = (640, 480)):
def calibrate_camera(dir_name = "calibrate_pics", camera = cv2.VideoCapture(1), checkerboard = checkerboard, criteria = criteria_setting, calc_reproj_err = True, save_to_file = True, w_h = (640, 480)):
    try:
        prep_workspace()

        # used to store points for each checkerboard image
        objpoints = []  # 3d points
        imgpoints = []  # 2d points

        # defining world coords for 3d points
        objp = np.zeros( (1, checkerboard[0] * checkerboard[1], 3), np.float32)             # initializing ndarray to hold data for one pic at a time
        objp[:, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)   # creating a mesh grid
        
        # store object points and image points from all images
        objpoints = []      # 3d points in real world space
        imgpoints = []      # 2d points in image plane
        
        prev_img_shape = None                                                               # hold shape of prev image

        # get list of pics in given directory
        #pics = os.listdir(dir_name)
        pics = glob.glob(os.path.join(dir_name + "*.png"))
        
        # go through and run calibration steps on each of the pics in the directory
        try:
            for pic in pics:
                img = cv2.imread(pic)                           # read the current pic into a variable
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert color pic to grayscale

                # find chessboard corners
                #   if desired num of corners not found, ret = False
                ret, corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

                # if we found the desired num of corners, refine pixel coords and display on checkerboard images
                if ret == True:
                    objpoints.append(objp)                                                              # append empty 3d data for current pic to objpoints (will be filled in during later step)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)     # refine pixel coords for given 2d points
                    imgpoints.append(corners_refined)                                                   # append 2d refined corners data for current pic to imgpoints
                    img = cv2.drawChessboardCorners(img, checkerboard, corners_refined, ret)            # get rows and cols of image
                
                # TESTING (replace with other code after finished testing)
                cv2.imshow("img", img)  # display image on screen
                cv2.waitKey(0)          # wait for keyboard interrupt

                prev_img_shape = gray.shape[::-1]   
                      
        except Exception as e:
            print("camera_calibration.py: Exception in calibrate_camera: \n\t%s" % str(e))  # overlay found corners onto img


        cv2.destroyAllWindows()     # exit opencv
        #h, w = img.shape[:2]        # height and width of img

        # do camera calibration by passing value of known 3d points (objpoints) and corresponding pixel coords of detected corners (imgpoints)
        #   returns (success check, camera matrix, distortion coefficients, rotation vector, translation vector) in that order
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, w_h, None, None)


        # calculate reprojection error
        try:
            if calc_reproj_err:
                reproj_err = calc_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
                print("camera_calibration.py: \n\tCalculated reprojection error: %s" % reproj_err)
        except Exception as e:
            print("camera_calibration.py: Exception calculating reprojection error in calibrate_camera: \n\t%s" % str(e))  # overlay found corners onto img

        # (optional) save data to file
        try:
            if save_to_file:
                os.chdir("calibration")                     # change to calibration directory
                
                # if file already exists, delete it before writing

                # open file, write new data into it
                with open(output_filename, "w") as file:
                    file.write(":".join(map(str, (ret, mtx, dist, rvecs, tvecs))))
                    file.close()
                    
        except Exception as e:
            print("camera_calibration.py: Exception saving calibration data to file: \n\t%s" % str(e))

        # print data to console
        print("Camera matrix: %s\nDistortion coefficients: %s\nRotation vectors: %s\n Translation vectors: %s" % (str(mtx), str(dist), str(rvecs), str(tvecs)))

        return (ret, mtx, dist, rvecs, tvecs)
        
    except Exception as e:
        print("camera_calibration.py: Exception in `calibrate_camera`: \n\t%s" % str(e))
        return (0)

# remove distortion from input image
#   it may be best to just do this from within the code dealing with the stream of images, to try to prevent unnecessary data passing
def get_undistorted(img, camera_matrix, dist_coeffs, camera_matrix_new):#, image_size, alpha, new_image_size):
    try:
        # get height and width of image
        h, w = img.shape[:2]

        # apply undistortion
        img_new = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix_new)

        # crop undistorted image using roi from prev step
        x, y, w, h = roi
        img_new = img_new[y:y + h, x:x + w]

        # return cropped and undistorted image
        return img_new
    except Exception as e:
        print("camera_calibration.py: Exception in `get_undistorted`: \n\t%s" % str(e))
        return [0]

# calculate reprojection error between projected point and measured point
#   quantifies how close an estimate of a 3d point gets to the points true projection
def calc_reprojection_error(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    try:
        mean_error = 0      # reprojection error

        # go thru for each of the images used for calibration
        for i in range(len(objpoints)):
            imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgponits_proj)
            mean_error += error
        
        return mean_error / len(objpoints)
    except Exception as e:
        print("camera_calibration.py: Exception in `calc_reprojection_error`: \n\t%s" % str(e))
        return -1


if __name__ == '__main__':
    calib = run_cam_calib_gui()
