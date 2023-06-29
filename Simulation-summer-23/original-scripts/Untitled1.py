#!/usr/bin/env python
# coding: utf-8

# In[2]:

# BD Notes:
    # Removed references of ".mediapipe.python.solutions", as it was causing errors.

#################
# This uses mediapipe + opencv2 to map right arm (3 joints) to a matplotlib graph
# To run this, you should install: 
    # version: python 3.9.* (3.10 not supported for some below dependencies)
    # modules: matplotlib, multiprocessing, cv2, mediapipe
#I would suggest running this once by itself before looking at the code
# todo: handle imports correctly so that variable type names can be shortened
#################

#to display live updating 3d graph
import matplotlib.pyplot as plt
from matplotlib import animation

#to use camera tracking and graph live update at the same time
from multiprocessing import Process
from multiprocessing.shared_memory import ShareableList

#for camera and arm tracking (public library)
import cv2
import mediapipe as mp

#this class handles all the graphing
#to live update the graph, the animate function should be called, 
#important variable to note: "sl_master" is the live updating arm data
#note: sl_master is a dict of shareablelists "landmarks", such that the dict has 3 entries for the 3 joints, 
# and each shareablelist refer to their data (ex. live 3D coord),

class LivePlotter:

    #initialzies the graph to default state
    def __init__(self, delta):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')     

        #as of the moment, the graph limits are hard coded
        self.ax.set_xlim(-1, 1) #-: move right/left of screen
        self.ax.set_ylim(-1, 1.5) #-: move up and up of screen
        self.ax.set_zlim(-1, 1) #-: move closer to screen

        self.ax.set_xlabel("left/right of screen")
        self.ax.set_ylabel("up/down of screen")
        self.ax.set_zlabel("closer/further from screen")

        #changes the perspective of the initial 3d graph angle
        self.ax.view_init(-90, -90)

        #arbitrarily initialize the points and lines that represent the arm on the graph - each will be updated live
        self.wrist_point, = self.ax.plot3D(0, 0, 0, "ro")
        self.elbow_point, = self.ax.plot3D(0, 0, 0, "go")
        self.shoulder_point, = self.ax.plot3D(0, 0, 0, "bo")
        self.forearm_line, = self.ax.plot3D([], [], [], "m-")
        self.uparm_line, = self.ax.plot3D([], [], [], "c-")

        self.delta = delta #delay between frames in ms (may be inconsistent)

    #mpl uses FuncAnimation to allow for live updating graphs
    #all it really does is take a delegate (ex. the update function after this) and execute it every interval
    #we set fargs as sl_master, which is the live updating arm data
    def animate(self, sl_master : dict):
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=self.delta, fargs=(sl_master, ))

    def update(self, _, sl_master : dict):
        self.update_points(sl_master["wrist"], sl_master["elbow"], sl_master["shoulder"])
        self.update_lines(sl_master["wrist"], sl_master["elbow"], sl_master["shoulder"])

    #for each frame, update the 3 joints on the graph (each coord is length 3 since we get a 3D coordinate)
    def update_points(self, wrist_coord, elbow_coord, shoulder_coord):
        self.wrist_point.set_data_3d(wrist_coord[0], wrist_coord[1], wrist_coord[2])
        self.elbow_point.set_data_3d(elbow_coord[0], elbow_coord[1], elbow_coord[2])
        self.shoulder_point.set_data_3d(shoulder_coord[0], shoulder_coord[1], shoulder_coord[2])
        
    #for each frame, update the 2 lines that connect between the 3 joints
    def update_lines(self, wrist_coord, elbow_coord, shoulder_coord):
        forearm_seg = [
            [wrist_coord[0], elbow_coord[0]],
            [wrist_coord[1], elbow_coord[1]],
            [wrist_coord[2], elbow_coord[2]]
        ]

        uparm_seg = [
            [elbow_coord[0], shoulder_coord[0]],
            [elbow_coord[1], shoulder_coord[1]],
            [elbow_coord[2], shoulder_coord[2]]
        ]

        self.forearm_line.set_data_3d(forearm_seg)
        self.uparm_line.set_data_3d(uparm_seg)


#Plotting entrypoint - creates an instance of LivePlotter, and passes in sl_master to animate
def plot_func(sl_master : dict, delta):
    obj = LivePlotter(delta)
    obj.animate(sl_master)
    plt.show()

#main tracking part of the software
def simulate(sl_master : dict):
    mp_drawing = mp.solutions.drawing_utils #shows the lines
    mp_pose = mp.solutions.pose #pose solution api

    capture = cv2.VideoCapture(0) #pass in webcam 0
    with mp_pose.Pose(min_detection_confidence=.5, min_tracking_confidence=.5) as pose: #play with args to see what tracking works better
        while capture.isOpened(): 
            _, frame = capture.read() #capture data from camera as frame variable

            #detect pose from frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #cv2 does BGR by default, so we want to convert to RGB
            img.flags.writeable = False #prevents the below process function to use unneccesary memory
            
            res = pose.process(img) #feed the frame into the mediapipe pose for tracking

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #recolor back to normal

            update_data(sl_master, res.pose_landmarks, mp_pose) #update the coordinates in sl_master to accurately display on mpl graph

            #draw the lines and dots onto the frame received from the camera
            #this draws for ALL joints, not sure if its possible to only display right arm
            mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS) 

            #display the frame with the added landmarks 
            cv2.imshow("feed", img) 

            #press q to quit 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        #cleanup
        capture.release()
        cv2.destroyAllWindows()
    
#updates sl_master for the plot_func data read
def update_data(sl_master : dict, pose_landmarks, mp_pose : mp.solutions.pose):  # BD: changed "mp.solutions.mediapipe.python.solutions.pose" to "mp.solutions.pose", now the program runs
    if(pose_landmarks):
        update_data_helper(sl_master["wrist"], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        update_data_helper(sl_master["elbow"], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        update_data_helper(sl_master["shoulder"], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

        #note: pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value] returns a 3D coordinate and 
            #visibility value for each joint. In this case we only care about the right arm 3 joints

        #uncomment below to see data in console

        print(type(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]))
        # print("R wrist x coord:\n", pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x)
        #print("R elbow all data:\n", pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        #print("R sholder all data:\n", pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        type(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

#i think there are different landmarks you can choose from, you'd have to see mediapipe documentation
def update_data_helper(sl: ShareableList, normalized_landmark):
    sl[0], sl[1], sl[2] = normalized_landmark.x, normalized_landmark.y, normalized_landmark.z

#entrypoint
if __name__ == '__main__':
    max_delta = 20 #func_anim framerate
    
    #init data
    sl_wrist = ShareableList([0.0,0.0,0.0], name="psm_sl_wrist") 
    sl_elbow = ShareableList([0.0,0.0,0.0], name="psm_sl_elbow") 
    sl_shoulder = ShareableList([0.0,0.0,0.0], name="psm_sl_shoulder") 
    sl_master = {"wrist": sl_wrist, "elbow": sl_elbow, "shoulder": sl_shoulder}

    #separate into processes so grapher and camera can run simultaneously, and start
    p1 = Process(target = plot_func, args = (sl_master, max_delta, ) )
    p2 = Process(target = simulate, args = (sl_master, ))
    p1.start()
    p2.start()

    #wait for user to quit both grapher and camera
    p1.join()
    p2.join()

    #mediapipe cleanup
    sl_wrist.shm.close()
    sl_wrist.shm.unlink()
    sl_elbow.shm.close()
    sl_elbow.shm.unlink()
    sl_shoulder.shm.close()
    sl_shoulder.shm.unlink()


# In[ ]:





# In[ ]:




