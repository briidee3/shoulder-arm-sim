# BD 2023
# This file used for testing mediapipe functionality for use in simulating forces pertaining to the human arm system
# as well as implementing and integrating a custom algorithm for discerning depth from 2D skeleton data

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# testing demo from https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # loop thru detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # draw the pose landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


img = cv2.imread("test_image.jpg")
cv2.imshow('tstimg', img)


# create PoseLandmarker object
base_options = python.BaseOptions(model_asset_path = './landmarkers/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options = base_options, 
    output_segmentation_masks = True
)
detector = vision.PoseLandmarker.create_from_options(options)

# load input image
image = mp.Image.create_from_file("test_image.jpg")

# detect pose landmarks from the input image
detection_result = detector.detect(image)

# process the detection result; in this case, visualize it
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow('thing', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


# visualize pose segmentation mask
segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis = 2) * 255
cv2.imshow('tstmask', visualized_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)