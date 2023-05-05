import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


bgr = cv2.imread('desert-1654439__340.jpg')
new_size = (640, 480)

# Resize the image
bgr = cv2.resize(bgr, new_size)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
segmentor = SelfiSegmentation()
# image_bg = cv2.resize(bgr,(640,480))
# segmentor.removeBG(img,bgr)

while True:
    success ,img = cap.read()
    imgout = segmentor.removeBG(img, bgr,threshold=0.7)
    frame_rgb = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    mp_drawing.draw_landmarks(imgout, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)





    cv2.imshow("Image",img)
    cv2.imshow("Imageout",imgout)
    cv2.waitKey(1)