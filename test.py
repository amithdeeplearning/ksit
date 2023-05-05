import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the image
image = cv2.imread('images.jpg')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()

    # Remove the background
    imgout = img.copy()
    frame_rgb = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    mp_drawing.draw_landmarks(imgout, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Get the bounding box of the body
    bboxes = []
    if pose_results.pose_landmarks is not None:
        height, width, _ = imgout.shape
        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            pt = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
            try:
                bboxes.append((pt[0], pt[1]))
            except:
                pass
        bboxes = cv2.boundingRect(np.array(bboxes))

        # Resize the image to fit the bounding box of the body
        image_resized = cv2.resize(image, (bboxes[2], bboxes[3]))

        # Get the position of the top-left corner of the bounding box
        x, y, w, h = bboxes

        # Replace the region inside the bounding box with the image
        imgout[y:y+h, x:x+w] = image_resized

    # Display the image
    cv2.imshow("Image", img)
    cv2.imshow("Imageout", imgout)
    cv2.waitKey(1)
