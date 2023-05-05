import cv2
import mediapipe as mp
import numpy as np

# Load the T-shirt image
tshirt_img = cv2.imread('images.jpg')

# Load the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the landmarks using MediaPipe
    results = pose.process(frame)

    # Check if landmarks were detected
    if results.pose_landmarks:

        # Get the coordinates of the landmark points
        landmark_points = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in results.pose_landmarks.landmark]

        # Create a mask from the T-shirt image
        tshirt_mask = np.zeros(tshirt_img.shape[:2], dtype=np.uint8)
        tshirt_mask[tshirt_img[:,:,0]>0] = 1
        tshirt_mask = cv2.resize(tshirt_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

        # Create a new image with the T-shirt color
        tshirt_color = tshirt_img[:,:,0:3]
        tshirt_color = cv2.resize(tshirt_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

        # Superimpose the T-shirt color on the frame
        frame = cv2.addWeighted(frame, 1.0 - tshirt_mask, tshirt_color, tshirt_mask, 0)

        # Get the bounding box of the body
        x, y, w, h = cv2.boundingRect(np.array(landmark_points))

        # Resize the T-shirt image to fit the bounding box
        tshirt_resized = cv2.resize(tshirt_img, (w, h), interpolation=cv2.INTER_AREA)

        # Superimpose the T-shirt image on the body
        frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.5, tshirt_resized, 0.5, 0)

    # Convert the frame back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow('T-Shirt Augmented Reality', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
