import cv2
import os
import time
import pickle
import numpy as np


# Initialize the cascade classifier and video capture
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1)  # 0 for default cam, 1 for external camera on Mac

image_counter = 0

output_folder = 'pickled_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Pickle the first image when 'P' is pressed
def pickle_image(grey_frame, filename):
    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(grey_frame, f)

# Function to extract the face region
def extract_face(image, face_coords):
    (x, y, w, h) = face_coords
    return image[y:y+h, x:x+w]

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with classifier against the grayscale image
    faces = face_detector.detectMultiScale(grey, 1.1, 8)

    for (x, y, w, h) in faces:
        # If 'P' is pressed, pickle the image
        if cv2.waitKey(1) == ord('p'):
            # Generate a unique filename
            filename = f"image_{image_counter}.pickle"
            pickle_image(grey, filename)
            print(f"Image pickled as {filename}.")
            image_counter += 1


    # Show the frame
    cv2.imshow("Camera", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
