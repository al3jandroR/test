import subprocess
import time
import cv2
import os
import pickle
import numpy as np
from picamera2 import Picamera2

databaseAdd = input("Would you like to add your face to the database?\n1: Yes\n2: No\n")

if databaseAdd == "1":
    subprocess.run(["python", "pickler.py"])

if os.path.isdir('pickled_images'):
    pickled = True  # set to true if you have pickled your faces with pickler.py
else:
    pickled = False

# Initialize face detector and PiCamera
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, controls={"FrameRate": 10}))
picam2.start()

# Folder where pickled images are saved
pickled_folder = "pickled_images"

# Folder where detected faces are saved
save_dir = "precision_test"
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
detections = []

# Function to extract the face region
def extract_face(image, face_coords):
    (x, y, w, h) = face_coords
    return image[y:y+h, x:x+w]

# Function to compare two images using MSE
def compare_faces(face1, face2):
    # Convert both faces to grayscale to ensure the same format
    if len(face1.shape) == 3:  # If the face is in color (RGB), convert to grayscale
        face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    if len(face2.shape) == 3:  # If the face is in color (RGB), convert to grayscale
        face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    # Resize face2 to the size of face1 for a fair comparison
    face2_resized = cv2.resize(face2, (face1.shape[1], face1.shape[0]))

    mse = np.sum((face1 - face2_resized) ** 2) / float(face1.size)
    return mse

# Load all pickled images from the pickled folder
def load_pickled_images():
    pickled_images = []
    for filename in os.listdir(pickled_folder):
        if filename.endswith(".pickle"):
            filepath = os.path.join(pickled_folder, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)  # Load the pickled object
                if isinstance(data, dict) and 'matrix' in data and 'person' in data:
                    pickled_images.append(data)  # ensure pickeled image in valid format and add to array
                else:
                    print(f"Pickled file {filename} does not contain valid data.")
    return pickled_images


# Load pickled images once at the beginning
if pickled:
    pickled_images = load_pickled_images()

prevTime = 0

if pickled:
    pickled_face_arr = []
    for pickled_image in pickled_images:
        matrix = pickled_image['matrix'] # get the image itself

        # Detect face in the pickled image
        pickled_faces = face_detector.detectMultiScale(matrix, 1.1, 8)
        if len(pickled_faces) > 0:
            pickled_face = extract_face(matrix, pickled_faces[0])
            pickled_face_arr.append(pickled_face)

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Convert to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with classifier against the grayscale image
    faces = face_detector.detectMultiScale(grey, 1.1, 8)

    #calculates fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    #put framerate on image
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save detected frame to folder every 10 frames
        if frame_count % 10 == 0:
            save_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(save_path, frame)
            detections.append((save_path, len(faces)))

        live_face = extract_face(frame, (x, y, w, h))

        if pickled and frame_count % 10 == 0:
            for face in pickled_face_arr:
                mse = compare_faces(live_face, pickled_face)

                # If MSE is low, faces are likely the same
                if mse < 100:
                    person = pickled_image['person'] # get the person associated with that pickeled image

                    match_found = True  # Set match_found to True when a match is found
                    break  # No need to check further pickled images
                else:
                    match_found = False

        # If no match found after comparing all pickled images, show "No match"
        if not match_found:
            cv2.putText(frame, "No match", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) #put invalid text on screen in red
        if match_found:
            cv2.putText(frame, "Face Matched!", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2, cv2.LINE_AA) # put valid text on screen in green
            cv2.putText(frame, person, (x,y-5), cv2.FONT_ITALIC, .8, (0, 255, 0), 2, cv2.LINE_AA) # put text to identify person

    frame_count += 1

    cv2.imshow("Camera", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print saved detection results
for detection in detections:
    print(f"Saved: {detection[0]} - Faces detected: {detection[1]}")
