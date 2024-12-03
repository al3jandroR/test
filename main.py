import subprocess
import time
import cv2
import os
import pickle
import numpy as np

#databaseAdd = input("Would you like to add your face to the database?\n1: Yes\n2: No\n")

#if databaseAdd == "1":
   # subprocess.run(["python", "pickler.py"])

if os.path.isdir('pickled_images'):
    pickled = True  # set to true if you have pickled your faces with pickler.py
else:
    pickled = False

# Initialize the cascade classifier and video capture
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)  # Index: 0 for Windows and Raspberry Pi, 1 for Mac

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

prevTime = time.time()

if pickled:
    pickled_face_arr = []
    for pickled_image in pickled_images:
        matrix = pickled_image['matrix'] # get the image itself

        # Detect face in the pickled image
        pickled_faces = face_detector.detectMultiScale(matrix, 1.1, 8)
        if len(pickled_faces) > 0:
            pickled_face = extract_face(matrix, pickled_faces[0])
            pickled_face_arr.append({'image': pickled_face, 'pickle' : pickled_image})

#variables for metrics
prevTime = time.time()
beforeDetection = 0
afterDetection = 0
runningCountOfDetectionTime = 0
averageTimeForDetection = 0
numberOfDetections = 0
numberOfMatches = 0
averageTimeForMatch = 0
runningCountOfMatchTime = 0
beforeMatch = 0
afterMatch = 0
missedMatches = 0

startTime = time.time()
while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    beforeDetection = time.time()

    # Detect faces with classifier against the scaled down grayscale image 
    small_frame = cv2.resize(grey, (0, 0), fx=0.5, fy=0.5)
    faces = face_detector.detectMultiScale(small_frame, 1.1, 8)

    afterDetection = time.time()
    print(f'Detection Time: {afterDetection - beforeDetection: .4f}')
    runningCountOfDetectionTime = (afterDetection - beforeDetection)
    numberOfDetections += 1

    # Calculates fps and put it on image
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        # Compensate for scaled down image
        x, y, w, h = 2*x, 2*y, 2*w, 2*h

        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        # Checks face in frame to pickle database every 5 frames
        if pickled:
            match_found = False
            beforeMatch = time.time()
            live_face = extract_face(small_frame, (int(x/2), int(y/2), int(w/2), int(h/2)))
            for face in pickled_face_arr:
                mse = compare_faces(live_face, face['image'])

                # If MSE is low, faces are likely the same
                if mse < 90:
                    numberOfMatches += 1
                    person = (face['pickle'])['person'] # get the person associated with that pickeled image
                    match_found = True  # Set match_found to True when a match is found
                    afterMatch = time.time()
                    runningCountOfMatchTime += (afterMatch - beforeMatch)
                    print(f"Match Time: {afterMatch - beforeMatch: .4f}")
                    cv2.putText(frame, "Face Matched!", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2, cv2.LINE_AA) # put valid text on screen in green
                    cv2.putText(frame, person, (x,y-5), cv2.FONT_ITALIC, .8, (0, 255, 0), 2, cv2.LINE_AA) # put text to identify person
                    break  # No need to check further pickled images
            if not (match_found):
                cv2.putText(frame, "No match", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) #put invalid text on screen in red

    cv2.imshow("Camera", frame)
    cv2.waitKey(20)

    # Exit loop if 'q' is pressed
    if int(time.time()) == int(startTime + 5):
        if (numberOfDetections > 0):
            averageTimeForDetection = runningCountOfDetectionTime/numberOfDetections
        if (numberOfMatches > 0):
             averageTimeForMatch = runningCountOfMatchTime/numberOfMatches
        print(f"Average time to detect faces in frame: {averageTimeForDetection:.6f}")
        print(f"Average time to match detected face: {averageTimeForMatch:.6f}")
        print(f'Match to Detection ratio: {numberOfMatches / numberOfDetections:.3f}')
        beforeDetection = 0
        afterDetection = 0
        runningCountOfDetectionTime = 0
        averageTimeForDetection = 0
        numberOfDetections = 0
        numberOfMatches = 0
        averageTimeForMatch = 0
        runningCountOfMatchTime = 0
        beforeMatch = 0
        afterMatch = 0
        missedMatches = 0

cap.release()
cv2.destroyAllWindows()


# Print saved detection results
for detection in detections:
    print(f"Saved: {detection[0]} - Faces detected: {detection[1]}")
