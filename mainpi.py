import time
import cv2
import os
import pickle
import numpy as np
from picamera2 import Picamera2

if os.path.isdir('pickled_images'):
    pickled = True  # set to true if you have pickled your faces with pickler.py
else:
    pickled = False

# Initialize face detector and PiCamera
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, controls={"FrameRate": 10}))
picam2.start()

# Folder for pickled and saved images
pickled_folder = "pickled_images"
save_dir = "precision_test"
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
detections = []

# Function to extract the face region
def extract_face(image, face_coords):
    x, y, w, h = face_coords
    return image[y:y + h, x:x + w]

# Function to compare faces using MSE
def compare_faces(face1, face2):
    face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY) if len(face1.shape) == 3 else face1
    face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY) if len(face2.shape) == 3 else face2
    face2_resized = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
    mse = np.sum((face1 - face2_resized) ** 2) / float(face1.size)
    return mse

# Load pickled images
def load_pickled_images():
    pickled_images = []
    for filename in os.listdir(pickled_folder):
        if filename.endswith(".pickle"):
            filepath = os.path.join(pickled_folder, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'matrix' in data and 'person' in data:
                    pickled_images.append(data)
    return pickled_images

# Load pickled images
if pickled:
    pickled_images = load_pickled_images()

prevTime = 0
detectedFrames = 0
DetectionTimes = 0
verifiedPickleTime = 0
verifiedPickles = 0

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Convert to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detectionStartTime = time.time()

    # Detect faces with classifier against the grayscale image
    faces = face_detector.detectMultiScale(grey, 1.1, 8)

    detectionEndTime = time.time()

    currTime = time.time()

    fps = 1 / (currTime - prevTime)

    prevTime = currTime

    #put framerate on image
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        DetectionTimes += (detectionEndTime - detectionStartTime)
        detectedFrames += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#        if frame_count % 20 == 0:  # Save every 20 frames
#            save_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
#            cv2.imwrite(save_path, frame)
#            detections.append((save_path, len(faces)))

        if pickled:
            verifiedPickles = 0
            verifiedPickleTime = 0
            for pickled_image in pickled_images:
                matrix = pickled_image['matrix'] # get the image itself

                # Detect face in the pickled image
                pickled_faces = face_detector.detectMultiScale(matrix, 1.1, 8)
                if pickled_faces.any():
                    pickleStartTime = time.time()
                    pickled_face = extract_face(matrix, pickled_faces[0])
                    live_face = extract_face(frame, (x, y, w, h))
                    mse = compare_faces(live_face, pickled_face)
                    print(mse)
                    if mse < 90:
                        verifiedPickles += 1
                        person = pickled_image['person']
                        cv2.putText(frame, f"Hello {person}!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        verifiedPickleTime = time.time() - pickleStartTime
                        break
            else:
                cv2.putText(frame, "No match", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Resize the frame for display (smaller window)
    display_frame = cv2.resize(frame, (640, 480))  # Resize

    frame_count += 1
    cv2.imshow("Face Detection", display_frame)

    if cv2.waitKey(1) == ord('q'):
        break

if detectedFrames > 0:
    avgDetectionTime = DetectionTimes / detectedFrames
    print(f"Avg Detection Time: {avgDetectionTime}")
if verifiedPickles > 0:
    avgPicklerTime = verifiedPickleTime / verifiedPickles
    print(f"Avg Match Time: {avgPicklerTime}")

cv2.destroyAllWindows()

for detection in detections:
    print(f"Saved: {detection[0]} - Faces detected: {detection[1]}")
