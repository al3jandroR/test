import sys
import cv2
import os
import pickle
import time


# Initialize the cascade classifier and video capture
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)  # Index: 0 for Windows and Raspberry Pi, 1 for Mac

image_counter = 0

output_folder = 'pickled_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# a pickled image is comprised of a matrix(image itself) and an associated name. 
def pickle_image(grey_frame, filename, person_name):
    filepath = os.path.join(output_folder, filename)
    data = {'matrix': grey_frame, 'person': person_name}  # Include the name and image

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

# Function to extract the face region
def extract_face(image, face_coords):
    (x, y, w, h) = face_coords
    return image[y:y+h, x:x+w]


person_name = input("What is your name?\n") 

print("You may add as many photos as you'd like to the database.\n\n-q, Press q to quit\n\n-p, Press p to add photo to database")
prevTime = 0
while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.\n")
        break

    # Convert to grayscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_frame = cv2.resize(grey, (0, 0), fx=0.5, fy=0.5)

    # Detect faces with classifier against the grayscale image
    faces = face_detector.detectMultiScale(small_frame, 1.1, 8)

    #calculates fps
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    #put framerate on image
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

    # If 'P' is pressed, pickle the image
    key = cv2.waitKey(40)
    if (key == ord('p')):
        # Generate a unique filename
        filename = f"image_{person_name}{time.time()}.pickle"
        pickle_image(grey, filename, person_name)
        print(f"Image pickled as {filename}.")
        image_counter += 1
    # Exit the loop if 'q' is pressed
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    # Show the frame
    cv2.imshow("Camera", frame)


