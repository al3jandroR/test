import cv2
import os
import time

# initializes cascade classifier and video capture
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1) # 0 for default cam, but its 1 for camera on Mac

save_dir = "precision_test"
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
detections = []

while True:
    # capture frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # convert to gray scale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # time taken to detect face in each frame
    start_time = time.time()

    # detects faces with classifier against gray scale image
    faces = face_detector.detectMultiScale(grey, 1.1, 8)

    end_time = time.time()

    for (x, y, w, h) in faces:
        print(f"Time taken: {end_time - start_time}")

        # draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # saves detected frame to folder
        save_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(save_path, frame)
        detections.append((save_path, len(faces)))

    frame_count += 1

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for detection in detections:
    print(f"Saved: {detection[0]} - Faces detected: {detection[1]}")
