Facial Detection with a Raspberry Pi 4
======

A. This package includes the following files.
|-- README.txt [This file]
|-- main.py
|-- mainpi.py
|-- pickler.py
|-- haarcascade_frontalface_default.xml
B. pickeled_images
    |-- .pickle files (binary streams of images)
C. output_images
    |-- .image files (captured and successfully detected frames)

About
======
This program utilizes Open Source Computer Vision (OpenCV) libraries in order to process images and utlizes its algorithms to recognize faces.
Using Python's Pickle module, we can store images as streams of data to uniquely identify inidividuals. 

Usage
======

In order to instantiate people to be recognized, input name as name variable in pickler.py and run to create pickeled images
To detect faces run: main.py

main.py
Image and video analysis subsystem that contains methods for decision-making and comparison subsystems

mainpi.py
Raspberry Pi implementation for main

pickler.py
Face enrollment database 

haarcascade_frontalface_default.xml:
Pre-trained object detection algorithm used to detect human faces from OpenCV

B. pickeled_images
This directory will hold pickeled images created from pickeler.py

C. output_images
This directory will hold images taken from live frames that were successfully detected/face matched





