import cv2
import dlib
import os
import time
import math
import ast
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Load the shape predictor model

basedir = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23/celeba'
images_dir = os.path.join(basedir,"img")
images_dir = images_dir.replace('\\', '/')
labels_filename = 'labels.csv'
predictor = dlib.shape_predictor("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/shape_predictor_68_face_landmarks.dat")
# Initialize an empty list to store the landmarks and empty landmarks

no_landmarks = []
image = cv2.imread(os.path.join(images_dir,"10.jpg"))
detector = dlib.get_frontal_face_detector()
faces = detector(image)
landmark = predictor(image, faces[0])
# Convert the landmarks to a list of (x, y) coordinates
landmark = [(point.x, point.y) for point in landmark.parts()]
# Add the landmarks to the list
# landmark.append(landmark)

# print(landmark)

for (x, y) in landmark:
    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

# Display image
cv2.imshow("Image with landmarks", image)
cv2.waitKey(0)