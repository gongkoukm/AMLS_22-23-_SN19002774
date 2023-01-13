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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import joblib
import matplotlib.pyplot as plt


basedir_t = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23_test/cartoon_set_test'
# basedir_t = '../Datasets/dataset_AMLS_22-23_test/cartoon_set_test'
images_dir_t = os.path.join(basedir_t,"img")
images_dir_t = images_dir_t.replace('\\', '/')
labels_filename_t = 'labels.csv'


def get_faceshape(basedir, labels_filename):
    # Get all cartoon's image paths
    with open(os.path.join(basedir, labels_filename), 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        # Skip the first row
        next(reader)

        faceshape_list = []
  
        for row in reader:
            value = row[0]
    
            # Split the value into parts
            parts = re.split("\s+", value)
    
            # parts[2] represents the third value, which is the label of face shape
            faceshape_label = parts[2]
            faceshape_list += [faceshape_label]

    return faceshape_list


def get_landmarks(folder):
  # Load the shape predictor model
  # predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
  predictor = dlib.shape_predictor("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/shape_predictor_68_face_landmarks.dat")

  # Initialize an empty list to store the landmarks and empty landmarks
  landmarks = []
  no_landmarks = []

  # Sort the filenames in numerical order in the folder
  filenames = (os.listdir(folder))
  filenames.sort(key=lambda x: int(x.split(".")[0]))

  # Loop through the images in the folder
  for file in filenames:
    # Load the image
    image = cv2.imread(os.path.join(folder, file))

    # Detect faces in the image
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # If no face is detected, append the filename to the no_landmarks list
    if len(faces) == 0:
      no_landmarks.append(file)
      continue

    landmark = predictor(gray, faces[0])

    # Convert the landmarks to a list of (x, y) coordinates
    landmark = [(point.x, point.y) for point in landmark.parts()]

    # Add the landmarks to the list
    landmarks.append(landmark)

  # Return the landmarks
  return landmarks, no_landmarks, filenames


def filter(faceshape_labels, filenames, no_landmarks):
    # Filter the labels with no face detected
    filtered_faceshape_labels = []
    # Check if the filename is not in the no_landmarks list
    for label, filename in zip(faceshape_labels, filenames):
        if filename not in no_landmarks:
            filtered_faceshape_labels.append(label)
    return filtered_faceshape_labels


# Get the features need for B1
def get_faceshape_features(landmarks):

    def get_points(a,b,tuples):

        elements = []

        for t in tuples:
            elements.append(t[a][b])
        return elements
    
    # Calculate distance between two points for lists
    def distance(x1, y1, x2, y2):
        distances = []
        for i in range(len(x1)):
            d = math.sqrt(math.pow(x1[i] - x2[i], 2) + math.pow(y1[i] - y2[i], 2))
            distances.append(d)
        return distances


    x1 = get_points(7,0,landmarks)
    y1 = get_points(7,1,landmarks)
    x2 = get_points(9,0,landmarks)
    y2 = get_points(9,1,landmarks)
    x3 = get_points(6,0,landmarks)
    y3 = get_points(6,1,landmarks)
    x4 = get_points(10,0,landmarks)
    y4 = get_points(10,1,landmarks)
    x5 = get_points(5,0,landmarks)
    y5 = get_points(5,1,landmarks)
    x6 = get_points(11,0,landmarks)
    y6 = get_points(11,1,landmarks)
    x7 = get_points(4,0,landmarks)
    y7 = get_points(4,1,landmarks)
    x8 = get_points(12,0,landmarks)
    y8 = get_points(12,1,landmarks)
    x9 = get_points(3,0,landmarks)
    y9 = get_points(3,1,landmarks)
    x10 = get_points(13,0,landmarks)
    y10 = get_points(13,1,landmarks)
    x11 = get_points(2,0,landmarks)
    y11 = get_points(2,1,landmarks)
    x12 = get_points(14,0,landmarks)
    y12 = get_points(14,1,landmarks)

    # Get distance of face shape points
    d1 = distance(x1, y1, x2, y2)
    d2 = distance(x3, y3, x4, y4)
    d3 = distance(x5, y5, x6, y6)
    d4 = distance(x7, y7, x8, y8)
    d5 = distance(x9, y9, x10, y10)
    d6 = distance(x11, y11, x12, y12)
    
    X = []
    for i in range(len(x1)):
        # Create a feature vector by combining the values 
        x = (d1[i], d2[i], d3[i], d4[i], d5[i], d6[i])
        # Add the feature vector to the list
        X.append(x)


    return X