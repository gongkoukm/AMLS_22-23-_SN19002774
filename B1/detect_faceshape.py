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


# Load the shape predictor model

basedir = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23/cartoon_set'
images_dir = os.path.join(basedir,"img")
images_dir = images_dir.replace('\\', '/')
labels_filename = 'labels.csv'


def get_landmarks(folder):
  # Load the shape predictor model
  predictor = dlib.shape_predictor("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/shape_predictor_68_face_landmarks.dat")

  # Initialize an empty list to store the landmarks and empty landmarks
  landmarks = []
  no_landmarks = []

  # Sort the filenames in numerical order in the folder
  filenames = (os.listdir(folder))
  filenames.sort(key=lambda x: int(x.split(".")[0]))

  # Get the current performance counter value
  start = time.perf_counter()

  # Loop through the images in the folder
  for file in filenames:
  # for i,file in enumerate(filenames):
  #   if i >= 10:
  #    break
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

  # Get the current performance counter value
  end = time.perf_counter()

  # Compute the elapsed time
  elapsed_time = end - start

  # Print the elapsed time
  print('Elapsed time:', elapsed_time)

  # Return the landmarks
  return landmarks, no_landmarks, filenames


def get_faceshape():
    # Get all celeba's image paths
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


# Read lists from txt files
# opening the landmarks.txt file in read mode
landmarks = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/landmarksB.txt", "r")
  
with landmarks as f:
  lines = f.readlines()

# Initialize an empty list to store the tuples
tuples = []

# Loop over the lines and parse each line using ast.literal_eval
for line in lines:
  t = ast.literal_eval(line)
  tuples.append(t)


# print(tuples[0][1][1])
def get_points(a,b):

  elements = []

  for t in tuples:
    elements.append(t[a][b])
  return elements


filenames = (os.listdir(images_dir))
filenames.sort(key=lambda x: int(x.split(".")[0]))
no_filenames_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/NOfilenamesB.txt", "r")
with no_filenames_file as f:
  no_landmarks = f.readlines()
no_landmarks = [label.strip() for label in no_landmarks]

faceshape_label = get_faceshape()

# Filter the labels with no face detected
filtered_faceshape_labels = []
# Check if the filename is not in the no_landmarks list
for label, filename in zip(faceshape_label, filenames):
  if filename not in no_landmarks:
    filtered_faceshape_labels.append(label)

# Points of landmarks = (8,10)(7,11)(6,12)(5,13)(4,14), which are points represent faceshape
x1 = get_points(7,0)
y1 = get_points(7,1)
x2 = get_points(9,0)
y2 = get_points(9,1)
x3 = get_points(6,0)
y3 = get_points(6,1)
x4 = get_points(10,0)
y4 = get_points(10,1)
x5 = get_points(5,0)
y5 = get_points(5,1)
x6 = get_points(11,0)
y6 = get_points(11,1)
x7 = get_points(4,0)
y7 = get_points(4,1)
x8 = get_points(12,0)
y8 = get_points(12,1)
x9 = get_points(3,0)
y9 = get_points(3,1)
x10 = get_points(13,0)
y10 = get_points(13,1)
x11 = get_points(2,0)
y11 = get_points(2,1)
x12 = get_points(14,0)
y12 = get_points(14,1)

def distance(x1, y1, x2, y2):
  distances = []
  for i in range(len(x1)):
    d = math.sqrt(math.pow(x1[i] - x2[i], 2) + math.pow(y1[i] - y2[i], 2))
    distances.append(d)
  return distances

# Get distance of face shape points
d1 = distance(x1, y1, x2, y2)
d2 = distance(x3, y3, x4, y4)
d3 = distance(x5, y5, x6, y6)
d4 = distance(x7, y7, x8, y8)
d5 = distance(x9, y9, x10, y10)


# X = []
# for i in range(len(d1)):
#   # Create a feature vector by combining the distance lists
#   x = (d1[i], d2[i], d3[i], d4[i], d5[i])
#   # Add the feature vector to the list
#   X.append(x)
# y = filtered_faceshape_labels

X = []
for i in range(len(d1)):
  # Create a feature vector by combining the distance lists
  x = (x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i], x9[i], x10[i], x11[i], x12[i])
  # Add the feature vector to the list
  X.append(x)
y = filtered_faceshape_labels

# Initialize the model
model = RandomForestClassifier()

# # Fit the model to the training data
# model.fit(X_train, y_train)

k = 5
scores = cross_val_score(model, X, y, cv=k)
# Print the scores for each fold
print("Scores for each fold: ", scores)
# Print the mean score
print("Mean score: ", scores.mean())