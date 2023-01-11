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
# from skimage import io


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
  eye_colors = []

  # Sort the filenames in numerical order in the folder
  filenames = (os.listdir(folder))
  filenames.sort(key=lambda x: int(x.split(".")[0]))
  # print(filenames[:20])
  # Get the current performance counter value
  start = time.perf_counter()

  # Loop through the images in the folder
  for file in filenames:
#   for i,file in enumerate(filenames):
#     if i >= 10:
#      break
    # Load the image
    image = cv2.imread(os.path.join(folder, file))

    # Detect faces in the image
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)
    
    # If no face is detected, append the filename to the no_landmarks list
    if len(faces) == 0:
      no_landmarks.append(file)
      continue

    landmark = predictor(image, faces[0])

    # Convert the landmarks to a list of (x, y) coordinates
    landmark = [(point.x, point.y) for point in landmark.parts()]
    
    # Add the landmarks to the list
    landmarks.append(landmark)
    # print(landmarks[0])
    # extract the color of the region defined by landmarks 38, 39, 41, and 42
    landmark1 = landmarks[0][38]
    landmark2 = landmarks[0][39]
    landmark3 = landmarks[0][41]
    landmark4 = landmarks[0][42]

    # find the top-left and bottom-right coordinates of the rectangular region
    top_left = (min(landmark1[0], landmark2[0], landmark3[0], landmark4[0]),
                min(landmark1[1], landmark2[1], landmark3[1], landmark4[1]))
    bottom_right = (max(landmark1[0], landmark2[0], landmark3[0], landmark4[0]),
                    max(landmark1[1], landmark2[1], landmark3[1], landmark4[1]))

    # extract the rectangular region using numpy slicing
    region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # now you can calculate the average color of this region
    avg_color_per_row = np.average(region, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = avg_color.tolist()

    # Now you can do something with the color, like appending it to a list of features
    eye_colors.append(avg_color)


  # Get the current performance counter value
  end = time.perf_counter()

  # Compute the elapsed time
  elapsed_time = end - start

  # Print the elapsed time
  print('Elapsed time:', elapsed_time)

  # Return the landmarks
  return landmarks, no_landmarks, filenames, eye_colors


# landmarks, no_landmarks, filenames, eye_colors = get_landmarks(images_dir)
# print(len(landmarks))
# print(eye_colors)

def get_eyecolor():
    # Get all cartoon's image paths
    # image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    with open(os.path.join(basedir, labels_filename), 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        # Skip the first row
        next(reader)

        eyecolor_list = []
  
        for row in reader:
            value = row[0]
    
            # Split the value into parts
            parts = re.split("\s+", value)
    
            # parts[2] represents the second value, which is the label of eye color
            eyecolor_label = parts[1]
            eyecolor_list += [eyecolor_label]

    return eyecolor_list

eyecolor_labels = get_eyecolor()
# print(eyecolor_list[:5])

# Filter the labels with no face detected
filtered_eyecolor_labels = []
# Check if the filename is not in the no_landmarks list
for label, filename in zip(eyecolor_labels, filenames):
  if filename not in no_landmarks:
    filtered_eyecolor_labels.append(label)


# Write landmarks, no_landmarks and eye colors in txt file
with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/landmarksB.txt', 'w') as f1:
  # Clear the contents of the file
  f1.truncate(0)
  # Join the elements of the list into a single string, with a newline character as the separator
  for element in landmarks:
   f1.write(str(element) + '\n')
with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/NOfilenamesB.txt', 'w') as f3:
  # Clear the contents of the file
  f3.truncate(0)
  # Join the elements of the list into a single string, with a newline character as the separator
  for element in no_landmarks:
   f3.write(str(element) + '\n')
with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/eye_colorsB.txt', 'w') as f4:
  # Clear the contents of the file
  f4.truncate(0)
  # Join the elements of the list into a single string, with a newline character as the separator
  for element in eye_colors:
   f4.write(str(element) + '\n')


X = eye_colors

y = filtered_eyecolor_labels

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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