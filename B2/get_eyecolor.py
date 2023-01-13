import cv2
import dlib
import os
import time
import math
import ast
import re
import csv
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import joblib
import matplotlib.pyplot as plt

# Load the shape predictor model

basedir = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23/cartoon_set'
basedir_t = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23_test/cartoon_set_test'
images_dir = os.path.join(basedir,"img")
images_dir = images_dir.replace('\\', '/')
images_dir_t = os.path.join(basedir_t,"img")
images_dir_t = images_dir_t.replace('\\', '/')
labels_filename = 'labels.csv'
labels_filename_t = 'labels.csv'

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
    # Extract the color of the region defined by landmarks 38, 39, 41, and 42
    landmark1 = landmarks[0][38]
    landmark2 = landmarks[0][39]
    landmark3 = landmarks[0][41]
    landmark4 = landmarks[0][42]

    # Find the top-left and bottom-right coordinates of the rectangular region
    top_left = (min(landmark1[0], landmark2[0], landmark3[0], landmark4[0]),
                min(landmark1[1], landmark2[1], landmark3[1], landmark4[1]))
    bottom_right = (max(landmark1[0], landmark2[0], landmark3[0], landmark4[0]),
                    max(landmark1[1], landmark2[1], landmark3[1], landmark4[1]))

    # Extract the rectangular region using numpy slicing
    region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Calculate the average color of this region
    avg_color_per_row = np.average(region, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = avg_color.tolist()

    # Make eye color to list
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

def get_eyecolor(basedir, labels_filename):
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
    
            # parts[1] represents the second value, which is the label of eye color
            eyecolor_label = parts[1]
            eyecolor_list += [eyecolor_label]

    return eyecolor_list

eyecolor_labels = get_eyecolor(basedir, labels_filename)
eyecolor_labels_t = get_eyecolor(basedir_t, labels_filename_t)


# Filter the labels with no face detected
def filter(eyecolor_labels, filenames, no_landmarks):
  # Filter the labels with no face detected
  filtered_eyecolor_labels = []
  # Check if the filename is not in the no_landmarks list
  for label, filename in zip(eyecolor_labels, filenames):
    if filename not in no_landmarks:
      filtered_eyecolor_labels.append(label)
  return filtered_eyecolor_labels


# # Write landmarks, no_landmarks and eye colors in txt file
# with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/landmarksB.txt', 'w') as f1:
#   # Clear the contents of the file
#   f1.truncate(0)
#   # Join the elements of the list into a single string, with a newline character as the separator
#   for element in landmarks:
#    f1.write(str(element) + '\n')
# with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/NOfilenamesB.txt', 'w') as f3:
#   # Clear the contents of the file
#   f3.truncate(0)
#   # Join the elements of the list into a single string, with a newline character as the separator
#   for element in no_landmarks:
#    f3.write(str(element) + '\n')
# with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/eye_colorsB.txt', 'w') as f4:
#   # Clear the contents of the file
#   f4.truncate(0)
#   # Join the elements of the list into a single string, with a newline character as the separator
#   for element in eye_colors:
#    f4.write(str(element) + '\n')


# Read lists from txt files
eye_colors_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/eye_colorsB.txt", "r")
  
with eye_colors_file as f:
  eye_colors = f.readlines()
eye_colors = [line.strip() for line in eye_colors]
eye_colors = [ast.literal_eval(s) for s in eye_colors]


# Read no_landmarks
filenames = (os.listdir(images_dir))
filenames.sort(key=lambda x: int(x.split(".")[0]))
no_filenames_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/NOfilenamesB.txt", "r")
with no_filenames_file as f:
  no_landmarks = f.readlines()
no_landmarks = [label.strip() for label in no_landmarks]


landmarks_t, no_landmarks_t, filenames_t, eye_colors_t = get_landmarks(images_dir_t)

X = eye_colors + eye_colors_t
# X = eye_colors
filtered_eyecolor_labels = filter(eyecolor_labels, filenames, no_landmarks)
filtered_eyecolor_labels_t = filter(eyecolor_labels_t, filenames_t, no_landmarks_t)
y = filtered_eyecolor_labels + filtered_eyecolor_labels_t
# y = filtered_eyecolor_labels

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# # # Fit the model to the training data
# # model.fit(X_train, y_train)

# k = 5
# scores = cross_val_score(model, X, y, cv=k)
# # Print the scores for each fold
# print("Scores for each fold: ", scores)
# # Print the mean score
# print("Mean score: ", scores.mean())

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
# calculate mean and standard deviation
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

model.fit(X, y)
joblib.dump(model, 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/B2/B2random_forest_model.pkl')

# plot the learning curve
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.xlabel("Eyecolor Training examples")
plt.ylabel("Score")
plt.ylim([0, 1])
plt.show()