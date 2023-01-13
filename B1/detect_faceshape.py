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


def get_faceshape(basedir, labels_filename):
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
def get_points(a,b,landmarksAll):

  elements = []

  for t in landmarksAll:
    elements.append(t[a][b])
  return elements

# Read no_landmarks
filenames = (os.listdir(images_dir))
filenames.sort(key=lambda x: int(x.split(".")[0]))
no_filenames_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/NOfilenamesB.txt", "r")
with no_filenames_file as f:
  no_landmarks = f.readlines()
no_landmarks = [label.strip() for label in no_landmarks]

faceshape_labels = get_faceshape(basedir, labels_filename)
faceshape_labels_t = get_faceshape(basedir_t, labels_filename_t)
landmarks_t, no_landmarks_t, filenames_t = get_landmarks(images_dir_t)
landmarksAll = tuples + landmarks_t

def filter(faceshape_labels, filenames, no_landmarks):
  # Filter the labels with no face detected
  filtered_faceshape_labels = []
  # Check if the filename is not in the no_landmarks list
  for label, filename in zip(faceshape_labels, filenames):
    if filename not in no_landmarks:
      filtered_faceshape_labels.append(label)
  return filtered_faceshape_labels

filtered_faceshape_labels = filter(faceshape_labels, filenames, no_landmarks)
filtered_faceshape_labels_t = filter(faceshape_labels_t, filenames_t, no_landmarks_t)


# Points of landmarks = (8,10)(7,11)(6,12)(5,13)(4,14), which are points represent faceshape
x1 = get_points(7,0,landmarksAll)
y1 = get_points(7,1,landmarksAll)
x2 = get_points(9,0,landmarksAll)
y2 = get_points(9,1,landmarksAll)
x3 = get_points(6,0,landmarksAll)
y3 = get_points(6,1,landmarksAll)
x4 = get_points(10,0,landmarksAll)
y4 = get_points(10,1,landmarksAll)
x5 = get_points(5,0,landmarksAll)
y5 = get_points(5,1,landmarksAll)
x6 = get_points(11,0,landmarksAll)
y6 = get_points(11,1,landmarksAll)
x7 = get_points(4,0,landmarksAll)
y7 = get_points(4,1,landmarksAll)
x8 = get_points(12,0,landmarksAll)
y8 = get_points(12,1,landmarksAll)
x9 = get_points(3,0,landmarksAll)
y9 = get_points(3,1,landmarksAll)
x10 = get_points(13,0,landmarksAll)
y10 = get_points(13,1,landmarksAll)
x11 = get_points(2,0,landmarksAll)
y11 = get_points(2,1,landmarksAll)
x12 = get_points(14,0,landmarksAll)
y12 = get_points(14,1,landmarksAll)

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
d6 = distance(x11, y11, x12, y12)


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
  x = (d1[i], d2[i], d3[i], d4[i], d5[i], d6[i])
  # Add the feature vector to the list
  X.append(x)
y = filtered_faceshape_labels + filtered_faceshape_labels_t

# Initialize the model
model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=0)

# # Fit the model to the training data
# model.fit(X_train, y_train)

# k = 5
# scores = cross_val_score(model, X, y, cv=k)
# # Print the scores for each fold
# print("Scores for each fold: ", scores)
# # Print the mean score
# print("Mean score: ", scores.mean())

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=6, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
# calculate mean and standard deviation
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

model.fit(X, y)
joblib.dump(model, 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/B1/B1random_forest_model.pkl')

# plot the learning curve
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.xlabel("Faceshape Training examples")
plt.ylabel("Score")
plt.ylim([0, 1])
plt.show()