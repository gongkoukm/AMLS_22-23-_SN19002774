import cv2
import dlib
import os
import csv
import re
import ast
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import joblib


basedir = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23/celeba'
basedir_t = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23_test/celeba_test'
images_dir = os.path.join(basedir,"img")
images_dir = images_dir.replace('\\', '/')
images_dir_t = os.path.join(basedir_t,"img")
images_dir_t = images_dir_t.replace('\\', '/')
labels_filename = 'labels.csv'
labels_filename_t = 'labels.csv'


def get_smile(basedir, labels_filename):
    # Get all celeba's image paths
    # image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    with open(os.path.join(basedir, labels_filename), 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        # Skip the first row
        next(reader)

        smile_list = []
  
        for row in reader:
            value = row[0]
    
            # Split the value into parts
            parts = re.split("\s+", value)
    
            # parts[3] represents the fourth value, which is the label of smile
            smile_label = parts[3]
            smile_list += [smile_label]
            # print(type(gender_label))
            # print(gender_label, end=",")

    return smile_list


def get_landmarks(folder):
  # Load the shape predictor model
  predictor = dlib.shape_predictor("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/shape_predictor_68_face_landmarks.dat")

  # Initialize an empty list to store the landmarks and empty landmarks
  landmarks = []
  no_landmarks = []

  # Sort the filenames in numerical order in the folder
  filenames = (os.listdir(folder))
  filenames.sort(key=lambda x: int(x.split(".")[0]))
  # print(filenames[:20])
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

filenames = (os.listdir(images_dir))
filenames.sort(key=lambda x: int(x.split(".")[0]))
smile_labels = get_smile(basedir, labels_filename)
smile_labels_t = get_smile(basedir_t, labels_filename_t)
landmarks_t, no_landmarks_t, filenames_t = get_landmarks(images_dir_t)

# Read lists from txt files
# opening the landmarks.txt file in read mode
landmarks_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/landmarks.txt", "r")
  
with landmarks_file as f:
  lines = f.readlines()

# Initialize an empty list to store the tuples
landmarks = []

# Loop over the lines and parse each line using ast.literal_eval
for line in lines:
  t = ast.literal_eval(line)
  landmarks.append(t)

landmarksall = landmarks + landmarks_t
# gender_labels_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/labels.txt", "r")

# with gender_labels_file as f:
#   gender_labels = f.readlines()

no_filenames_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/filenames.txt", "r")
with no_filenames_file as f:
  no_landmarks = f.readlines()
no_landmarks = [label.strip() for label in no_landmarks]

# Filter the labels with no face detected
def filter(smile_labels, filenames, no_landmarks):
  # Filter the labels with no face detected
  filtered_gender_labels = []
  # Check if the filename is not in the no_landmarks list
  for label, filename in zip(smile_labels, filenames):
    if filename not in no_landmarks:
      filtered_gender_labels.append(label)
  return filtered_gender_labels

filtered_smile_labels = filter(smile_labels, filenames, no_landmarks)
filtered_smile_labels_t = filter(smile_labels_t, filenames_t, no_landmarks_t)



def get_points(a,b,landmarksall):

  elements = []

  for t in landmarksall:
    elements.append(t[a][b])
  return elements


# Points of corners of the mouth(landmarks = 49,55)
x1 = get_points(48,0,landmarksall)
y1 = get_points(48,1,landmarksall)
x2 = get_points(54,0,landmarksall)
y2 = get_points(54,1,landmarksall)
# Points of the temple(landmarks = 1,17)
x3 = get_points(0,0,landmarksall)
y3 = get_points(0,1,landmarksall)
x4 = get_points(16,0,landmarksall)
y4 = get_points(16,1,landmarksall)


# Calculate distance between two points for lists
def distance(x1, y1, x2, y2):
  distances = []
  for i in range(len(x1)):
    d = math.sqrt(math.pow(x1[i] - x2[i], 2) + math.pow(y1[i] - y2[i], 2))
    distances.append(d)
  return distances


def getAngle(x1,y1,x2,y2,x3,y3):
    angle = []
    for i in range(len(x1)):
        ang = 180-(math.degrees(math.atan2(abs(y3[i]-y2[i]), abs(x3[i]-x2[i])) + math.atan2(abs(y1[i]-y2[i]), abs(x1[i]-x2[i]))))
        angle.append(ang)
    return angle


lip_width = distance(x1, y1, x2, y2)
temple_width = distance(x3, y3, x4, y4)

# Calculate the ratio of lip width and temple width 
# Temple width is constant no matter smile or not
lt_ratios = []
for i in range(len(lip_width)):
  ratio = lip_width[i] / temple_width[i]
  lt_ratios.append(ratio)


# Points of the bottom of eyes(landmarks = 42,47)
x5 = get_points(41,0,landmarksall)
y5 = get_points(41,1,landmarksall)
x6 = get_points(46,0,landmarksall)
y6 = get_points(46,1,landmarksall)

eyemouth_dis = []
eyemouth_dis_left = distance(x5, y5, x1, y1)
eyemouth_dis_right = distance(x6, y6, x2, y2)
for i in range(len(eyemouth_dis_left)):
    emd = eyemouth_dis_left[i] + eyemouth_dis_right[i]
    eyemouth_dis.append(emd)


# Calculate the ratio of eyemouth distance and temple width 
emt_ratios = []
for i in range(len(eyemouth_dis)):
  ratio = eyemouth_dis[i] / temple_width[i]
  emt_ratios.append(ratio)


# Points of the top and bottom lip(landmarks = 52,58)
x7 = get_points(51,0,landmarksall)
y7 = get_points(51,1,landmarksall)
x8 = get_points(57,0,landmarksall)
y8 = get_points(57,1,landmarksall)
lip_updown = distance(x7, y7, x8, y8)
udt_ratios = []
for i in range(len(lip_updown)):
  ratio = lip_updown[i] / temple_width[i]
  udt_ratios.append(ratio)


# Need landmark 49(x1,y1)),67,55(x2,y2) to calculate the curvature of mouth
x9 = get_points(66,0,landmarksall)
y9 = get_points(66,1,landmarksall)
curvatures = getAngle(x1, y1, x9, y9, x2, y2)
curvatures = np.abs(curvatures)


X1 = lt_ratios
X2 = emt_ratios

# If there are more than one features, combine them to one list
X = []
for i in range(len(X2)):
  # Create a feature vector by combining the values from X1 and X2
  x = (curvatures[i], X1[i], X2[i], udt_ratios[i])
  # Add the feature vector to the list
  X.append(x)


# Reshape if X is 1D array

y = filtered_smile_labels + filtered_smile_labels_t

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use random forest
model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
# # Fit the model to the training data
# model.fit(X_train, y_train)

# # Use linear SVC
# model = LinearSVC()
# model.fit(X_train, y_train)

# # Use k-nearest neighbors
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(X_train, y_train)


# k = 5
# scores = cross_val_score(model, X, y, cv=k)
# # Print the scores for each fold
# print("Scores for each fold: ", scores)
# # Print the mean score
# print("Mean score: ", scores.mean())


# # Evaluate the model on the test data
# y_train_pred = model.predict(X_train)
# train_score = accuracy_score(y_train, y_train_pred)
# print("Accuracy: {:.2f}".format(train_score))

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=6, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
# calculate mean and standard deviation
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

model.fit(X, y)
joblib.dump(model, 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/A2/A2random_forest_model.pkl')

# plot the learning curve
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.xlabel("Smile Training examples")
plt.ylabel("Score")
plt.ylim([0, 1])
plt.show()