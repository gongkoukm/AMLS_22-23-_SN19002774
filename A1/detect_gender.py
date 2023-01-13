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
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
import joblib


# Load the shape predictor model

basedir = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23/celeba'
basedir_t = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23_test/celeba_test'
images_dir = os.path.join(basedir,"img")
images_dir = images_dir.replace('\\', '/')
images_dir_t = os.path.join(basedir_t,"img")
images_dir_t = images_dir_t.replace('\\', '/')
labels_filename = 'labels.csv'
labels_filename_t = 'labels.csv'



def get_gender(basedir, labels_filename):
    # Get all celeba's image paths
    # image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    with open(os.path.join(basedir, labels_filename), 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        # Skip the first row
        next(reader)

        gender_list = []
  
        for row in reader:
            value = row[0]
    
            # Split the value into parts
            parts = re.split("\s+", value)
    
            # parts[2] represents the third value, which is the label of gender
            gender_label = parts[2]
            gender_list += [gender_label]
            # print(type(gender_label))
            # print(gender_label, end=",")

    return gender_list


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
gender_labels = get_gender(basedir, labels_filename)
gender_labels_t = get_gender(basedir_t, labels_filename_t)
# landmarks, no_landmarks, filenames = get_landmarks(images_dir)
landmarks_t, no_landmarks_t, filenames_t = get_landmarks(images_dir_t)


# # Write landmarks in txt file
# with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/landmarks.txt', 'w') as f1:
#   # Clear the contents of the file
#   f1.truncate(0)
#   # Join the elements of the list into a single string, with a newline character as the separator
#   for element in landmarks:
#    f1.write(str(element) + '\n')

# with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/labels.txt', 'w') as f2:
#   # Clear the contents of the file
#   f2.truncate(0)
#   # Join the elements of the list into a single string, with a newline character as the separator
#   for element in filtered_labels:
#    f2.write(str(element) + '\n')

# with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/filenames.txt', 'w') as f3:
#   # Clear the contents of the file
#   f3.truncate(0)
#   # Join the elements of the list into a single string, with a newline character as the separator
#   for element in no_landmarks:
#    f3.write(str(element) + '\n')


# Read lists from txt files
no_filenames_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/filenames.txt", "r")
with no_filenames_file as f:
  no_landmarks = f.readlines()
no_landmarks = [label.strip() for label in no_landmarks]

def filter(gender_labels, filenames, no_landmarks):
  # Filter the labels with no face detected
  filtered_gender_labels = []
  # Check if the filename is not in the no_landmarks list
  for label, filename in zip(gender_labels, filenames):
    if filename not in no_landmarks:
      filtered_gender_labels.append(label)
  return filtered_gender_labels

filtered_gender_labels = filter(gender_labels, filenames, no_landmarks)
filtered_gender_labels_t = filter(gender_labels_t, filenames_t, no_landmarks_t)


# opening the landmarks.txt file in read mode
landmarks_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/landmarks.txt", "r")
  
with landmarks_file as f:
  lines = f.readlines()
# print(len(lines))

# Initialize an empty list to store the tuples
tuples = []

# Loop over the lines and parse each line using ast.literal_eval
for line in lines:
  t = ast.literal_eval(line)
  tuples.append(t)

landmarksall = tuples + landmarks_t
# gender_labels_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/labels.txt", "r")
  
# with gender_labels_file as f:
#   gender_labels = f.readlines()
# # print(len(gender_labels))


# print(tuples[0][1][1])
def get_points(a,b,tuples):

  elements = []

  for t in tuples:
    elements.append(t[a][b])
  return elements


x1 = get_points(48,0,landmarksall)
y1 = get_points(48,1,landmarksall)
x2 = get_points(54,0,landmarksall)
y2 = get_points(54,1,landmarksall)
x3 = get_points(22,0,landmarksall)
y3 = get_points(22,1,landmarksall)
x4 = get_points(26,0,landmarksall)
y4 = get_points(26,1,landmarksall)
x5 = get_points(21,0,landmarksall)
y5 = get_points(21,1,landmarksall)
x6 = get_points(17,0,landmarksall)
y6 = get_points(17,1,landmarksall)
x7 = get_points(42,0,landmarksall)
y7 = get_points(42,1,landmarksall)
x8 = get_points(45,0,landmarksall)
y8 = get_points(45,1,landmarksall)
x9 = get_points(39,0,landmarksall)
y9 = get_points(39,1,landmarksall)
x10 = get_points(36,0,landmarksall)
y10 = get_points(36,1,landmarksall)
x11 = get_points(27,0,landmarksall)
y11 = get_points(27,1,landmarksall)
x12 = get_points(33,0,landmarksall)
y12 = get_points(33,1,landmarksall)
# jaw
x13 = get_points(4,0,landmarksall)
y13 = get_points(4,1,landmarksall)
x14 = get_points(12,0,landmarksall)
y14 = get_points(12,1,landmarksall)


# Calculate distance between two points for lists
def distance(x1, y1, x2, y2):
  distances = []
  for i in range(len(x1)):
    d = math.sqrt(math.pow(x1[i] - x2[i], 2) + math.pow(y1[i] - y2[i], 2))
    distances.append(d)
  return distances


X = []
for i in range(len(x1)):
  # Create a feature vector by combining the values from X1 and X2
  x = (x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i], x9[i], x10[i], x11[i], x12[i], x13[i], x14[i] )
  # Add the feature vector to the list
  X.append(x)
print(X[:10])

y = filtered_gender_labels + filtered_gender_labels_t
print(y[:10])
# Initialize the model
model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=0)

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
joblib.dump(model, 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/A1/random_forest_model.pkl')

# plot the learning curve
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.xlabel("Gender Training examples")
plt.ylabel("Score")
plt.ylim([0, 1])
plt.show()