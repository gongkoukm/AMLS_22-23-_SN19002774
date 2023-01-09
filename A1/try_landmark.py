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


def get_gender():
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
  print(filenames[:20])
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

    # # Loop through the faces
    # for face in faces:
    #   # Predict the face landmarks
    #   landmark = predictor(image, face)

    #   # Convert the landmarks to a list of (x, y) coordinates
    #   landmark = [(point.x, point.y) for point in landmark.parts()]

    #   # Add the landmarks to the list
    #   landmarks.append(landmark)

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


gender_labels = get_gender()
landmarks, no_landmarks, filenames = get_landmarks(images_dir)
# print(filenames[:20])
print(len(landmarks))
# print(no_landmarks[:20])
print(len(gender_labels))




# Filter the labels with no face detected
filtered_labels = []
# Check if the filename is not in the no_landmarks list
for label, filename in zip(gender_labels, filenames):
  if filename not in no_landmarks:
    filtered_labels.append(label)
print(filtered_labels[:20])


# Write landmarks in txt file
with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/landmarks.txt', 'w') as f1:
  # Clear the contents of the file
  f1.truncate(0)
  # Join the elements of the list into a single string, with a newline character as the separator
  for element in landmarks:
   f1.write(str(element) + '\n')

with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/labels.txt', 'w') as f2:
  # Clear the contents of the file
  f2.truncate(0)
  # Join the elements of the list into a single string, with a newline character as the separator
  for element in filtered_labels:
   f2.write(str(element) + '\n')

with open('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/filenames.txt', 'w') as f3:
  # Clear the contents of the file
  f3.truncate(0)
  # Join the elements of the list into a single string, with a newline character as the separator
  for element in no_landmarks:
   f3.write(str(element) + '\n')


# Read lists from txt files
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


gender_labels_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/labels.txt", "r")
  
with gender_labels_file as f:
  gender_labels = f.readlines()
# print(len(gender_labels))


# print(tuples[0][1][1])
def get_points(a,b):

  elements = []

  for t in tuples:
    elements.append(t[a][b])
  return elements


# Points of inner corners of the eyes(landmarks = 40,43)
x1 = get_points(39,0)
y1 = get_points(39,1)
x2 = get_points(42,0)
y2 = get_points(42,1)
# Points of tip of the nose and bottom of the chin(landmarks = 34,9)
x3 = get_points(33,0)
y3 = get_points(33,1)
x4 = get_points(8,0)
y4 = get_points(8,1)


# Calculate distance between two points for lists
def distance(x1, y1, x2, y2):
  distances = []
  for i in range(len(x1)):
    d = math.sqrt(math.pow(x1[i] - x2[i], 2) + math.pow(y1[i] - y2[i], 2))
    distances.append(d)
  return distances

eye_dis = distance(x1, y1, x2, y2)
nose_chin_dis = distance(x3, y3, x4, y4)


ratios = []
for i in range(len(eye_dis)):
  ratio = eye_dis[i] / nose_chin_dis[i]
  ratios.append(ratio)

# print(ratios) 


# X = ratios
# X = np.array(X)
# X = X.reshape(-1, 1)
# y = gender_labels

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the model
# model = RandomForestClassifier()

# # Fit the model to the training data
# model.fit(X_train, y_train)

# # Evaluate the model on the test data
# accuracy = model.score(X_test, y_test)
# print("Accuracy: {:.2f}".format(accuracy))
