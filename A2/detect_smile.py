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



basedir = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23/celeba'
images_dir = os.path.join(basedir,"img")
images_dir = images_dir.replace('\\', '/')
labels_filename = 'labels.csv'


def get_smile():
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


filenames = (os.listdir(images_dir))
filenames.sort(key=lambda x: int(x.split(".")[0]))
smile_labels = get_smile()

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


# gender_labels_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/labels.txt", "r")

# with gender_labels_file as f:
#   gender_labels = f.readlines()

no_filenames_file = open("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/filenames.txt", "r")
with no_filenames_file as f:
  no_landmarks = f.readlines()
no_landmarks = [label.strip() for label in no_landmarks]

# Filter the labels with no face detected
filtered_smile_labels = []
# Check if the filename is not in the no_landmarks list
for label, filename in zip(smile_labels, filenames):
  if filename not in no_landmarks:
    filtered_smile_labels.append(label)
# print(filtered_smile_labels[:20])
# print(no_landmarks)
# print(len(landmarks))



def get_points(a,b):

  elements = []

  for t in landmarks:
    elements.append(t[a][b])
  return elements


# Points of corners of the mouth(landmarks = 49,55)
x1 = get_points(48,0)
y1 = get_points(48,1)
x2 = get_points(54,0)
y2 = get_points(54,1)
# Points of the temple(landmarks = 1,17)
x3 = get_points(0,0)
y3 = get_points(0,1)
x4 = get_points(16,0)
y4 = get_points(16,1)


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
x5 = get_points(41,0)
y5 = get_points(41,1)
x6 = get_points(46,0)
y6 = get_points(46,1)

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


# Points of the bottom of eyes(landmarks = 52,58)
x7 = get_points(51,0)
y7 = get_points(51,1)
x8 = get_points(57,0)
y8 = get_points(57,1)
lip_updown = distance(x7, y7, x8, y8)
lt_ratios = []
for i in range(len(lip_updown)):
  ratio = lip_updown[i] / temple_width[i]
  lt_ratios.append(ratio)


# Need landmark 49(x1,y1)),67,55(x2,y2) to calculate the curvature of mouth
x9 = get_points(66,0)
y9 = get_points(66,1)
point1 = [x1, y1]
point2 = [x9, y9] 
point3 = [x2, y2] 
curvatures = getAngle(x1, y1, x9, y9, x2, y2)
curvatures = np.abs(curvatures)
# print(curvatures[:30])
# print(filtered_smile_labels[:30])

# # Remove '\n' newline characters
# gender_labels = [label.strip() for label in gender_labels]
# colors = []
# for label in gender_labels:
#     if label == 1:
#         colors.append("red")
#     else:
#         colors.append("blue")


# # print(curvatures[:30])    
# print(gender_labels[:30])
# print(no_filenames)


# print(gender_labels[4])
# print(colors[:30])
# # Create the plot
# plt.scatter(curvatures, [0]*len(curvatures), c=colors, cmap='RdYlBu')

# # Show the plot
# plt.show()













# X1 = lt_ratios
# X2 = emt_ratios

# X = []
# for i in range(len(X1)):
#   # Create a feature vector by combining the values from X1 and X2
#   x = (X1[i], X2[i])
#   # Add the feature vector to the list
#   X.append(x)


# Reshape if X is 1D array
curvatures = np.array(curvatures)
curvatures = curvatures.reshape(-1, 1)
# print(lt_ratios[:20])
y = filtered_smile_labels

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(curvatures, y, test_size=0.2, random_state=42)

# # # # Use random forest
# # # model = RandomForestClassifier()
# # # # Fit the model to the training data
# # # model.fit(X_train, y_train)

# # # # Use linear SVC
# # # model = LinearSVC()
# # # model.fit(X_train, y_train)

# Use k-nearest neighbors
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)


# Evaluate the model on the test data
y_pred = model.predict(X_test)
accuracy1 = model.score(X_test, y_test)
accuracy2 = model.score(X_test, y_test)
print("Accuracy: {:.2f}".format(accuracy1))
print("Accuracy: {:.2f}".format(accuracy2))