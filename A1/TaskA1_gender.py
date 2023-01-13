import cv2
import dlib
import os
import math
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
import joblib


# basedir_t = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23_test/celeba_test'
basedir_t = '../Datasets/dataset_AMLS_22-23_test/celeba_test'
images_dir_t = os.path.join(basedir_t,"img")
images_dir_t = images_dir_t.replace('\\', '/')
labels_filename_t = 'labels.csv'


# Get the gender labels from label.csv
def get_gender(basedir, labels_filename):
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

    return gender_list


def get_landmarks(folder):
    # Load the shape predictor model
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    # predictor = dlib.shape_predictor("D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/shape_predictor_68_face_landmarks.dat")

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


# Filter the labels with no face detected
def filter(gender_labels, filenames, no_landmarks):
    filtered_gender_labels = []
    # Check if the filename is not in the no_landmarks list
    for label, filename in zip(gender_labels, filenames):
        if filename not in no_landmarks:
            filtered_gender_labels.append(label)\

    return filtered_gender_labels


# Get the features need for A1
def get_gender_features(landmarks):

    def get_points(a, b, tuples):

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

    x1 = get_points(48,0,landmarks)
    y1 = get_points(48,1,landmarks)
    x2 = get_points(54,0,landmarks)
    y2 = get_points(54,1,landmarks)
    x3 = get_points(22,0,landmarks)
    y3 = get_points(22,1,landmarks)
    x4 = get_points(26,0,landmarks)
    y4 = get_points(26,1,landmarks)
    x5 = get_points(21,0,landmarks)
    y5 = get_points(21,1,landmarks)
    x6 = get_points(17,0,landmarks)
    y6 = get_points(17,1,landmarks)
    x7 = get_points(42,0,landmarks)
    y7 = get_points(42,1,landmarks)
    x8 = get_points(45,0,landmarks)
    y8 = get_points(45,1,landmarks)
    x9 = get_points(39,0,landmarks)
    y9 = get_points(39,1,landmarks)
    x10 = get_points(36,0,landmarks)
    y10 = get_points(36,1,landmarks)
    x11 = get_points(27,0,landmarks)
    y11 = get_points(27,1,landmarks)
    x12 = get_points(33,0,landmarks)
    y12 = get_points(33,1,landmarks)
    x13 = get_points(4,0,landmarks)
    y13 = get_points(4,1,landmarks)
    x14 = get_points(12,0,landmarks)
    y14 = get_points(12,1,landmarks)
    
    X = []
    for i in range(len(x1)):
        # Create a feature vector by combining the values 
        x = (x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i], x9[i], x10[i], x11[i], x12[i], x13[i], x14[i])
        # Add the feature vector to the list
        X.append(x)

    return X


# model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=0)

# train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=6, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# # calculate mean and standard deviation
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# model.fit(X, y)
# joblib.dump(model, 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/A1/random_forest_model.pkl')

# # plot the learning curve
# plt.grid()
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.legend(loc="best")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.ylim([0, 1])
# plt.show()