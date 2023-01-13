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


basedir_t = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23_test/celeba_test'
# basedir_t = '../Datasets/dataset_AMLS_22-23_test/celeba_test'
images_dir_t = os.path.join(basedir_t,"img")
images_dir_t = images_dir_t.replace('\\', '/')
labels_filename_t = 'labels.csv'


# Get the smile labels from label.csv
def get_smile(basedir, labels_filename):
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
    
            # parts[3] represents the third value, which is the label of smile
            smile_label = parts[3]
            smile_list += [smile_label]

    return smile_list


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

    # Get the current performance counter value
    start = time.perf_counter()

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

    # Get the current performance counter value
    end = time.perf_counter()

    # Compute the elapsed time
    elapsed_time = end - start

    # Print the elapsed time
    print('Elapsed time:', elapsed_time)

    # Return the landmarks
    return landmarks, no_landmarks, filenames


# Filter the labels with no face detected
def filter(smile_labels, filenames, no_landmarks):
    
    filtered_smile_labels = []
    # Check if the filename is not in the no_landmarks list
    for label, filename in zip(smile_labels, filenames):
        if filename not in no_landmarks:
            filtered_smile_labels.append(label)
    return filtered_smile_labels


# Get the features need for A2
def get_smile_features(landmarks):

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


    # Calculate angle between three points
    def getAngle(x1,y1,x2,y2,x3,y3):
        angle = []
        for i in range(len(x1)):
            ang = 180-(math.degrees(math.atan2(abs(y3[i]-y2[i]), abs(x3[i]-x2[i])) + math.atan2(abs(y1[i]-y2[i]), abs(x1[i]-x2[i]))))
            angle.append(ang)
        return angle


    x1 = get_points(48,0,landmarks)
    y1 = get_points(48,1,landmarks)
    x2 = get_points(54,0,landmarks)
    y2 = get_points(54,1,landmarks)
    x3 = get_points(0,0,landmarks)
    y3 = get_points(0,1,landmarks)
    x4 = get_points(16,0,landmarks)
    y4 = get_points(16,1,landmarks)
    x5 = get_points(41,0,landmarks)
    y5 = get_points(41,1,landmarks)
    x6 = get_points(46,0,landmarks)
    y6 = get_points(46,1,landmarks)
    x7 = get_points(51,0,landmarks)
    y7 = get_points(51,1,landmarks)
    x8 = get_points(57,0,landmarks)
    y8 = get_points(57,1,landmarks)
    x9 = get_points(66,0,landmarks)
    y9 = get_points(66,1,landmarks)

    lip_width = distance(x1, y1, x2, y2)
    temple_width = distance(x3, y3, x4, y4)
    eyemouth_dis = []
    eyemouth_dis_left = distance(x5, y5, x1, y1)
    eyemouth_dis_right = distance(x6, y6, x2, y2)
    for i in range(len(eyemouth_dis_left)):
        emd = eyemouth_dis_left[i] + eyemouth_dis_right[i]
        eyemouth_dis.append(emd)
    # Calculate the ratio of lip width to temple width 
    # Temple width is constant no matter smile or not
    lt_ratios = []
    for i in range(len(lip_width)):
        ratio = lip_width[i] / temple_width[i]
        lt_ratios.append(ratio)
    # Calculate the ratio of eyemouth distance to temple width 
    emt_ratios = []
    for i in range(len(eyemouth_dis)):
        ratio = eyemouth_dis[i] / temple_width[i]
        emt_ratios.append(ratio)
    lip_updown = distance(x7, y7, x8, y8)
    # Calculate the ratio of thickness of lip to temple width 
    udt_ratios = []
    for i in range(len(lip_updown)):
        ratio = lip_updown[i] / temple_width[i]
        udt_ratios.append(ratio)
    # Calculate the curvature of mouth
    curvatures = getAngle(x1, y1, x9, y9, x2, y2)
    curvatures = np.abs(curvatures)
    
    X = []
    for i in range(len(x1)):
        # Create a feature vector by combining the values 
        x = (curvatures[i], lt_ratios[i], emt_ratios[i], udt_ratios[i])
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