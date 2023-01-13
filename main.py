import A1.TaskA1_gender as gender
import A2.TaskA2_smile as smile
import os
import joblib
from sklearn.metrics import accuracy_score
import time

def solve_taskA1A2():

    start = time.perf_counter()

    basedir_t = 'D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/Datasets/dataset_AMLS_22-23_test/celeba_test'
    # basedir_t = '../Datasets/dataset_AMLS_22-23_test/celeba_test'
    images_dir_t = os.path.join(basedir_t,"img")
    images_dir_t = images_dir_t.replace('\\', '/')
    labels_filename_t = 'labels.csv'
    landmarks_t, no_landmarks_t, filenames_t = gender.get_landmarks(images_dir_t)
    gender_labels_t = gender.get_gender(basedir_t, labels_filename_t)
    smile_labels_t = smile.get_smile(basedir_t, labels_filename_t)
    filtered_gender_labels_t = gender.filter(gender_labels_t, filenames_t, no_landmarks_t)
    filtered_smile_labels_t = smile.filter(smile_labels_t, filenames_t, no_landmarks_t)

    X1 = gender.get_gender_features(landmarks_t)
    y1 = filtered_gender_labels_t

    X2 = smile.get_smile_features(landmarks_t)
    y2 = filtered_smile_labels_t

    # model1 = joblib.load('./A1/random_forest_model.pkl')
    model1 = joblib.load('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/A1/random_forest_model.pkl')
    predictions1 = model1.predict(X1)
    accuracy1 = accuracy_score(y1, predictions1)
    # model2 = joblib.load('./A2/A2random_forest_model.pkl')
    model2 = joblib.load('D:/UCL 4th year/ELEC0134 Applied Machine Learning Systems 2223/final-assignment/AMLS_22-23 _SN19002774/A2/A2random_forest_model.pkl')
    predictions2 = model2.predict(X2)
    accuracy2 = accuracy_score(y2, predictions2)

    print("Accuracy: {:.2f}%".format(accuracy1 * 100))
    print("Accuracy: {:.2f}%".format(accuracy2 * 100))

    end = time.perf_counter()
    elapsed_time = end - start
    print('Elapsed time:', elapsed_time)

    return


def main():
    solve_taskA1A2()


if __name__ == "__main__":
    main()