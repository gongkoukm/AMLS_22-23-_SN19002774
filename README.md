# AMLS_22-23-_SN19002774
My project is used 68 landmarks extraction on faces, first extract all faces' landmarks for trainning, download the models, and access the models for testing. Therfore,
the test files are mess, but I still uploaded them. The final files are named: TaskA1_gender, TaskA2_smile, TaskB1_faceshape and TaskB2_eyecolor. I have contain the 
training methods in these files but with # to invalidate them, you can have a look.

You can simply click run in the main.py file to run the project, it will return the test accuracy for the four tasks. If there are errors, I think that should be problems
with file path, just change to correct path. I assume you put same format test datasets and labels in same way as the datasets for us to test during the project, I mean, 
same file name, same order and so on. 

The necessary packages are: cv2, dlib, os, re, csv, numpy, joblib, time, sklearn, matplotlib, math
