# Hand Detection Project inspired from Anime : Jujutsu Kaisen [ Gojo Satoru ]

To download all of the required resources, use :
```pip install -r requirements.txt```

## Quick Overview of this Project
First step of this project was learn **Python OpenCV**. OpenCV is very useful when the project requires the usage of the device Webcam, and in my case, I make use of OpenCV to train the Hand Gesture Model

Second, learning how to use Google Mediapipe is also a key lesson because Google Mediapipe consist of all kinds of different models that can detect different types of objects like Hand Landmarks, Body Landmarks, Face Landmarks and etc. In this project, I have made use of the Face and Hand Landmarks model. Hand Landmarks model essentially help me to detect where my hands are and each of the Landmarks has a value and this value is used to train my own model

Lastly, Three.JS is heavily used for the front-end effects for each of the gestures. Making use of Three.JS Points, it creates this particles-like environment where each of the particles come together to create distinct effects. 

## Training of my own model
Because the hand gestures that I want is not really common, training of my own customised model was needed. Moreover because there was gestures consisting of only one hand and gestures of containing two hands, I decided to create 2 models 

Data collected was through a CSV file where the points of the Hand Landmarks are collected throughout the collection process and inserted into the CSV file. It is then used to train the 2 models. For the 1st Model (Single Hand Gesture), Random Tree Classifiers from sci-kit learn machine learning algorithm was used to train because there were many different gestures and I want to make use of Supervised Learning to help the model to distinguish between each of the gestures. For the 2nd Model (Duo Hand Gesture), since there was only gesture containing 2 hands, One-Class SVM Unsupervised Machine Learning algorithm was used to allow the model to know when the gesture will be "normal".

Data is also split into 80 & 20. 80% of the data is used for training while 20% is used to test and evaluate the model. Making use of a Confusion Matrix, I have also use it to evaluate the effectiveness of my models

## Starting the Program
Run the program from the ```testing_model.py``` file
