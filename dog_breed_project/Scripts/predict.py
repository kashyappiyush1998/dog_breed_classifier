import keras
import os
from keras.utils import np_utils
import numpy as np
import cv2                
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input, preprocess_input, decode_predictions   
from preprocess import path_to_tensor 

# %matplotlib inline

face_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_alt.xml')
loaded_model = keras.models.load_model("../data/Resnet_dog_detector.hdf5")

Resnet50_model = keras.models.load_model('../data/weights.best.Resnet50.hdf5')
dog_names =  np.load("../data/dog_names.npy")

def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = extract_Resnet50.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(loaded_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def predict_breed(img_path):
    if(face_detector(img_path)):
        print("This is image of a human. Resembling breed is - ", end=" ")
    elif (dog_detector(img_path)):
        print("This image is of dog breed -", end=" ")
    else:
        print("Image does not belong to dog or human")
        return
    
    breed = Resnet50_predict_breed(img_path)
    
    print(breed)
    return

print(dog_detector("dog.png"))

