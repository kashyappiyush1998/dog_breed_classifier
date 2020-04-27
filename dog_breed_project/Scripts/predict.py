import keras
import os
from keras.utils import np_utils
import numpy as np
import cv2                
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input, preprocess_input, decode_predictions   
from .preprocess import path_to_tensor 
from keras.models import Model
import tensorflow as tf

# %matplotlib inline

# import frontal face for human face detection
global face_cascade
face_cascade = cv2.CascadeClassifier(os.path.dirname(__file__)+'/data/haarcascade_frontalface_alt.xml')

# creates graph for streamining all loaded models
global graph

#import model to be used for dog detection
global loaded_model
loaded_model = keras.models.load_model(os.path.dirname(__file__)+'/data/Resnet_dog_detector.hdf5')
graph = tf.get_default_graph()

# import model to be used for extracting Resnet50 bottleneck features
global new_model_include_top_false
new_model_include_top_false = keras.models.load_model(os.path.dirname(__file__) + '/data/Resnet_dog_detector.hdf5')
new_model_include_top_false.layers.pop()
new_model_include_top_false.layers.pop()
new_model_include_top_false = Model(new_model_include_top_false.input, new_model_include_top_false.layers[-1].output)
graph = tf.get_default_graph()

# import model we trained that resulted in best accuracy for dog prediction
global model_best_weights
model_best_weights = keras.models.load_model(os.path.dirname(__file__) + '/data/weights.best.Resnet50.hdf5')
graph = tf.get_default_graph()

#import dog breeds stored as numpy file
global dog_names
dog_names =  np.load(os.path.dirname(__file__) + '/data/dog_names.npy')

def extract_Resnet50(tensor):

    ''' Gets a tensor and returns bottle neck features '''

    with graph.as_default():
        preds = new_model_include_top_false.predict(preprocess_input(tensor))
    return preds

def ResNet50_predict_labels(img_path):

    ''' takes image path as input and returns prediction vector for image located at img_path '''

    img = preprocess_input(path_to_tensor(img_path))
    with graph.as_default():
        preds = loaded_model.predict(img)
    return np.argmax(preds)

def dog_detector(img_path):

    ''' returns "True" if a dog is detected in the image stored at img_path else "False" '''

    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img_path):

    ''' detect if image stored at image path contains human and returns number of faces detected '''

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def predict_breed(img_path):

    ''' Takes input as image path and return the string suitable, examples given below
    Human -  "This is image of a human. Resembling breed is - Mastiff"
    Dog - "This image is of dog breed - Mastiff"
    Neither dog nor human - "Image does not belong to dog or human"
    '''

    if(face_detector(img_path)):
        text = "This is image of a human. Resembling breed is - "
    elif (dog_detector(img_path)):
        text = "This image is of dog breed - "
    else:
        text = "Image does not belong to dog or human"
        return text
    
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    with graph.as_default():
        breed_class = model_best_weights.predict(bottleneck_feature)
    breed = dog_names[np.argmax(breed_class)]
    
    text = text + breed 
    return text