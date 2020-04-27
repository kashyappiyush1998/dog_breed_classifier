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

global face_cascade
face_cascade = cv2.CascadeClassifier(os.path.dirname(__file__)+'/data/haarcascade_frontalface_alt.xml')

global graph
global model
loaded_model = keras.models.load_model(os.path.dirname(__file__)+'/data/Resnet_dog_detector.hdf5')
graph = tf.get_default_graph()

global new_model_include_top_false
new_model_include_top_false = keras.models.load_model(os.path.dirname(__file__) + '/data/Resnet_dog_detector.hdf5')
new_model_include_top_false.layers.pop()
new_model_include_top_false.layers.pop()
new_model_include_top_false = Model(new_model_include_top_false.input, new_model_include_top_false.layers[-1].output)
graph = tf.get_default_graph()

global model_best_weights
model_best_weights = keras.models.load_model(os.path.dirname(__file__) + '/data/weights.best.Resnet50.hdf5')
graph = tf.get_default_graph()

global dog_names
dog_names =  np.load(os.path.dirname(__file__) + '/data/dog_names.npy')

def extract_Resnet50(tensor):
    with graph.as_default():
        preds = new_model_include_top_false.predict(preprocess_input(tensor))
    return preds

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    with graph.as_default():
        preds = loaded_model.predict(img)
    return np.argmax(preds)

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

# print(predict_breed("dog.png"))

