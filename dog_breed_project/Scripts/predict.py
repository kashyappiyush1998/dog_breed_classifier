import keras
import os
from keras.utils import np_utils
import numpy as np
import cv2                
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input, preprocess_input, decode_predictions   
from .preprocess import path_to_tensor 
from keras.models import Model

# %matplotlib inline
my_path = os.path.abspath(os.path.dirname(__file__))
print(my_path)
print("Print os path " + os.path.dirname(__file__))
face_cascade = cv2.CascadeClassifier(os.path.dirname(__file__)+'/data/haarcascade_frontalface_alt.xml')

loaded_model = keras.models.load_model(os.path.dirname(__file__)+'/data/Resnet_dog_detector.hdf5')
new_model_include_top_false = keras.models.load_model(os.path.dirname(__file__) + '/data/Resnet_dog_detector.hdf5')
new_model_include_top_false.layers.pop()
new_model_include_top_false.layers.pop()
new_model_include_top_false = Model(new_model_include_top_false.input, new_model_include_top_false.layers[-1].output)

model_best_weights = keras.models.load_model(os.path.dirname(__file__) + '/data/weights.best.Resnet50.hdf5')
dog_names =  np.load(os.path.dirname(__file__) + '/data/dog_names.npy')


def extract_Resnet50(tensor):
	return new_model_include_top_false.predict(preprocess_input(tensor))

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
        text = "This is image of a human. Resembling breed is - "
    elif (dog_detector(img_path)):
        text = "This image is of dog breed - "
    else:
        text = "Image does not belong to dog or human"
        return text
    
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    breed = dog_names[np.argmax(model_best_weights.predict(bottleneck_feature))]
    
    text = text + breed 
    return text

# print(predict_breed("dog.png"))

