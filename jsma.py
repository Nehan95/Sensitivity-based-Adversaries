# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:14:56 2019

@author: nagar

Genrating Adversaries of MNIST  from JSMA attack
"""
# Importing libraries
import os
os.getcwd()
os.chdir('C:\Python\pro')
import pandas as pd
import tensorflow as tf
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils import other_classes
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import numpy as np
import keras
import random
from keras import backend
import matplotlib.pyplot as plt
#tf.set_random_seed(1234)
#sess = tf.Session()
#keras.backend.set_session(sess)

backend.set_learning_phase(False)
sess =  backend.get_session()
#Loading MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshaping the training and test data
image_vector_size = 28*28
X_train = x_train.reshape(x_train.shape[0], image_vector_size).astype('float32')
X_test = x_test.reshape(x_test.shape[0], image_vector_size).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
X_train.shape

#loading a simple model trained and saved previously
keras_model=load_model('simple_neural_network.h5')
predicted_classes = keras_model.predict_classes(X_test)

#Checking the accuracy and classification report with previously trained model
new=np.nonzero(Y_test)
results=pd.DataFrame()
results['index']=new[0]
results['Actual']=new[1]
results['Predicted']=predicted_classes
actual=results['Actual']
predicted=results['Predicted']
results1 = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results1) 
print ('Accuracy Score :',accuracy_score(actual, predicted)) 
print ('Report : ')
print (classification_report(actual, predicted))

#Creating a wrapper for keras model
wrap = KerasModelWrapper(keras_model)

#Instantiate tensorflow placeholders
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))


# Instantiate a SaliencyMapMethod attack object
jsma = SaliencyMapMethod(wrap, sess=sess)
jsma_params = {'theta': 1., 
               'gamma': 0.1,
               'clip_min': 0., 
               'clip_max': 1.,
               'y_target': None}




#Genrating adversaries for jsma
adv=X_test
for index in range((len(X_test))):
    sample = X_test[index: index + 1]
    current = int(np.argmax(Y_test[index]))
    target_classes = other_classes(10, current)
    target=random.choice(target_classes)
    one_hot_target = np.zeros((1, 10), dtype=np.float32)
    one_hot_target[0, target] = 1
    jsma_params['y_target'] = one_hot_target
    adv_x = jsma.generate_np(sample, **jsma_params)
    adv[index]=adv_x
    
   
        
#predicting the classes for images generated from JSMA attack with the loaded network       
predicted_classes = keras_model.predict_classes(adv)

#Checking accuracy and classification report of adversaries
new=np.nonzero(Y_test)
results=pd.DataFrame()
results['index']=new[0]
results['Actual']=new[1]
results['Predicted']=predicted_classes
actual=results['Actual']
predicted=results['Predicted']
results1 = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results1) 
print ('Accuracy Score :',accuracy_score(actual, predicted)) 
print ('Report : ')
print (classification_report(actual, predicted))

##Saving the adversaries for further use in a numpy file
np.save('adv_jsma',adv)

#loading the numpy file to load the adversaries and for verification
new_adv=np.load('adv_jsma.npy')
predicted_classes = keras_model.predict_classes(new_adv)
new=np.nonzero(Y_test)
results=pd.DataFrame()
results['index']=new[0]
results['Actual']=new[1]
results['Predicted']=predicted_classes
actual=results['Actual']
predicted=results['Predicted']
results1 = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results1) 
print ('Accuracy Score :',accuracy_score(actual, predicted)) 
print ('Report : ')
print (classification_report(actual, predicted))


#Visualizing the adversaries

fig = plt.figure()

plt.subplot(1,2,1)
plt.tight_layout()
plt.imshow(X_test[0].reshape((28, 28)), cmap='gray', interpolation='none')
plt.title("Digit: {}".format((np.argmax(Y_test[0]))))
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.tight_layout()
plt.imshow(adv[0].reshape((28, 28)), cmap='gray', interpolation='none')
plt.title("Digit: {}".format(predicted[0]))
plt.xticks([])
plt.yticks([])
fig
