# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:08:00 2019

@author: nagar
"""

###############################################################################
## Importing required libraries
import os
os.getcwd()
os.chdir('C:\Python\pro')
import SensitivityFunction as sf
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import pandas as pd
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend
import random


## Importing MNIST dataset, splitting the data into training and test sets
##  preprocessing the data for neural network  training and testing 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
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

###############################################################################
## loading a previously trained model

model = load_model('model.h5')
model_60pixels=load_model('model_60pixels_changed.h5')
test_loss,test_accuracy=model.evaluate(X_test, Y_test, verbose=False)   
print(test_accuracy)
print(test_loss)
weights=model.get_weights()
weights=model_60pixels.get_weights()
print(weights)

w=[weights[0],weights[2],weights[4]]
b=[weights[1],weights[3],weights[5]]

## predicting classes for original test data
predicted_classes = model.predict_classes(X_test)
predicted_classes = model_60pixels.predict_classes(X_test)

## Accuracy of network with original test data 
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


##############################################################################
## changing a number of pixels in the test data based on 
## highly sensitive neurons in the input layern=int(input('Enter number of pixels:'))

l=[]
n=int(input('Enter number of pixels:'))
for i in range(len(X_test)):
    inpt=X_test[i]
    neuron=sf.sensitive_neuron(inpt,w,b,Y_test[i])
    l.append(neuron)
    wt=w[0][:,neuron]
    X_test[i]=sf.image_pixels(inpt,wt,n)
    

    
predicted_classes_new = model_60pixels.predict_classes(X_test)
results['Predicted_new']=predicted_classes_new
predicted_new=results['Predicted_new']


results2 = confusion_matrix(actual, predicted_new) 
print ('Confusion Matrix :')
print(results2) 
print ('Accuracy Score :',accuracy_score(actual, predicted_new)) 
print ('Report : ')
print (classification_report(actual, predicted_new))

###############################################################################
## generating adversarial images with fgsm attack
backend.set_learning_phase(False)
sess =  backend.get_session()
keras_model=load_model('simple_nn_retrain_iccad.h5')
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.4,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(X_test, **fgsm_params)  #FGSM attack X_test
np.save('adv_mnist_fgsm',adv_x)
adv_x=np.load('adv_jsma.npy')  #jsma attack X_test
l=[]
n=int(input('Enter number of pixels:'))
for i in range(len(adv_x)):
    inpt=adv_x[i]
    neuron=sf.sensitive_neuron(inpt,w,b,Y_test[i])
    l.append(neuron)
    wt=w[0][:,neuron]
    adv_x[i]=sf.image_pixels(inpt,wt,n)
## predicting classes for adversarial data
predicted_classes_adv = model_60pixels.predict_classes(adv_x)
results['Predicted_adv']=predicted_classes_adv
Predicted_adv=results['Predicted_adv']

## Accuracy of the network on adversarial images
results3 = confusion_matrix(actual, Predicted_adv) 
print ('Confusion Matrix :')
print(results3) 
print ('Accuracy Score :',accuracy_score(actual, Predicted_adv)) 
print ('Report : ')
print (classification_report(actual, Predicted_adv))
###############################################################################
## mixing some percent of adversarial images to  original test data
inpt1=int(input('Enter % of adversarial images:'))
no=int((inpt1/100)*len(X_test))
for i in range(no):
    x=random.randrange(0,9999)
    X_test[x]=adv_x[x]



## removing the pixels from newly formed dataset which has combination of 
## original images and some percent of adversarial images

n=int(input('Enter number of pixels:'))
for i in range(len(X_test)):
    inpt=X_test[i]
    neuron=sf.sensitive_neuron(inpt,w,b,Y_test[i])
    wt=w[0][:,neuron]
    X_test[i]=sf.image_pixels(inpt,wt,n)

predicted_mix_data = model.predict_classes(X_test)
results['Predicted_mix']=predicted_mix_data
Predicted_mix=results['Predicted_mix']


results4 = confusion_matrix(actual, Predicted_mix) 
print ('Confusion Matrix :')
print(results4) 
print ('Accuracy Score :',accuracy_score(actual, Predicted_mix)) 
print ('Report : ')
print (classification_report(actual, Predicted_mix))    

###############################################################################

## adding some percent of adversarial images to  original test data
inpt1=int(input('Enter % of adversarial images:'))
no=int((inpt1/100)*len(X_test))
    
X_test=np.append(X_test,adv_x[0:no],axis=0)
Y_test=np.append(Y_test,Y_test[0:no],axis=0)
len(Y_test)
    
l1=[]
n=int(input('Enter number of pixels:'))
for i in range(len(X_test)):
    inpt=X_test[i]
    neuron=sf.sensitive_neuron(inpt,w,b,Y_test[i])
    l1.append(neuron)
    wt=w[0][:,neuron]
    X_test[i]=sf.image_pixels(inpt,wt,n)

predicted_add_data = model_60pixels.predict_classes(X_test)
#predicted_add_data = model.predict_classes(X_test)

len(predicted_add_data)

new1=np.nonzero(Y_test)
results_add=pd.DataFrame()
results_add['index']=new1[0]
len(results_add)
results_add['Actual']=new1[1]
results_add['Predicted']=predicted_add_data
new_actual=results_add['Actual']
Predicted_add=results_add['Predicted']
results5 = confusion_matrix(new_actual, Predicted_add) 
print ('Confusion Matrix :')
print(results5) 
print ('Accuracy Score :',accuracy_score(new_actual, Predicted_add)) 
print ('Report : ')
print (classification_report(new_actual, Predicted_add))   
