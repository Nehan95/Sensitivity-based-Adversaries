# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:12:18 2020

@author: nagar
"""

import os
os.getcwd()
os.chdir('C:\Python\pro')
import pywt
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.utils import np_utils

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
X_train = np.reshape(X_train, (len(X_train), 28, 28))  # adapt this if using `channels_first` image data format
X_test = np.reshape(X_test, (len(X_test), 28, 28))

#adv_x=np.load('adv_jsma.npy')
adv_x=np.load('adv_mnist_fgsm.npy')
'''
adv_x.shape
adv_x=adv_x.reshape(10000,28,28)
adv_x = adv_x.reshape(-1, 28, 28)
adv_x=np.stack([adv_x, adv_x, adv_x], axis=-1)
'''
#adv_x=np.load('mnist_adversaries_DF.npy')
adv_x=adv_x.reshape(adv_x.shape[0], 28,28)

x_te=np.zeros(shape=(40000,16,16))
c=0
for i in range(len(X_test)):
    x=X_test[i]
    z=adv_x[i]
    ll,(lh,hl,hh)=pywt.dwt2(x,'bior1.3',mode='symmetric')
    ll1,(lh1,hl1,hh1)=pywt.dwt2(z,'bior1.3',mode='symmetric')
    x_te[c]=ll1
    x_te[c+1]=lh
    x_te[c+2]=hl
    x_te[c+3]=hh1
    c=c+4
    
y=x_te
x_test=np.zeros(shape=(10000,28,28))
c=0
for i in range(len(x_test)):
    ll=y[c]
    lh=y[c+1]
    hl=y[c+2]
    hh=y[c+3]
    coeffs=ll,(lh,hl,hh)
    x_test[i]=pywt.idwt2(coeffs, 'bior1.3')
    c=c+4

x_test=np.reshape(x_test,[10000,784])
model123 = load_model('simple_neural_network.h5')
#model123 = load_model('MNIST_Densennet_model.h5')    
pred=model123.predict(x_test)
predicted = [np.argmax(i) for i in pred]


new=np.nonzero(Y_test)
results=pd.DataFrame()
results['index']=new[0]
results['Actual']=new[1]
actual=results['Actual']
print ('Accuracy Score :',accuracy_score(actual, predicted)*100) 