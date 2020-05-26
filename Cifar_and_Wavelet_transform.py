# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:50:33 2020

@author: nagar
"""

import os
os.getcwd()
os.chdir('C:\Python\pro')
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import pywt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

nClasses = 10
y_train = np_utils.to_categorical(y_train,nClasses)
y_test2 = np_utils.to_categorical(y_test,nClasses)

x_train.shape


#adv_x=np.load('Cifar10_adversaries_jsma.npy')

#adv_x=np.load('adv_cifar_fgsm.npy')
adv_x=np.load('Cifar10_adversaries_DF.npy')

## DWT for training and test
l1=list(map(lambda x: x.reshape(3,32,32), adv_x))
l2=list(map(lambda x: x.reshape(3,32,32), x_test)) 
adv_X=np.array(l1)   
X_test=np.array(l2)

tr=[]
for i in range(len(adv_X)):
    x=adv_X[i]
    for j in range(3):
        tr.append(x[j])
adv=np.array(tr)

te=[]
for i in range(len(X_test)):
    x=X_test[i]
    for j in range(3):
        te.append(x[j])
x_te=np.array(te)
x_te.shape
x_te1=np.zeros(shape=(120000,18,18))
c=0
for i in range(len(x_te)):
    x=x_te[i]
    y=adv[i]
    ll,(lh,hl,hh)=pywt.dwt2(x,'bior1.3',mode='symmetric')
    ll1,(lh1,hl1,hh1)=pywt.dwt2(y,'bior1.3',mode='symmetric')
    x_te1[c]=ll
    x_te1[c+1]=lh1
    x_te1[c+2]=hl1
    x_te1[c+3]=hh
    c=c+4


#IDWT 
y1=x_te1
X_test1=np.zeros(shape=(30000,32,32))
c=0
for i in range(len(X_test1)):
    ll=y1[c]
    lh=y1[c+1]
    hl=y1[c+2]
    hh=y1[c+3]
    coeffs=ll,(lh,hl,hh)
    X_test1[i]=pywt.idwt2(coeffs, 'bior1.3')
    c=c+4
X_test2= np.zeros(shape=(10000,3,32,32))
c=0   
for i in range(len(X_test2)):
    t=[]
    for j in range(3):
        t.append(X_test1[c])
        c=c+1
    X_test2[i]=np.array(t)

l2=list(map(lambda x: x.reshape(32,32,3), X_test2)) 
test=np.array(l2)



model=load_model('saved_models/cifar10_trained_model_v.h5')
scores = model.evaluate(test, y_test2, verbose=1)
print('Test accuracy:', scores[1]*100)
