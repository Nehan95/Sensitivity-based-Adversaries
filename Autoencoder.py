# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:17:47 2019

@author: nagar
"""
'''
The code below is used for testing the classification accuracy of a pre trained network on MNIST data.
The accuracy of the network is 98%. The MNIST data and its adversaries are passed through DWT --> Variational Autoencoder --> IDWT .
The acuuracy of reconstructed MNIST images is same as mentioned previously. 
Next, Adversarial MNIST data is passed through same steps and reconstructed data is tested with the pre trained network to give same accuracy as before.
'''

###############################################################################
## Importing required libraries
import os
os.getcwd()
os.chdir('C:\Python\pro')
import pywt
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Lambda, Dense,Dropout, Activation
from keras.models import Model,Sequential
from keras import backend as K
from keras.losses import mse
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend


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
X_train = np.reshape(X_train, (len(X_train), 28, 28))  # adapt this if using `channels_first` image data format
X_test = np.reshape(X_test, (len(X_test), 28, 28))  # adapt this if using `channels_first` image data format


X_test.shape

###############################################################################

## DWT & VAE for X_test######

adv_x=np.load('adv_jsma.npy')
def pixels(inpt,n):
    prod=inpt
    dat=pd.DataFrame()
    dat['prod']=list(prod)
    index1=dat.sort_values(['prod'],ascending=False)    
    list_of_indexes=index1.index.tolist()
    y=list_of_indexes[0:n]
    for i in range(len(y)):
        inpt[y[i]]=0.0
    return inpt

n=int(input('Enter number of pixels:'))
for i in range(len(adv_x)):
    adv_x[i]=pixels(adv_x[i],n)


    
X_train = np.reshape(X_train, (len(X_train), 28, 28))

x_tr=np.zeros(shape=(240000,16,16))
c=0
for i in range(len(X_train)):
    x=X_train[i]
    ll,(lh,hl,hh)=pywt.dwt2(x,'bior1.3',mode='symmetric')
    x_tr[c]=ll
    x_tr[c+1]=lh
    x_tr[c+2]=hl
    x_tr[c+3]=hh
    c=c+4
        
       
    


adv_x=adv_x.reshape(adv_x.shape[0], 28,28)

X_test=X_test.reshape(X_test.shape[0], 28,28)

x_te=np.zeros(shape=(40000,16,16))
c=0
for i in range(len(adv_x)):
    x=adv_x[i]
    ll,(lh,hl,hh)=pywt.dwt2(x,'bior1.3',mode='symmetric')
    x_te[c]=ll
    x_te[c+1]=lh
    x_te[c+2]=hl
    x_te[c+3]=hh
    c=c+4


     
image_size = x_tr.shape[1]
original_dim = image_size * image_size
x_tr = np.reshape(x_tr, [-1, original_dim])
x_te = np.reshape(x_te, [-1, original_dim])
x_tr[0].shape
# network parameters
input_shape = (original_dim, )
intermediate_dim = 256
batch_size = 64
latent_dim = 2
epochs = 50



def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
models = (encoder, decoder)
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
vae.fit(x_tr,epochs=epochs,batch_size=batch_size, validation_data=(x_te, None))

latentA, latentB, latentC =encoder.predict(x_te)


decoded_x=decoder.predict(latentA)
decoded_x.shape

y=np.reshape(decoded_x,[40000,16,16])    
#y=np.reshape(x_te,[40000,16,16]) 
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
predicted_classes = model123.predict_classes(x_test)


new=np.nonzero(Y_test)
results=pd.DataFrame()
results['index']=new[0]
results['Actual']=new[1]
results['Predicted']=predicted_classes
actual=results['Actual']
predicted=results['Predicted']
print ('Accuracy Score :',accuracy_score(actual, predicted)*100) 


###############################################################################
## DWT & VAE for adv testing

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
## generating adversarial images with fgsm attack
backend.set_learning_phase(False)
sess =  backend.get_session()
keras_model=load_model('simple_nn_retrain_iccad.h5')
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.7,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(X_test, **fgsm_params)

adv_x.shape

adv_x = np.reshape(adv_x, (len(adv_x), 28, 28))



adv_te=np.zeros(shape=(40000,16,16))
c=0
for i in range(len(adv_x)):
    x=adv_x[i]
    ll,(lh,hl,hh)=pywt.dwt2(x,'bior1.3',mode='symmetric')
    adv_te[c]=ll
    adv_te[c+1]=lh
    adv_te[c+2]=hl
    adv_te[c+3]=hh
    c=c+4
    
image_size = adv_te.shape[1]
original_dim = image_size * image_size
adv_te = np.reshape(adv_te, [-1, original_dim])




encoded_x=encoder.predict(adv_te)
encoded_x[1].shape
len(encoded_x)
len(encoded_x[0])
decoded_x=decoder.predict(encoded_x[1])
decoded_x.shape

y=np.reshape(decoded_x,[40000,16,16]) 


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
predicted_classes = model123.predict_classes(x_test)

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

###############################################################################
## Latent space Accuracy
train_encoded=encoder.predict(x_tr)
train_encoded[0].shape
len(train_encoded[0])
test_encoded=encoder.predict(x_te)

    

test=np.zeros(shape=(10000,4,2))
c=0
for i in range(10000):
    ll=test_encoded[0][c]
    lh=test_encoded[0][c+1]
    hl=test_encoded[0][c+2]
    hh=test_encoded[0][c+3]
    test[i]=np.vstack([ll,lh,hl,hh])
    c=c+4

test.shape
test=test.reshape(len(test),8)

train=np.zeros(shape=(60000,4,2))
c=0
for i in range(60000):
    ll=train_encoded[0][c]
    lh=train_encoded[0][c+1]
    hl=train_encoded[0][c+2]
    hh=train_encoded[0][c+3]
    train[i]=np.vstack([ll,lh,hl,hh])
    c=c+4

test.shape
test=test.reshape(len(test),8)
train=train.reshape(len(train),8)



model = Sequential()
model.add(Dense(30, input_shape=(8,)))
model.add(Activation('sigmoid'))                        

model.add(Dense(30))
model.add(Activation('sigmoid'))

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

history = model.fit(train, Y_train,epochs=30)

predicted_classes = model.predict_classes(test)
predicted_classes[0]
new=np.nonzero(Y_test)
results=pd.DataFrame()
results['index']=new[0]
results['Actual']=new[1]
results['Predicted']=predicted_classes
actual=results['Actual']
predicted=results['Predicted']
 
results = confusion_matrix(actual, predicted) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(actual, predicted)) 
print ('Report : ')
print (classification_report(actual, predicted))