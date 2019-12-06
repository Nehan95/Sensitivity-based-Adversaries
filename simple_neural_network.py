# -*- coding: utf-8 -*-
"""
Simple Neural Network for classification of Mnist Images
"""
# Importing libraries
import os
os.getcwd()
os.chdir('C:\Python\pro')
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from keras.optimizers import Adam
from keras.models import load_model

#Parameter tuning
batch_size = 128
num_classes = 10
epochs =20
learning_rate=0.001

#Loading the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshaping the data
image_vector_size = 28*28
X_train = x_train.reshape(x_train.shape[0], image_vector_size).astype('float32')
X_test = x_test.reshape(x_test.shape[0], image_vector_size).astype('float32')
X_train = X_train / 255
X_test = X_test / 255

#plotting the images of training set
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(x_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig

#One-hot encoding for labels
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
X_train.shape

#Initializing the neural network
model = Sequential()

#Input and 1st hidden layer
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))                        

#2nd hidden layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#output layer
model.add(Dense(n_classes))
model.add(Activation('softmax'))

#compiling the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=learning_rate))

#printing the summary of the neural network
model.summary()

#Training the neural network
history = model.fit(X_train, Y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test, Y_test))

#Model Evaluation
#Test
score = model.evaluate(X_test, Y_test, verbose=0) 
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#train
score = model.evaluate(X_train, Y_train, verbose=0) 
print('Train loss:', score[0])
print('Train accuracy:', score[1])

#Visualizing the loss in training and test
# List all data in history
print(history.history.keys())
# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predicting classes for test images
predicted_classes = model.predict_classes(X_test)

#Accuracy, Confusion matrix and Classification report
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


#saving the model
model.save('simple_neural_network.h5') 

#load the model we saved
new_model=load_model('simple_neural_network.h5')

#Evaluating the loaded  model

#predicting classes for test images
predicted_classes = new_model.predict_classes(X_test)

#Accuracy of loaded model
print ('Accuracy Score :',accuracy_score(actual, predicted_classes))
