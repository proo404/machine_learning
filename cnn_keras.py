# -*- coding: utf-8 -*-
"""labExp_Keras.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1auhpflyt9UuG9uVSOJMIFQ6_BQz1OIvu
"""

#load packages and modules
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt

#load data
(X_train,Y_train),(X_valid,Y_valid) = mnist.load_data()

#preprocessing the data
X_train =X_train .reshape(60000,784).astype('float32')
X_valid =X_valid .reshape(10000,784).astype('float32')

#normalization
X_train /=255
X_valid /=255

X_valid[0]

#convert the labels to one hot representation.
from keras import utils as np_utils
n_classes=10
Y_train=keras.utils.np_utils.to_categorical(Y_train,n_classes)
Y_valid=keras.utils.np_utils.to_categorical(Y_valid,n_classes)

Y_valid[0]

#Defining the model
model=Sequential()

#Adding dense layer
model.add(Dense(64,activation='sigmoid',input_shape=(784,)))

#Adding the final layer
model.add(Dense(10,activation='softmax'))

model.summary()

#compile the network
model.compile(loss='mean_squared_error',optimizer=SGD(learning_rate=0.01),metrics=['accuracy'])

#train
history=model.fit(X_train,Y_train,batch_size=128,epochs=150,verbose=1)