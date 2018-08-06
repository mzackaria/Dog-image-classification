# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:59:32 2018

@author: zakis
"""

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential, Model
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

directory = r'C:\Users\zakis\Documents\OpenClassroom\Projet 7\Images'
n_class = 10#♣len(os.listdir(directory))
side_length = 128

nb_photos = 1000
Y = []
X = []
for i in range(0, n_class):#len(os.listdir(directory))):
    print(str(i))
    folder = directory + '\\' + os.listdir(directory)[i]
    nb = min(nb_photos, len(os.listdir(folder)))
    for j in range(0, nb):
        file = os.listdir(folder)[j]
        img = folder + '\\' + file
        img = load_img(img, target_size=(side_length, side_length))  # Charger l'image
        img = img_to_array(img)  # Convertir en tableau numpy
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
        img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16
        y = np.zeros((n_class,1))
        y[i] = 1
        if len(X) == 0:
            X = img
        else:
            X = np.concatenate([X, img])
        Y.append(y)

X = np.array(X)
Y = np.array(Y).reshape((len(Y),n_class))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

vgg16_base = VGG16(include_top=False, 
                   weights='imagenet', 
                   input_shape=(side_length, side_length, 3))

model = Sequential()
model.add(vgg16_base)
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

vgg16_base.trainable = True
set_trainable = False
for layer in vgg16_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable
    
model.compile(optimizer=optimizers.Adam(lr=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['acc'])

callbacks = [
    EarlyStopping(patience=5),
    ModelCheckpoint('vgg16_simple.h5', save_best_only=True),
]

history = model.fit(x=X_train, y=y_train,  
                    validation_data=(X_test, y_test), 
                    batch_size=32, epochs=100, callbacks=callbacks)

