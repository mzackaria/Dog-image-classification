# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 16:26:59 2018

@author: zakis
"""
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras import models

import numpy as np
import os

side_length = 128
 
#importing the model
model_path = r'C:\Users\zakis\Documents\OpenClassroom\Projet 7\final_model'
m = models.load_model(model_path)

#test on one image
#get the image path
img_path = r'C:\Users\zakis\Documents\OpenClassroom\Projet 7\Images\n02087046-toy_terrier\n02087046_357.jpg'

img = load_img(img_path, target_size=(side_length, side_length))  # Charger l'image
img = img_to_array(img)  # Convertir en tableau numpy
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

#predict in percent the probability of every image fo that particular dog
pred = m.predict(img)

directory = r'C:\Users\zakis\Documents\OpenClassroom\Projet 7\Images'
folder_name = os.listdir(directory)[np.argmax(pred.T)]
ind = folder_name.find('-')
race_name = folder_name[ind+1:]
print('The race of that dog is '+race_name)

