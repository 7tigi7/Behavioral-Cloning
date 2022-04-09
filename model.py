import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import flatten
import sklearn
from sklearn.utils import shuffle
from skimage import io
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Input, Activation, MaxPooling2D,Conv2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import scipy.misc
tf.set_random_seed(1234)

#Read data
data = pd.read_csv('./data/driving_log.csv')
print("Log readed")
print("Size of data is",len(data))

# Arrays to collect the images and angles to validate
images = []
angles = []

for i, row in data.iterrows():
    filename = 'data/' + row['center'].strip()
    if os.path.isfile(filename):
        image = io.imread('data/' + row['center'].strip())
        angle = row['steering']
        images.append(image)
        images.append(image)
        images.append(image)
        angles.append(angle)
        angles.append(angle)
        angles.append(angle)
        image = io.imread('data/' + row['left'].strip())
        images.append(image)
        angles.append(angle+0.2345)
        images.append(image)
        angles.append(angle+0.3)
        images.append(image)
        angles.append(angle+0.26)
        image = io.imread('data/' + row['right'].strip())
        images.append(image)
        angles.append(max(-0.92,angle-0.2345))
        images.append(image)
        angles.append(max(-0.92,angle-0.3))
        images.append(image)
        angles.append(max(-0.92,angle-0.26))
        
sklearn.utils.shuffle(images, angles)
#np.save('images',  np.array(images))
#np.save('angles', np.array(angles))
print("Data readed")
images = np.array(images)
angles = np.array(angles)
print(len(images))

# Define model(NVIDIA model) https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
model = Sequential()
# Cropping images. The irrelevant information is cropped from the image : trees lakes etc..
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Normalization
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(90, 320, 3)))
# 5 Convolutional feature map
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dropout(0.4))
# 3 Fully connected layer
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

# Compile model
learningRate = 1e-4
model.compile(optimizer=Adam(learningRate), loss="mse", )

# Train model
model.fit(images, angles, epochs=30, validation_split=0.1)

# Save model
model.save('model.h5')
print("Model saved")
