# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:54:44 2019

@author: Suchit
"""

# Importing the Keras libraries and packages
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(96, (6, 6), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(256, (4, 4), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(384, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 512, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))

classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('dataset',
                                                 color_mode="rgb",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = _, #steps_per_epoch = (total training images)/batch_size
                         epochs = 20)
#Accuracy (training) : 0.9643

#----------------------------------------------------------------


# Test the model
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('test/',
                                            color_mode="rgb",
                                            target_size = (64, 64),
                                            batch_size = , #enter number of test images
                                            shuffle = False,
                                            class_mode = None)

proba = classifier.predict_generator(test_set, steps=1)

from sklearn.metrics import confusion_matrix
y_true = np.array([0] * _ + [1] * _) #enter number of female and male images respectively
y_pred = proba > 0.5
confusion_matrix(y_true, y_pred)
#----------------------------------------------------------------


# Test on individual image
import cv2
from keras.preprocessing.image import ImageDataGenerator
from  numpy import expand_dims

n = cv2.imread('') #enter the image path
n = n[:, :, ::-1]
n = cv2.resize(n, (64, 64)) 

x = expand_dims(n, 0)
datagen = ImageDataGenerator(rescale = 1./255)
it = datagen.flow(x, batch_size=1)
b = it.next()

#proba = classifier.predict_generator(it, steps=1)
print(classifier.predict(b))
print(classifier.predict_classes(b))
