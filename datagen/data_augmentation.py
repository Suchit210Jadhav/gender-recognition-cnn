# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 23:47:33 2019

@author: Suchit
"""

import glob
import cv2

import numpy as np
from  numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

imgs = []
imgn = []
for img in glob.glob('*'):
    imgn.append(img)
    n = cv2.imread(img)
    imgs.append(n)
    
    
for y in range(len(imgs)):
    n = imgs[y]
    x = img_to_array(n)
    samples= expand_dims(x, 0)
    
    #Horizontal and Vertical Flip, Rotation, Zoom
    datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, zoom_range = [0.8, 1.4], rotation_range = 270)
    it = datagen.flow(samples, batch_size=1)
    k = 0
    while(k!=__):  #enter a value
        b = it.next()
        i = b[0].astype('uint8')
        if np.any(n!=i):
            name = imgn[y].split('.')[0]+"_"+str(k)+".jpg"
            cv2.imwrite(name, i)
        k+=1