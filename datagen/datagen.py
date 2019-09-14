# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:35:11 2019

@author: Suchit
"""

import json
import cv2
import numpy as np
import urllib.request

missed = []

def url_to_image(url):
    try:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except:
        return None


with open("dataset.json") as datafile:
    d = json.load(datafile)
    

for i in range(119):
       
    img = url_to_image(d["data"][i]["content"])

    if np.all(img == None ):
        missed.append(i)
        continue
    
    X = d["data"][i]["annotation"][0]["imageWidth"]
    Y = d["data"][i]["annotation"][0]["imageHeight"]
    for j in range(len(d["data"][i]["annotation"])):
            
            lab = len(d["data"][i]["annotation"][j]["label"])
            if(lab==1):
                emo = d["data"][i]["annotation"][j]["label"][0]
                
            if(lab==2):
                emo = d["data"][i]["annotation"][j]["label"][0]
                age = d["data"][i]["annotation"][j]["label"][1]
            if(lab==3):
                emo = d["data"][i]["annotation"][j]["label"][0]
                age = d["data"][i]["annotation"][j]["label"][1]
                eth = d["data"][i]["annotation"][j]["label"][2]
            if(lab==4):
                emo = d["data"][i]["annotation"][j]["label"][0]
                age = d["data"][i]["annotation"][j]["label"][1]
                eth = d["data"][i]["annotation"][j]["label"][2]
                gen = d["data"][i]["annotation"][j]["label"][3]
            
            if(emo!="Not_Face"):
                    x1 = int(d["data"][i]["annotation"][j]["points"][0]["x"] * X)
                    y1 = int(d["data"][i]["annotation"][j]["points"][0]["y"] * Y)
                    
                    x2 = int(d["data"][i]["annotation"][j]["points"][1]["x"] * X)
                    y2 = int(d["data"][i]["annotation"][j]["points"][1]["y"] * Y)
                    
                    crop = np.zeros([y2-y1+1, x2-x1+1, 3], dtype=np.uint8)
                    
                    a=0
                    for y in range(y1, y2): 
                        b=0
                        for x in range(x1, x2):
                            crop[a, b] = img[y, x]
                            b+=1
                        a+=1
                    
#                    cv2.imshow('yeesss', crop)
#                    cv2.waitKey()
                    
                    if age=="Age_below20":
                        p = 'age/%s/%d_%d.jpg'%(age,i,j)
                        cv2.imwrite(p, crop)
                        
                    if age=="Age_20_30":
                        p = 'age/%s/%d_%d.jpg'%(age,i,j)
                        cv2.imwrite(p, crop)
                        
                    if age=="Age_30_40":
                        p = 'age/%s/%d_%d.jpg'%(age,i,j)
                        cv2.imwrite(p, crop)
                        
                    if age=="Age_40_50":
                        p = 'age/%s/%d_%d.jpg'%(age,i,j)
                        cv2.imwrite(p, crop)
                        
                    if age=="Age_above_50":   
                        p = 'age/%s/%d_%d.jpg'%(age,i,j)
                        cv2.imwrite(p, crop)
                        
                        
                    if emo=="Emotion_Happy":
                        p = 'emo/%s/%d_%d.jpg'%(emo,i,j)
                        cv2.imwrite(p, crop)
                        
                    if emo=="Emotion_Neutral":
                        p = 'emo/%s/%d_%d.jpg'%(emo,i,j)
                        cv2.imwrite(p, crop)
                        
                    if emo=="Emotion_Sad":
                        p = 'emo/%s/%d_%d.jpg'%(emo,i,j)
                        cv2.imwrite(p, crop)
                        
                        
                    if eth=="E_Arab":
                        p = 'eth/%s/%d_%d.jpg'%(eth,i,j)
                        cv2.imwrite(p, crop)
                        
                    if eth=="E_Asian":
                        p = 'eth/%s/%d_%d.jpg'%(eth,i,j)
                        cv2.imwrite(p, crop)
                        
                    if eth=="E_Black":
                        p = 'eth/%s/%d_%d.jpg'%(eth,i,j)
                        cv2.imwrite(p, crop)
                        
                    if eth=="E_Hispanic":
                        p = 'eth/%s/%d_%d.jpg'%(eth,i,j)
                        cv2.imwrite(p, crop)
                        
                    if eth=="E_Indian":
                        p = 'eth/%s/%d_%d.jpg'%(eth,i,j)
                        cv2.imwrite(p, crop)
                        
                    if eth=="E_White":
                        p = 'eth/%s/%d_%d.jpg'%(eth,i,j)
                        cv2.imwrite(p, crop)
                        
                        
                    if gen=="G_ Female":
                        p = 'gen/%s/%d_%d.jpg'%(gen,i,j)
                        cv2.imwrite(p, crop)
                        
                    if gen=="G_Male":
                        p = 'gen/%s/%d_%d.jpg'%(gen,i,j)
                        cv2.imwrite(p, crop)
                        
                    emo = None
                    age = None
                    eth = None
                    gen = None
