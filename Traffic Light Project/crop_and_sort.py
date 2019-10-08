#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:12:36 2019

@author: felix
"""

import numpy as np
import cv2
from read_label_file import get_all_labels

images = get_all_labels('/home/felix/Downloads/Computer Vision Project/test.yaml')
i = 24068

for image in images:
    
    img = cv2.imread('/home/felix/Downloads/Computer Vision Project/rgb/test/'+ str(i) + '.png',1)
    
    for box in image['boxes']:
        y1 = int(box['y_min'])
        x1 = int(box['x_min'])
        y2 = int(box['y_max'])
        x2 = int(box['x_max'])
        
        if (y2 - y1 > 10) and (x2 - x1 > 5):
            if not box['occluded']:
                if box['label'] == 'Green':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Green/' + str(i) + '.jpg', crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Green/Reference/' + str(i) + 'annotated.jpg', img )
                elif box['label'] == 'Red':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Red/' + str(i) + '.jpg', crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Red/Reference/' + str(i) + 'annotated.jpg', img )
                elif box['label'] == 'Yellow':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Yellow/' + str(i) + '.jpg', crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Yellow/Reference/' + str(i) + 'annotated.jpg', img )
    i += 2

cv2.destroyAllWindows()
