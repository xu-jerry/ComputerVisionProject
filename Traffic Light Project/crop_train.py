import numpy as np
import cv2
from read_label_file import get_all_labels

images = get_all_labels('/home/felix/Downloads/Computer_Vision_Project/train.yaml')

for image in images:
    image_path = image['path']
    image_id = image_path.split('/')[-2] + '/' + image_path.split('/')[-1]
    image_id2 = image_path.split('/')[-1]
    img = cv2.imread('/home/felix/Downloads/Computer_Vision_Project/rgb/train/' + image_id,1)
    
    for box in image['boxes']:
        y1 = int(box['y_min'])
        x1 = int(box['x_min'])
        y2 = int(box['y_max'])
        x2 = int(box['x_max'])
        
        if (y2 - y1 > 10) and (x2 - x1 > 5):
            if not box['occluded']:
                if box['label'] == 'Green':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Train/Green/' + image_id2, crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Train/Green/Reference/' + image_id2, img)
                elif box['label'] == 'Red':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Train/Red/' + image_id2, crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Train/Red/Reference/' + image_id2, img)
                elif box['label'] == 'Yellow':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Train/Yellow/' + image_id2, crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Train/Yellow/Reference/' + image_id2, img)

cv2.destroyAllWindows()
