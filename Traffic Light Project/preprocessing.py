import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
from read_label_file import get_all_labels
import os
import yaml

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)

import constants

""" 
    Gets all labels within label file
    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    Args:
        input_yaml->str: Path to yaml file
        riib->bool: If True, change path to labeled pictures
        clip->bool: If True, clips boxes so they do not go out of image bounds
    Returns: Labels for traffic lights
"""

def get_all_labels(input_yaml, riib=False, clip=True):
    
    assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
    with open(input_yaml, 'rb') as iy_handle:
        images = yaml.load(iy_handle)

    if not images or not isinstance(images[0], dict) or 'path' not in images[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml),
                                                         images[i]['path']))

        # There is (at least) one annotation where xmin > xmax
        for j, box in enumerate(images[i]['boxes']):
            if box['x_min'] > box['x_max']:
                images[i]['boxes'][j]['x_min'], images[i]['boxes'][j]['x_max'] = (
                    images[i]['boxes'][j]['x_max'], images[i]['boxes'][j]['x_min'])
            if box['y_min'] > box['y_max']:
                images[i]['boxes'][j]['y_min'], images[i]['boxes'][j]['y_max'] = (
                    images[i]['boxes'][j]['y_max'], images[i]['boxes'][j]['y_min'])

        # There is (at least) one annotation where xmax > 1279
        if clip:
            for j, box in enumerate(images[i]['boxes']):
                images[i]['boxes'][j]['x_min'] = max(min(box['x_min'], constants.WIDTH - 1), 0)
                images[i]['boxes'][j]['x_max'] = max(min(box['x_max'], constants.WIDTH - 1), 0)
                images[i]['boxes'][j]['y_min'] = max(min(box['y_min'], constants.HEIGHT - 1), 0)
                images[i]['boxes'][j]['y_max'] = max(min(box['y_max'], constants.HEIGHT - 1), 0)

        # The raw imager images have additional lines with image information
        # so the annotations need to be shifted. Since they are stored in a different
        # folder, the path also needs modifications.
        if riib:
            images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
            images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
            images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
            for box in images[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return images

'''
    Prints statistic data for the traffic light yaml files, including the distribution of height, width, size(area), and labels.
    :param input_yaml: Path to yaml file of published traffic light set
    Creates visual graphs for comparison.
'''

def quick_stats(input_yaml):

    images = get_all_labels(input_yaml)

    widths = []
    heights = []
    sizes = []

    num_images = len(images)
    num_lights = 0
    appearances = {'Green': 0, 'occluded': 0}

    for image in images:
        num_lights += len(image['boxes'])
        for box in image['boxes']:
            try:
                appearances[box['label']] += 1
            except KeyError:
                appearances[box['label']] = 1

            if box['occluded']:
                appearances['occluded'] += 1
                
            if box['x_max'] < box['x_min']:
                box['x_max'], box['x_min'] = box['x_min'], box['x_max']
            if box['y_max'] < box['y_min']:
                box['y_max'], box['y_min'] = box['y_min'], box['y_max']

            width = box['x_max'] - box['x_min']
            height = box['y_max'] - box['y_min']
            if width < 0:
                logging.warning('Box width smaller than one at ' + image)
            widths.append(width)
            heights.append(height)
            sizes.append(width * height)

    avg_width = sum(widths) / float(len(widths))
    avg_height = sum(heights) / float(len(heights))
    avg_size = sum(sizes) / float(len(sizes))

    median_width = sorted(widths)[len(widths) // 2]  
    median_height = sorted(heights)[len(heights) // 2] 
    median_size = sorted(sizes)[len(sizes) // 2]
    
    #statistics
    print('Number of images:', num_images)
    print('Number of traffic lights:', num_lights, '\n')

    print('Minimum width:', min(widths))
    print('Average width:', avg_width)
    print('median width:', median_width)
    print('maximum width:', max(widths), '\n')

    print('Minimum height:', min(heights))
    print('Average height:', avg_height)
    print('median height:', median_height)
    print('maximum height:', max(heights), '\n')

    print('Minimum size:', min(sizes))
    print('Average size:', avg_size)
    print('median size:', median_size)
    print('maximum size:', max(sizes), '\n')

    print('Labels:')
    numbers = []
    for key, label in appearances.items():
        print('\t{}: {}'.format(key, label))
        if key == 'Red':
            numbers.append(label)
        if key == 'Yellow':
            numbers.append(label)
        if key == 'Green':
            numbers.append(label)
        if key == 'occluded':
            numbers.append(label)
        if key == 'off':
            numbers.append(label)
    
    #box plots of width, height, and size
    ax1 = plt.subplots()
    ax1 = plt.boxplot(widths, showfliers=False)
    ax1 = plt.title('Train Data Widths')
    ax2 = plt.subplots()
    ax2 = plt.boxplot(heights, showfliers=False)
    ax2 = plt.title('Train Data Heights')
    ax3 = plt.subplots()
    ax3 = plt.boxplot(sizes, showfliers=False)
    ax3 = plt.title('Train Data Sizes')
    
    #histogram of number of images
    ax4 = plt.subplots()
    labels = ['Green', 'Occluded', 'Yellow', 'Red', 'Off']
    x = np.arange(0, len(labels), 1)
    plt.bar(x, numbers, align='center', alpha=0.5)
    ax4 = plt.xticks(x, labels)
    ax4 = plt.title('Train Data Distribution of Color')
        
quick_stats('/home/felix/Downloads/Computer_Vision_Project/train.yaml')

'''
    Crops each image to show only the traffic light while annotating the original image. Stores the annotations and cropped images in separate folders by label.
'''

images = get_all_labels('/home/felix/Downloads/Computer Vision Project/test.yaml')

for image in images:
    image_path = image['path']
    image_id = image_path.split('/')[-1]
    img = cv2.imread('/home/felix/Downlaods/Computer Vision Project/rgb/test/' + image_id,1)
    
    for box in image['boxes']:
        y1 = int(box['y_min'])
        x1 = int(box['x_min'])
        y2 = int(box['y_max'])
        x2 = int(box['x_max'])
        
        if (y2 - y1 > 10) and (x2 - x1 > 5):
            if not box['occluded']:
                if box['label'] == 'Green':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Green/' + image_id, crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Green/Reference/' + image_id, img)
                elif box['label'] == 'Red':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Red/' + image_id, crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Red/Reference/' + image_id, img)
                elif box['label'] == 'Yellow':
                    crop_img = img[y1:y2, x1:x2]
                    cv2.imwrite('/home/felix/Desktop/Cropped/Yellow/' + image_id, crop_img)
                    
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
                    cv2.imwrite('/home/felix/Desktop/Cropped/Yellow/Reference/' + image_id, img)

cv2.destroyAllWindows()

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
