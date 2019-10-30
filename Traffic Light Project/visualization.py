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
