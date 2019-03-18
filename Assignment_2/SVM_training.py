import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle

'''The role of this file is to train a SVM. The SVM classifies an image (seen 
as matrix into two classes:
    -a person
    -not a person
    
Basically the SVM takes an image and says whether its a person or not
We use the given data in order to train this SVM'''


def read_frame(path):
    '''Returns the image as a np array'''
    return cv2.imread(path)

def read_gt(filename):
    """Read gt and create list of bb-s"""
    assert os.path.exists(filename) # Check whether the file exists or not
    with open(filename, 'r') as file:
        lines = file.readlines()
    # truncate data (last columns are not needed)
    return [list(map(lambda x: int(x), line.split(',')[:6])) for line in lines]

def annotations_for_frame(solution, frame):
    return [bb for bb in solution if int(bb[0])==int(frame)]

def preprocess_image(image):
    '''Takes an image as input and returns the image preprocessed for the svm:
        - Resized
        - With Hog Descriptor corresponding to the image'''
    resized = cv2.resize(image, (64, 128), interpolation = cv2.INTER_AREA) # Resize it
    # plt.imshow(a)
    # plt.show()
    hog = cv2.HOGDescriptor()
    desc = hog.compute(resized) # Compute the Hog Descriptor corresponding to the subimage
    return desc

def create_persons_list(images_directory, gt_directory):
    '''This functions takes as input the directory of images and the corresponding
    bounding boxes that will be used to train the svm
    It returns a list of hog descriptor of images representing a person.'''
    
    list_persons = [] # The list containing the person images
    
    # gt_path = './gt/gt.txt'
    gt_path = gt_directory
    gt = read_gt(gt_path) # Read the bounding boxes
    
    for image_nb in range(1, len(os.listdir(images_directory))+1): # Let's loop over the images
        if image_nb < 10:
            image = read_frame(images_directory+'/00%s.jpg' % image_nb)
        elif image_nb < 100:
            image = read_frame(images_directory+'/0%s.jpg' % image_nb)
        else:
            image = read_frame(images_directory+'/%s.jpg' % image_nb)
            
        print(image_nb) # image_nb corresponds to the number of the frame we will work on
        # plt.imshow(image)
        # plt.show()
        bbs = annotations_for_frame(gt, image_nb) # Get the boundings boxes (bbs) corresponding to this frame
        
        for bb in bbs:
            x, y = bb[2:4]
            dx, dy = bb[4:6] # Coordinates of the bounding box
        
            subimage = image[y:y+dy, x:x+dx] # Get the sub image corresponding to the bb
            desc = preprocess_image(subimage)
            # list_persons.append(list(itertools.chain.from_iterable(h.flatten().tolist())))
            list_persons.append(desc.flatten().tolist())
            
    return list_persons # Returns the full list

def inter_over_union(bb1, bb2):
    '''Returns the intersection over union of two subimages of on image
    using their coordinates'''
    
    [x1, dx1, y1, dy1] = bb1
    [x2, dx2, y2, dy2] = bb2
    
    inter = (- x1 - dx1 + x2 + 2 * dx2) * (- y1 - dy1 + y2 + 2 * dy2)
    # inter = abs(inter)
    
    union = dx1 * dy1 + dx2 * dy2 - inter
    
    return inter / union

def create_empty_list(images_path, image_nb, gt_directory):
    '''This functions will return the negative sample for the SVM training:
        It takes an image number as input and returns a list with all the subimages 
        not containing a person (hog descriptor of the subimage of course)'''
    
    list_empty = []
    
    # gt_path = './gt/gt.txt'
    gt_path = gt_directory
    gt = read_gt(gt_path)
    
    # Sizes studied
    sizes = [16, 32, 64, 128, 256] # Sizes of the subimages (dx) that will be created
        
    image = read_frame(images_path+'/'+str(image_nb)+'.jpg')   
    # plt.imshow(image)
    # plt.show()
    bbs = annotations_for_frame(gt, image_nb)
    
    for dx in sizes: # Let's loop over the different sizes
        dy = 2 * dx # We take subimages twice as high as they are wide
        
        x = 0 # Initializing the position of the subimage
        y = 0
        
        # Basically we slide one window over the entire image and take subimage each time
        
        while x+dx < len(image[0]): # Make sure we stay within the image
            
             while y+dy < len(image): # Same
                
                no_person = True # Is there a person in our subimage? initializing as no
        
                for bb in bbs: # We checks the bbs of this image to be sure no one is on our subimage
                    x_pers, y_pers = bb[2:4]
                    dx_pers, dy_pers = bb[4:6]
                    '''
                    if x_pers < x and x < x_pers + dx_pers:
                        if y_pers < y and y < y_pers + dy_pers:
                            no_person = False
                        if y_pers < y + dy and y + dy < y_pers + dy_pers:
                            no_person = False
                    if x_pers < x + dx and x + dx < x_pers + dx_pers:
                        if y_pers < y and y < y_pers + dy_pers:
                            no_person = False
                        if y_pers < y + dy and y + dy < y_pers + dy_pers:
                            no_person = False
                    if x -10 < x_pers and x_pers + dx_pers < x + dx +10:
                        if y -10< y_pers and y_pers + dy_pers < y + dy+10:
                            no_person = False
                    '''
                    
                    iou = inter_over_union([x, dx, y, dy], [x_pers, dx_pers, y_pers, dy_pers]) # Checking the iou
                    if iou > 0.2: # If the iou of the subimage and the bb is to high : there is a person on the subimage
                        no_person = False
                    
                if no_person: # If there is no one on the subimage: we preprocess it and will use it as a negative sample
                    subimage = image[y:y+dy, x:x+dx]
                    # plt.imshow(subimage)
                    # plt.show()
                    desc = preprocess_image(subimage)
                    # list_empty.append(list(itertools.chain.from_iterable(h.flatten().tolist())))
                    list_empty.append(desc.flatten().tolist())

                y = y + 4 # We slide the windows by 4 pixel vertically
            
             x = x + 4 # We slide the windows by 4 pixel horizontally
            
    return list_empty

def train_svm(positive_samples, negative_samples):
    '''This function trains a svm using the inputs given .
    The svm classifies the positive samples with 1 and the negative ones as 0.
    The function takes as inputs the pisitive and negative samples 
    and returns the trained classifier.'''
    
    X = positive_samples + negative_samples # training input data
    Y = [1]*len(positive_samples) + [0] * len(negative_samples) # training output data
    
    clf = svm.SVC(probability=True) # Initializing the svm
    clf.fit(X, Y) # training svm
    return clf

def run():
    '''Main function of the script should be called if we want to train and save
    a svm using the values given for the project that are used below.'''
    
    list_persons = create_persons_list('img1', 'gt/gt.txt') # Compute the positive samples
    with open('data_person', 'wb') as file: # Save them
        my_pickler = pickle.Pickler(file)
        my_pickler.dump(list_persons)
        
    list_empty = create_empty_list('img1', 684, 'gt/gt.txt') # Compute the negative samples
    with open('data_empty', 'wb') as file: # Save them
        my_pickler = pickle.Pickler(file)
        my_pickler.dump(list_empty)  
        
    print('Preprocessing Done')
    print(len(list_empty))
    print(len(list_persons))
    
    svm_clf = train_svm(list_persons, list_empty) # Train svm
    with open('svm_trained', 'wb') as file: # Save it
        my_pickler = pickle.Pickler(file)
        my_pickler.dump(svm_clf)



         