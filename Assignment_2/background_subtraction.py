import cv2
import numpy as np
import os
import imutils
import pickle
import SVM_training as st


def from_frames_to_video(directory, name):
    '''This function creates a video from frames.
    It takes the directory of the frames and a name as inputs.
    It returns the full name of the video (name+ .mp4)
    It created and saved the video in the current directory.'''
    
    img_array = [] # List with all the frames
    for filename in os.listdir(directory): # Iteration over all frames
        img = cv2.imread(directory+'/'+filename) # reading frame
        height, width, layers = img.shape 
        size = (width,height)
        img_array.append(img) # constituating the list
 
    out = cv2.VideoWriter(name+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size) # initializing the video
 
    for i in range(len(img_array)): # adding frames one by one
        out.write(img_array[i])
    out.release() # releasing
    
    return name+'.mp4'


def back_sub_preprocess(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    return frame_gray


def detect_bounding_boxes(clf, video, bbs_file_name, reference_frame=0):
    '''This function detects the person on a video.
    First it uses a background subtraction procedure and aon a second time it 
    uses a classifier to be sure that the moving object is a person.
    
    It takes the classifier and the video as input plus a frame that will be
    the frame with no person on it and that is supposed to be the beginning point 
    of the background subtraction.
    It returns a list with all the bounding boxes. More precisely it returns the name
    of the text file with the bbs.'''
    
    with open(bbs_file_name+'.txt', 'w') as file:
        file.write('')
        
    bb_list = []
    
    cap = cv2.VideoCapture(video) # Reading the video
    
    if reference_frame == 0: # we take the first frame as the frame with no one on screen
        _, frame = cap.read()
    else:
        first_frame = cv2.imread(reference_frame) # Otherwise, this is supposed to be the frame with no one on it
         
    first_gray = back_sub_preprocess(first_frame) # preprocess first frame
    
    frame_id = 1
     
    video_not_finished = True # variable to stop when the video ends
    while video_not_finished:
        
        if reference_frame != 0 or frame_id != 1: # If the initialization has been made with the first frame then no need to read it again
            _, frame = cap.read() # reading one frame of the video
        video_not_finished = _
        
        if video_not_finished:
            
            gray_frame = back_sub_preprocess(frame) # preprocess current frame
         
            difference = cv2.absdiff(first_gray, gray_frame) # Difference between current frame and reference frame
            _, difference = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY) # applying threshold
            
            cnts = cv2.findContours(difference.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts) # finding contours
                
            det = []
            for contour in cnts:# looping over the contours
                if cv2.contourArea(contour) < 100: # if the contour is too small, ignore it
                    continue
         
        		# compute the bounding box for the contour
                (x, y, dx, dy) = cv2.boundingRect(contour)
                det.append([x, y, dx, dy])
            
            det_2 = []
            added = []
            
            for i in range(len(det)): 
                '''This module tries to take car of bounding boxes bad determined
                Sometimes two bbs are close to each other and should only be one.
                We are fiwing this here.'''
                modified = False
                for j  in range(i, len(det)):
                    [x1, y1, w1, h1] = det[i]
                    [x2, y2, w2, h2] = det[j]
                    
                    if x1-10 < x2 < x1+10 and w1-10 < w2 < w1+10:
                        if y1+h1-10 < y2 < y1+h1+10 or y2+h2-10 < y1 < y2+h2+10:
                            # If we are here it means that the two bbs have similar sizes and are really clos to each other
                            x3 = min(x1, x2)
                            w3 = max(w1, w2)
                            y3 = min(y1, y2)
                            h3 = h1 + h2 + y3 - min(y1, y2)
                            det_2.append([x3, y3, w3, h3])
                            modified = True
                            added.append(j)
                
                if not modified and j not in added:
                    det_2.append(det[i])
                    
            bb_id = 1
                
            for [x, y, w, h] in det_2:
                
                if h > 1 * w and h < 3.5 * w:
                    # We consider bbs with only a shape corresponding to a walking person
                    subim = frame[y:y+h, x:x+w]
                    descriptor = st.preprocess_image(subim)
                    probability = clf.predict_proba([descriptor.flatten()]) # Get the probability that a person is in the subimage
                    
                    if probability[0][1] > 0.5: # If the probability is high enough        
                   
                        with open(bbs_file_name+'.txt', 'a') as file:
                            file.write(str(frame_id)+', '+str(bb_id)+', '+str(x)+', '+str(y)+', '+str(w)+', '+str(h)+'\n')
                        
                        bb_list.append([frame_id, bb_id, x, y, dx, dy])
                        bb_id += 1
                   
            frame_id += 1
            
     
    cap.release()
    cv2.destroyAllWindows() # Ending video
    return bb_list

def run(directory):
    '''Main funtion taking a directory and returning a list with all the
    bounding boxes in the right format'''
    
    full_video_name = from_frames_to_video(directory, 'project') # Create video
    
    with open('svm_trained', 'rb') as file: # Load trained SVM
        my_pickler = pickle.Unpickler(file)
        clf = my_pickler.load()
        
    bbs_list = detect_bounding_boxes(clf, full_video_name, 'bb_file', 'img1/684.jpg')   # Calling the function returning bbs
    return bbs_list

if __name__ == '__main__':
    pass
    