# import the necessary packages
from __future__ import print_function
import os

import SVM_training as st
import background_subtraction as bs

def pedestrians(data_root, _W, _H, _N):
    ''' Return a list of bounding boxes in the format frame, bb_id, x,y,dx,dy '''
    
    if 'svm_trained' not in os.listdir(): # If the svm has not been trained, do it
        print('not')
        st.run()
        
    bbs_list = bs.run(data_root)   # calling the background_subtraction funtion with dataroot
    return bbs_list
