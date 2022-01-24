# -*- coding: utf-8 -*-
"""
Created on Sun Jan 2 2022

@author: Morais

"""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,'/mrcnn/')
import mrcnn, mrcnn.config, mrcnn.model, mrcnn.visualize, cv2

# Loading procedure for Mask RCNN
WEIGHTS_PATH = "./final models/mask_rcnn_object_0057.h5"  

class CustomConfig(mrcnn.config.Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3  
    STEPS_PER_EPOCH = 15
    VALIDATION_STEPS= 1
    BATCH_SIZE=1
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.76
    
#config = CustomConfig()

# config.display()
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

model = mrcnn.model.MaskRCNN(mode="inference", model_dir='./', 
                             config=CustomConfig())
weights_path = WEIGHTS_PATH

print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# seperate_thresholds = [0.0,0.72,0.85,0.6]
# The class ids are:- short wall:1 stairs:2 railings:3 (BG is BackGround)
CLASS_NAMES = ['BG','Short wall','Stairs','Railings']

def evaluate_streetview_img(file, output_img_path=None):
    """
    ........
    """
    img = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
    
    start_time = time.time() 
    # add verbose=1 to display some tensor and image related stats 
    results1 = model.detect([img])
    print(f"--- Prediction time in seconds {time.time() - start_time:2f} ---")
    ax = get_ax(1)
    r1 = results1[0]  
    
    # Count of each class
    sw = np.sum(r1['class_ids'] == 1)
    stairs = np.sum(r1['class_ids'] == 2)
    rail = np.sum(r1['class_ids'] == 3)
    
    pred_str = (f"Short walls = {sw}; Railings = {rail}; Stairs = {stairs}")
    print(pred_str)
    mrcnn.visualize.display_instances(img, r1['rois'], r1['masks'], 
                                      r1['class_ids'], CLASS_NAMES, 
                                      r1['scores'], ax=ax, title=pred_str,
                                      output_path=output_img_path)
    return (sw, stairs, rail)

#%% Test if it's working:
    
import glob

img_folder = "./imgs to convince PT/"

for file in glob.glob(img_folder + '*original.jpg'):
    print(file)
    per_class_hits = evaluate_streetview_img(file, file.replace('original', 'pred'))
    n_class_hits = sum(per_class_hits)
    