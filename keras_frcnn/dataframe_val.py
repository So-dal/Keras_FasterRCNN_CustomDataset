#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:48:52 2021

@author: sophie
"""

import pandas as pd
import os.path
import os


def get_dataframe (dataset):
    
    """ to create the dataframe required for the datagen.flow_from_dataframe method, 
        dataset must be the name of the folder 'train_images' , 'val_images' or 'test_images' in string format """
    
    path=[]

    ROOT_DIR = './'
    IMGS_PATH = os.path.join(ROOT_DIR, dataset+"/")
    path=[f for f in os.listdir(IMGS_PATH) if f.endswith('.png')]
    path.append(path)
    del path[-1]
    df=pd.DataFrame(path,columns=['image_path'])
    df.insert(1, "class", "handwritten")
    return df
    
    
def filenames_per_batch (gen):
    
    """ arg = name of the data generator (datagen.flow_from_dataframe) """
    
    img_paths_per_batch=[]
    batches_per_epoch = gen.samples // gen.batch_size + (gen.samples % gen.batch_size > 0)

    for i in range(batches_per_epoch):
        batch = next(gen)
        current_index = ((gen.batch_index-1) * gen.batch_size)
        if current_index < 0:
            if gen.samples % gen.batch_size > 0:
                current_index = max(0, gen.samples - gen.samples % gen.batch_size)
            else:
                current_index = max(0,gen.samples - gen.batch_size)
                
        index_array = gen.index_array[current_index:current_index + gen.batch_size].tolist()
        img_paths = [gen.filepaths[idx] for idx in index_array]
        img_paths_per_batch.append(img_paths)
        
    return img_paths_per_batch