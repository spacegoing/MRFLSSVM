# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import pickle
from Checkboard import Instance, Options
from GrabCut.GrabCutInstance import GrabCutInstance, get_name_path_arr
import os

image_dir = './GrabCut/Data/grabCut/images/'

mask_dir = './GrabCut/Data/grabCut/labels/'
mask_ground_truth_type = ''
dump_dir = './GrabCut/Data/grabCut/UnaryPairwiseCliques/'

mask_input_type_list = ['_new', '_rect', '_lasso']
mask_input_type = '_new'
name_image_mask_truemask = get_name_path_arr(image_dir, mask_dir,
                                             mask_ground_truth_type, mask_input_type)
i=31
image_name = name_image_mask_truemask[i, 0]
image_path = name_image_mask_truemask[i, 1]
mask_path = name_image_mask_truemask[i, 2]
true_mask_path = name_image_mask_truemask[i, 3]

# slic
grabInstance = GrabCutInstance(image_path, mask_path, true_mask_path,
                               method='slic', numSegments=300)

# quickshift
grabInstance1 = GrabCutInstance(image_path, mask_path, true_mask_path,
                               method='quickshift')

# felzenszwalb
grabInstance2 = GrabCutInstance(image_path, mask_path, true_mask_path,
                               method='quickshift')
