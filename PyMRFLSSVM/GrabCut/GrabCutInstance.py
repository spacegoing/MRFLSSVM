# -*- coding: utf-8 -*-
import numpy as np
from GrabCut.GMM_Unary import GrabCut
from GrabCut.Superpixel import superpixel
import os

__author__ = 'spacegoing'

image_dir = './GrabCut/Data/grabCut/images/'

mask_dir = './GrabCut/Data/grabCut/labels/'
mask_ground_truth_type = ''
mask_input_type = '_new'


def get_name_path_arr(image_dir, mask_dir,
                      mask_ground_truth_type, mask_input_type):
    image_files = os.listdir(image_dir)
    name_image_mask_truemask = np.ndarray([len(image_files), 4], dtype=np.object)
    # filename, image_path, mask_path, mask_ground_truth_path

    for i in range(len(image_files)):
        filename = image_files[i].split('.')[0]

        image_path = image_dir + image_files[i]

        mask_ground_truth_name = filename + mask_ground_truth_type
        mask_input_name = filename + mask_input_type

        mask_path = mask_dir + mask_input_name + '.bmp'
        mask_ground_truth_path = mask_dir + mask_ground_truth_name + '.bmp'

        name_image_mask_truemask[i, :] = np.asarray([filename, image_path,
                                                     mask_path, mask_ground_truth_path])

    return name_image_mask_truemask


name_image_mask_truemask = get_name_path_arr(image_dir, mask_dir,
                                             mask_ground_truth_type, mask_input_type)

image_name = name_image_mask_truemask[0, 0]
image_path = name_image_mask_truemask[0, 1]
mask_path = name_image_mask_truemask[0, 2]
true_mask_path = name_image_mask_truemask[0, 3]

img_grabcut = GrabCut(image_path, mask_path)

input_mask = img_grabcut.input_mask
output_mask = img_grabcut.output_mask
true_mask = img_grabcut.read_mask_img(true_mask_path)

unary_observed = img_grabcut.get_unary_observed()
cliques = superpixel(image_path)+1
