# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
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

i = 0
image_name = name_image_mask_truemask[i, 0]
image_path = name_image_mask_truemask[i, 1]
mask_path = name_image_mask_truemask[i, 2]
true_mask_path = name_image_mask_truemask[i, 3]

img = cv2.imread(image_path).astype(np.int32)

#######################Debug CalcBeta#################################
beta = 0
test_list = list()
for y in range(img.shape[0]):
    for x in range(img.shape[1]):

        color = img[y, x, :]

        if x > 0:  # left
            diff = color - img[y, x - 1]
            beta += diff.dot(diff)
            test_list.append([y, x, diff.dot(diff)])

        if y > 0 and x > 0:  # upleft
            diff = color - img[y - 1, x - 1]
            beta += diff.dot(diff)
            test_list.append([y, x, diff.dot(diff)])

        if y > 0:  # up
            diff = color - img[y - 1, x]
            beta += diff.dot(diff)
            test_list.append([y, x, diff.dot(diff)])

        if y > 0 and (x < img.shape[1] - 1):  # upright
            diff = color - img[y - 1, x + 1]
            beta += diff.dot(diff)
            test_list.append([y, x, diff.dot(diff)])

if beta <= 1e-16:
    beta = 0
else:
    beta = 1.0 / (
        2 * beta / (4 * img.shape[1] * img.shape[0] - 3 * img.shape[1] -
                    3 * img.shape[0] + 2))

with open("/Users/spacegoing/macCodeLab-MBP2015/"
          "Python/MRFLSVM/PyMRFLSSVM/beta.txt", 'r') as f:
    c_beta = [[int(j) for j in i.strip('\n').split(' ')] for i in f.readlines()]

t=np.asarray(test_list)
c = np.asarray(c_beta)
np.sum(t==c)