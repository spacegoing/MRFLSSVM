# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
from GrabCut.GMM_Unary import GrabCut, GMM
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

t = np.asarray(test_list)
c = np.asarray(c_beta)
np.sum(t == c)

#################################Debug Pairwise####################################
gamma = 50
gammadivsqrt2 = gamma / np.sqrt(2.0)
leftw = np.zeros(img.shape[:2], dtype=np.double)
upleftw = np.zeros(img.shape[:2], dtype=np.double)
upw = np.zeros(img.shape[:2], dtype=np.double)
uprightw = np.zeros(img.shape[:2], dtype=np.double)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):

        color = img[y, x, :]

        if x > 0:  # left
            diff = color - img[y, x - 1]
            leftw[y, x] = gamma * np.exp(-beta * diff.dot(diff))
        else:
            leftw[y, x] = 0.0

        if y > 0 and x > 0:  # upleft
            diff = color - img[y - 1, x - 1]
            upleftw[y, x] = gammadivsqrt2 * np.exp(-beta * diff.dot(diff))
        else:
            upleftw[y, x] = 0.0

        if y > 0:  # up
            diff = color - img[y - 1, x]
            upw[y, x] = gamma * np.exp(-beta * diff.dot(diff))
        else:
            upw[y, x] = 0.0

        if y > 0 and (x < img.shape[1] - 1):  # upright
            diff = color - img[y - 1, x + 1]
            uprightw[y, x] = gammadivsqrt2 * np.exp(-beta * diff.dot(diff))
        else:
            uprightw[y, x] = 0.0

#################################Debug Unary####################################
root_dir = "/Users/spacegoing/macCodeLab-MBP2015/Python/MRFLSVM/PyMRFLSSVM/"
bgd_model_name = "bgd_model.txt"
bgd_det_name = "bgd_convDeterms.txt"
bgd_invCov_name = "bgd_inverseConv.txt"

fgd_model_name = "fgd_model.txt"
fgd_det_name = "fgd_convDeterms.txt"
fgd_invCov_name = "fgd_inverseConv.txt"

with open(root_dir + fgd_model_name, "r") as f:
    fgd_model = np.asarray([float(i.strip('\n')) for i in f.readlines()])
    fgd_model = fgd_model[np.newaxis, :]

with open(root_dir + fgd_det_name, "r") as f:
    fgd_det = np.asarray([float(i.strip('\n')) for i in f.readlines()])

with open(root_dir + fgd_invCov_name, "r") as f:
    fgd_invCov_list = np.asarray([[float(j) for j in i.strip('\n').split(' ')] for i in
                                  f.readlines()])
fgd_invCov = np.zeros([5, 3, 3])
for i in fgd_invCov_list:
    fgd_invCov[int(i[0]), int(i[1]), int(i[2])] = i[3]

fgdGMM = GMM(fgd_model)
err = 0
for i in range(5):
    for j in range(3):
        for k in range(3):
            err += np.abs(fgdGMM.inverseCovs[i, j, k] - fgd_invCov[i, j, k])

print(err)
# print(fgdGMM.inverseCovs == fgd_invCov)  # [0,0,0]
print(fgdGMM.covDeterms == fgd_det)
print(fgd_model[0, 20:] == fgdGMM.covs.flatten())

with open(root_dir + bgd_model_name, "r") as f:
    bgd_model = np.asarray([float(i.strip('\n')) for i in f.readlines()])
    bgd_model = bgd_model[np.newaxis, :]

with open(root_dir + bgd_det_name, "r") as f:
    bgd_det = np.asarray([float(i.strip('\n')) for i in f.readlines()])

with open(root_dir + bgd_invCov_name, "r") as f:
    bgd_invCov_list = np.asarray([[float(j) for j in i.strip('\n').split(' ')] for i in
                                  f.readlines()])
bgd_invCov = np.zeros([5, 3, 3])
for i in bgd_invCov_list:
    bgd_invCov[int(i[0]), int(i[1]), int(i[2])] = i[3]

bgdGMM = GMM(bgd_model)
err = 0
for i in range(5):
    for j in range(3):
        for k in range(3):
            err += np.abs(bgdGMM.inverseCovs[i, j, k] - bgd_invCov[i, j, k])

print(err)
# print(bgdGMM.inverseCovs == bgd_invCov)  # [0,0,0]
print(bgdGMM.covDeterms == bgd_det)
print(bgd_model[0, 20:] == bgdGMM.covs.flatten())

###################################unary sink source debug##############################
source_unary_name = 'source_unary.txt'
sink_unary_name = 'sink_unary.txt'
with open(root_dir + source_unary_name, "r") as f:
    source_unary_arr = np.asarray([[float(j) for j in i.strip(' \n').split(' ')] for i in
                                   f.readlines()])
with open(root_dir + sink_unary_name, "r") as f:
    sink_unary_arr = np.asarray([[float(j) for j in i.strip(' \n').split(' ')] for i in
                                 f.readlines()])

bgd_unary = bgdGMM.get_img_unary(img)
fgd_unary = fgdGMM.get_img_unary(img)

print(np.sum(np.abs(bgd_unary-source_unary_arr)))
print(np.sum(np.abs(fgd_unary-sink_unary_arr)))