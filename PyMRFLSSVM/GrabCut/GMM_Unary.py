# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import numpy as np
import cv2
from matplotlib import pyplot as plt

image_path = './GrabCut/Data/grabCut/images/'
mask_path = './GrabCut/Data/grabCut/labels/'
filename = '106024'
image_suffix = '.jpg'
mask_suffix = '.bmp'


class GMM:
    '''
    This class is a python version of part of (GMM) opencv grabcut.cpp
    (opencv/modules/imgproc/src/grabcut.cpp)

    Magic numbers in __init__ are:
    modelSize = 1 + 3 + 9  # component weight + mean + covariance
    '''

    def __init__(self, Model, Gaussian_Num=5):
        '''

        :param Model: cv2.grabcut()->fgdModel/bgdModel
        :param Gaussian_Num:
        '''
        # Those are found in opencv/modules/imgproc/src/grabcut.cpp
        # source code. Component means one guassian distribution
        self.componentsCount = Gaussian_Num
        coefs_start = 0
        mean_start = coefs_start + self.componentsCount
        cov_start = mean_start + 3 * self.componentsCount

        # assign coefs indices
        coefs_inds = np.arange(coefs_start, coefs_start + 1 * self.componentsCount)
        coefs_inds = np.reshape(coefs_inds, [self.componentsCount, 1])
        self.coefs = Model[0, coefs_inds]
        # assign means indices
        mean_inds = np.arange(mean_start, mean_start + 3 * self.componentsCount)
        mean_inds = np.reshape(mean_inds, [self.componentsCount, 3])
        self.means = Model[0, mean_inds]
        # assign covariances indices
        cov_inds = np.arange(cov_start, cov_start + 9 * self.componentsCount)
        cov_inds = np.reshape(cov_inds, [self.componentsCount, 9])
        self.covs = Model[0, cov_inds]

        self.covDeterms = np.zeros(self.componentsCount)
        self.inverseCovs = np.zeros([self.componentsCount, 3, 3])
        self.calcInverseCovAndDeterm()

    def calcInverseCovAndDeterm(self):
        for i in range(self.componentsCount):
            if self.coefs[i] > 0:
                cov = self.covs[i, :]
                self.covDeterms[i] = cov[0] * (cov[4] * cov[8] - cov[5] * cov[7]) - \
                                     cov[1] * (cov[3] * cov[8] - cov[5] * cov[6]) + \
                                     cov[2] * (cov[3] * cov[7] - cov[4] * cov[6])
                dtrm = self.covDeterms[i]
                if dtrm != 0.0:
                    self.inverseCovs[i][0][0] = (cov[4] * cov[8] - cov[5] * cov[7]) / dtrm
                    self.inverseCovs[i][1][0] = -(cov[3] * cov[8] - cov[5] * cov[6]) / dtrm
                    self.inverseCovs[i][2][0] = (cov[3] * cov[7] - cov[4] * cov[6]) / dtrm
                    self.inverseCovs[i][0][1] = -(cov[1] * cov[8] - cov[2] * cov[7]) / dtrm
                    self.inverseCovs[i][1][1] = (cov[0] * cov[8] - cov[2] * cov[6]) / dtrm
                    self.inverseCovs[i][2][1] = -(cov[0] * cov[7] - cov[1] * cov[6]) / dtrm
                    self.inverseCovs[i][0][2] = (cov[1] * cov[5] - cov[2] * cov[4]) / dtrm
                    self.inverseCovs[i][1][2] = -(cov[0] * cov[5] - cov[2] * cov[3]) / dtrm
                    self.inverseCovs[i][2][2] = (cov[0] * cov[4] - cov[1] * cov[3]) / dtrm

    def calculate_pixel_unary(self, color):
        '''

        :param color: np.ndarray(3,)
        :return:
        '''
        res = 0.0
        for i in range(self.componentsCount):
            res_i = 0.0
            if self.coefs[i] > 0:
                if self.covDeterms[i] > -1e-16:
                    m = self.means[i, :]
                    diff = color - m
                    mult = diff[0] * (diff[0] * self.inverseCovs[i][0][0] +
                                      diff[1] * self.inverseCovs[i][1][0] +
                                      diff[2] * self.inverseCovs[i][2][0]) + \
                           diff[1] * (diff[0] * self.inverseCovs[i][0][1] +
                                      diff[1] * self.inverseCovs[i][1][1] +
                                      diff[2] * self.inverseCovs[i][2][1]) + \
                           diff[2] * (diff[0] * self.inverseCovs[i][0][2] +
                                      diff[1] * self.inverseCovs[i][1][2] +
                                      diff[2] * self.inverseCovs[i][2][2])
                    res_i = 1.0 / np.sqrt(self.covDeterms[i]) * np.exp(-0.5 * mult)
            res += self.coefs[i] * res_i

        return res

    def get_img_unary(self, img):
        '''

        :param img: cv2.imread()->img
        :return:
        '''
        unary = np.zeros(img.shape, dtype=np.double)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                unary[i, j, :] = self.calculate_pixel_unary(img[i, j, :])

        return unary

if __name__ =='__main__':

    mask_type = ''
    componentsCount = 5
    modelSize = 1 + 3 + 9  # component weight + mean + covariance
    bgdModel = np.zeros((1, modelSize * componentsCount), np.float64)
    fgdModel = np.zeros((1, modelSize * componentsCount), np.float64)

    img = cv2.imread(image_path + filename + image_suffix)

    # newmask is the mask image I manually labelled
    newmask = cv2.imread(mask_path + filename + mask_type + mask_suffix, 0)
    # todo: 0 1 2 3 newmask
    # whereever it is marked white (sure foreground), change mask=1
    # whereever it is marked black (sure background), change mask=0
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1

    # train grabCut models
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    # mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img = img * mask[:, :, np.newaxis]
    # plt.imshow(img), plt.colorbar(), plt.show()

    # Compute Unary fgd/bgd
    fgdGMM = GMM(fgdModel)
    fgd_unary = fgdGMM.get_img_unary(img)
    bgdGMM = GMM(bgdModel)
    bgd_unary = bgdGMM.get_img_unary(img)


