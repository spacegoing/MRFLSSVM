# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import numpy as np
import cv2
from matplotlib import pyplot as plt


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

        return -np.log(res)

    def get_img_unary(self, img):
        '''

        :param img: cv2.imread()->img
        :return:
        '''
        unary = np.zeros([img.shape[0], img.shape[1]], dtype=np.double)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                unary[i, j] = self.calculate_pixel_unary(img[i, j, :])

        return unary


class GrabCut:
    componentsCount = 5
    modelSize = 1 + 3 + 9  # component weight + mean + covariance
    bgdModel = np.zeros((1, modelSize * componentsCount), np.float64)
    fgdModel = np.zeros((1, modelSize * componentsCount), np.float64)

    def __init__(self, image_path, mask_path):
        self.img = cv2.imread(image_path).astype(np.int32)
        self.input_mask = self.read_mask_img(mask_path)
        self._train()

    def read_mask_img(self, mask_path):
        mask_img = cv2.imread(mask_path, 0).astype(np.int32)  # 0 stands for greyscale image
        input_mask = np.zeros(self.img.shape[:2], np.int32)
        # todo: check all have 0 64 128 255
        # cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD
        # 0 1 2 3
        # wherever it is marked sure background (0), change input_mask=0
        # wherever it is marked sure foreground (1), change input_mask=1
        input_mask[mask_img == 0] = 0
        input_mask[mask_img == 255] = 1
        input_mask[mask_img == 64] = 2
        input_mask[mask_img == 128] = 3

        return input_mask

    def _train(self):
        # _train grabCut models
        self.output_mask, self.bgdModel, self.fgdModel = \
            cv2.grabCut(self.img.astype('uint8'), self.input_mask.astype('uint8'), None,
                        self.bgdModel, self.fgdModel, 5,
                        cv2.GC_INIT_WITH_MASK)

        self.output_mask = np.where((self.output_mask == 2) |
                                    (self.output_mask == 0), 0, 1).astype(np.int32)

    def get_unary_observed(self):
        # Compute Unary fgd/bgd
        self.bgdGMM = GMM(self.bgdModel)
        bgd_unary = self.bgdGMM.get_img_unary(self.img)
        self.fgdGMM = GMM(self.fgdModel)
        fgd_unary = self.fgdGMM.get_img_unary(self.img)

        self.unary_observed = np.zeros([fgd_unary.shape[0], fgd_unary.shape[1], 2], dtype=np.double)
        self.unary_observed[:, :, 0] = bgd_unary
        self.unary_observed[:, :, 1] = fgd_unary

        return self.unary_observed

    def plot_images(self):
        f, ax = plt.subplots(2)
        ax[0].set_title('Original Image')
        ax[0].imshow(self.img)
        ax[0].axis('off')

        ax[1].set_title('Predicted Image')
        img = self.img * self.output_mask[:, :, np.newaxis]
        ax[1].imshow(img)
        ax[1].axis('off')

    def plot_raw_mask(self):
        plt.figure()
        plt.imshow(self.raw_mask)
        plt.show()


if __name__ == '__main__':
    image_dir = './GrabCut/Data/grabCut/images/'
    mask_dir = './GrabCut/Data/grabCut/labels/'
    filename = '106024'
    image_suffix = '.jpg'
    mask_suffix = '.bmp'

    image_path = image_dir + filename + image_suffix

    mask_type = '_rect'
    mask_path = mask_dir + filename + mask_type + mask_suffix

    img_grabcut = GrabCut(image_path, mask_path)
    img_grabcut.plot_images()
    img_grabcut.plot_raw_mask()
