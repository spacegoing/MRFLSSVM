# -*- coding: utf-8 -*-
import numpy as np
from GrabCut.GMM_Unary import GrabCut
from GrabCut.Superpixel import superpixel
import os

__author__ = 'spacegoing'


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


class GrabCutInstance:
    def __init__(self, image_path, mask_path, true_mask_path,
                 method='quickshift', numSegments=200):

        self.img_grabcut = GrabCut(image_path, mask_path)

        self.input_mask = self.img_grabcut.input_mask
        self.output_mask = self.img_grabcut.output_mask
        self.true_mask = self.img_grabcut.read_mask_img(true_mask_path)
        self.img = self.img_grabcut.img

        self.unary_observed = self.get_unary_observed()
        self.pairwise = self.get_pairwise(self.img)
        self.cliques = self.get_cliques(image_path, method, numSegments)

    def get_unary_observed(self):
        return self.img_grabcut.get_unary_observed()

    def get_cliques(self, image_path, method='quickshift', numSegments=200):
        '''
        Clique id starts from 1

        :param image_path:
        :return:
        '''
        if method == 'slic':
            cliques = superpixel(image_path, method, numSegments=numSegments) + 1
        else:
            cliques = superpixel(image_path) + 1

        return cliques

    # calculate beta - parameter of grabcut algorithm.
    # beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
    def calcbeta(self, img):
        beta = 0
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):

                color = img[y, x, :]

                if x > 0:  # left
                    diff = color - img[y, x - 1]
                    beta += diff.dot(diff)

                if y > 0 and x > 0:  # upleft
                    diff = color - img[y - 1, x - 1]
                    beta += diff.dot(diff)

                if y > 0:  # up
                    diff = color - img[y - 1, x]
                    beta += diff.dot(diff)

                if y > 0 and (x < img.shape[1] - 1):  # upright
                    diff = color - img[y - 1, x + 1]
                    beta += diff.dot(diff)

        if beta <= 1e-16:
            beta = 0
        else:
            beta = 1.0 / (
                2 * beta / (4 * img.shape[1] * img.shape[0] - 3 * img.shape[1] -
                            3 * img.shape[0] + 2))

        return beta

    #   calculate weights of noterminal vertices of graph.
    #   beta and gamma - parameters of grabcut algorithm.
    def calcnweights(self, img, beta, gamma=50):
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

        return upleftw, leftw, upw, uprightw

    def get_pairwise(self, img):
        beta = self.calcbeta(img)
        upleftw, leftw, upw, uprightw = self.calcnweights(img, beta)

        pairwise_list = list()
        current_indx = 0
        cols = img.shape[1]
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):

                if x > 0:  # left
                    pairwise_list.append([current_indx, current_indx - 1,
                                          leftw[y, x]])

                if y > 0 and x > 0:  # upleft
                    pairwise_list.append([current_indx, current_indx - cols - 1,
                                          upleftw[y, x]])

                if y > 0:  # up
                    pairwise_list.append([current_indx, current_indx - cols,
                                          upw[y, x]])

                if y > 0 and (x < img.shape[1] - 1):  # upright
                    pairwise_list.append([current_indx, current_indx - cols + 1,
                                          uprightw[y, x]])

                current_indx += 1

        pairwise = np.asarray(pairwise_list, dtype=np.double)

        return pairwise


if __name__ == '__main__':
    image_dir = './GrabCut/Data/grabCut/images/'

    mask_dir = './GrabCut/Data/grabCut/labels/'
    mask_ground_truth_type = ''
    mask_input_type = '_new'

    name_image_mask_truemask = get_name_path_arr(image_dir, mask_dir,
                                                 mask_ground_truth_type, mask_input_type)

    i = 20
    image_name = name_image_mask_truemask[i, 0]
    image_path = name_image_mask_truemask[i, 1]
    mask_path = name_image_mask_truemask[i, 2]
    true_mask_path = name_image_mask_truemask[i, 3]

    grabInstance = GrabCutInstance(image_path, mask_path, true_mask_path,
                                   method='slic', numSegments=300)
    print(grabInstance.img.shape)
    print(np.max(grabInstance.unary_observed))
    print(np.min(grabInstance.unary_observed))
    print(np.unique(grabInstance.cliques))
    print(np.max(grabInstance.pairwise))
    print(grabInstance.pairwise.shape)
