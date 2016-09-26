# -*- coding: utf-8 -*-
import numpy as np
from skimage.segmentation import quickshift, slic, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

__author__ = 'spacegoing'


def superpixel(image_path, method='quickshift', if_plot=False, numSegments=200):
    # load the image and convert it to a floating point data type
    img = io.imread(image_path)
    image = img_as_float(img)
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    if method == 'quickshift':
        segments = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    elif method == 'slic':
        segments = slic(img, numSegments, sigma=5)
    elif method == 'felzenszwalb':
        segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    else:
        raise TypeError('Method %s doesn\'t exists' % method)

    if if_plot:
        # show the output of SLIC
        fig = plt.figure("Superpixels -- %s segments" % method)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
        # show the plots
        plt.show()

    return segments.astype(np.int32)


if __name__ == '__main__':
    image_dir = './GrabCut/Data/grabCut/images/'
    filename = '106024'
    image_suffix = '.jpg'
    image_path = image_dir + filename + image_suffix
    numSegments = 200

    segments = superpixel(image_path, if_plot=True)
    np.unique(segments)
