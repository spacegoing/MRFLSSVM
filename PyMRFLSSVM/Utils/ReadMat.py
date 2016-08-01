# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

from scipy.io import loadmat
import numpy as np


# def loadCheckboard():
#     mat = loadmat('tmpData/linEnvLearn.mat')
#
#     return mat['options'], mat['instance']


def loadCheckboard():
    mat = loadmat('tmpData/linEnvLearn.mat')

    return mat['instance']['unary'][0][0]

def loadTestInf():
    mat = loadmat('tmpData/testInf.mat')
    unary_observed = mat['unary_observed'].astype(np.float)
    pairwise = mat['pairwise'].astype(np.float)
    w = mat['w']
    y_inferred = mat['y_inferred'].reshape([128, 128], order='F')
    z_inferred = mat['z_inferred'].reshape([9, 64], order='F')
    e = mat['e']

    return unary_observed, pairwise, w, y_inferred, z_inferred, e
