# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

from scipy.io import loadmat
import numpy as np


def loadMatPairwise():
    mat = loadmat('tmpData/learnPairwise.mat')
    observed_unary = mat['instance']['unary'][0][0]
    observed_unary = np.reshape(observed_unary, [128, 128, 2], order='F')
    pairwise_raw = mat['instance']['pairwise'][0][0]

    unary_observed = np.zeros([128, 128, 2], order='C', dtype=np.double)
    for i in range(128):
        for j in range(128):
            unary_observed[i][j][0] = observed_unary[i][j][0]
            unary_observed[i][j][1] = observed_unary[i][j][1]

    pairwise = np.zeros(pairwise_raw.shape, order='C', dtype=np.double)
    for i in range(pairwise_raw.shape[0]):
        for j in range(3):
            pairwise[i][j] = pairwise_raw[i][j]
    pairwise[:, 0] -= 1
    pairwise[:, 1] -= 1

    return unary_observed, pairwise


def loadTestInf():
    mat = loadmat('tmpData/testInf.mat')
    observed_unary = mat['unary_observed'].astype(np.double)
    pairwise_raw = mat['pairwise'].astype(np.double)
    w = mat['w']
    y_inferred = mat['y_inferred'].reshape([128, 128], order='F')
    z_inferred = mat['z_inferred'].reshape([9, 64], order='F')
    e = mat['e']

    unary_observed = np.zeros([128, 128, 2], order='C', dtype=np.double)
    for i in range(128):
        for j in range(128):
            unary_observed[i][j][0] = observed_unary[i][j][0]
            unary_observed[i][j][1] = observed_unary[i][j][1]

    pairwise = np.zeros(pairwise_raw.shape, order='C', dtype=np.double)
    for i in range(pairwise_raw.shape[0]):
        for j in range(3):
            pairwise[i][j] = pairwise_raw[i][j]

    theta = np.zeros(1 + 2 * 9, order='C', dtype=np.double)
    theta[0] = w[0, 0]
    for i in range(1, 10):
        theta[i] = w[i, 0] - w[i - 1, 0]
        theta[i + 9] = w[i, 1] - w[i - 1, 1]

    return unary_observed, pairwise, theta, y_inferred, z_inferred, e
