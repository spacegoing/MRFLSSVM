# -*- coding: utf-8 -*-
import numpy as np
from numpy.matlib import repmat

# from pprint import pprint as pp
# from Utils.ReadMat import loadCheckboard

__author__ = 'spacegoing'

# initialize random number generator
np.random.seed(0)


class Options:
    # Image Configs
    gridStep = 16  # grid size for defining clique_indexes
    H = 128  # rows image height
    W = 128  # cols image width
    numCliques = (int)(H / gridStep) ** 2  # number of clique_indexes
    numVariables = H * W
    N = H * W  # number of variables

    dimUnary = 2
    dimPairwise = 3

    # Learning Configs
    K = 10  # number of lower linear functions
    sizeHighPhi = 2 * K - 1
    sizePhi = sizeHighPhi + 2
    maxIters = 100  # maximum learning iterations
    eps = 1.0e-16  # constraint violation threshold

    # Other Configs
    learningQP = 1  # encoding for learning QP (1, 2, or 3)
    figWnd = 0  # figure for showing results
    hasPairwise = True  # dimPairwise = 0 when it's false
    rowsPairwise = H * W * 2 - H - W


# Generate checkboard data
def checkboardHelper():
    H = Options.H
    W = Options.W

    # create checkboard data (clique and ground_truth y)
    clique_indexes = np.zeros([H, W], dtype=np.int32, order='C')  # mapping of variables to clique_indexes
    y = np.zeros([H, W], dtype=np.int32, order='C')  # ground-truth labels
    _black = True  # indicate _black True or white False
    _cliqueID = 1.0  # clique index starts from 1
    for _rowIndx in range(0, H, Options.gridStep):
        for _colIndx in range(0, W, Options.gridStep):
            clique_indexes[_rowIndx:_rowIndx + Options.gridStep,
            _colIndx:_colIndx + Options.gridStep] = _cliqueID
            _cliqueID += 1.0

            y[_rowIndx:_rowIndx + Options.gridStep,
            _colIndx:_colIndx + Options.gridStep] = 0.0 if _black else 1.0
            _black = not _black

        _black = not _black

    # create unary features
    unary_observed = np.zeros([H, W, 2], dtype=np.double, order='C')
    _eta = [0.1, 0.1]
    for i in range(H):
        for j in range(W):
            unary_observed[i][j][1] = 2 * (np.random.rand(1, 1)[0, 0] - 0.5) + \
                                      _eta[0] * (1 - y[i][j]) - _eta[1] * y[i][j]

    # create pairwise features
    pairwise = np.zeros([Options.rowsPairwise, 3], dtype=np.double, order='C')
    if (Options.hasPairwise):
        u = repmat(np.arange(W), H - 1, 1) * H + \
            repmat(np.arange(H - 1).reshape([H - 1, 1]), 1, W)
        v = repmat(np.arange(W - 1), H, 1) * H + \
            repmat(np.arange(H).reshape([H, 1]), 1, W - 1)

        for i, (u_i, v_i) in enumerate(zip(u.flatten('F'), v.flatten('F'))):
            pairwise[i, 0] = u_i
            pairwise[i, 1] = u_i + 1
            pairwise[i + u.shape[0] * u.shape[1], 0] = v_i
            pairwise[i + u.shape[0] * u.shape[1], 1] = v_i + H

    return clique_indexes, y, unary_observed, pairwise


class Instance:
    def __init__(self):
        # clique index starts from 1
        self.clique_indexes, self.y, self.unary_observed, self.pairwise = checkboardHelper()
        self.latent_var = np.zeros([Options.numCliques, Options.K - 1])
        # self.unary = loadCheckboard().astype(np.double)


if __name__ == '__main__':
    instance = Instance()
    options = Options()
