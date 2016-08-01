# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

from CyInf.WllepGraphCut import Inf_Algo
from Utils.ReadMat import loadTestInf, loadCheckboard
import numpy as np
from pprint import pprint as pp

# initialize random number generator
np.random.seed(0)


# Generate checkboard data
def checkboardHelper(H, W):
    cliques = np.zeros([H, W])  # mapping of variables to clique_indexes
    y = np.zeros([H, W])  # ground-truth labels

    # create checkboard data
    _black = True  # indicate _black True or white False
    _cliqueID = 1.0
    for _rowIndx in range(0, H, Options.gridStep):
        for _colIndx in range(0, W, Options.gridStep):
            cliques[_rowIndx:_rowIndx + Options.gridStep,
            _colIndx:_colIndx + Options.gridStep] = _cliqueID
            _cliqueID += 1.0

            y[_rowIndx:_rowIndx + Options.gridStep,
            _colIndx:_colIndx + Options.gridStep] = 0.0 if _black else 1.0
            _black = not _black

        _black = not _black

    return cliques.flatten('F'), y.flatten('F')


class Options:
    K = 10  # number of lower linear functions
    gridStep = 16  # grid size for defining clique_indexes
    maxIters = 100  # maximum learning iterations
    eps = 1.0e-16  # constraint violation threshold
    learningQP = 1  # encoding for learning QP (1, 2, or 3)
    figWnd = 0  # figure for showing results


class Instance:
    W = 128  # image width
    H = 128  # image height
    numCliques = (W / Options.gridStep) ** 2  # number of clique_indexes

    N = W * H  # number of variables

    def __init__(self):
        # flatten arrays
        self.cliques, self.y = checkboardHelper(self.H, self.W)

        # # create noisy observations
        # _eta = [0.1, 0.1]
        # self.unary = np.zeros([self.N, 2])
        # self.unary[:, 1] = 2 * (np.random.rand(self.N, 1).flatten() - 0.5) + \
        #               (_eta[0] * (1 - self.y) - _eta[1] * self.y)
        self.unary = loadCheckboard().astype(np.float64)

        # no pairwise edges
        self.pairwise = np.array([])


# # U+H experiments --------------------------------------------
instance = Instance()


class Options:
    H = 128
    W = 128
    numCliques = 64
    K = 10
    learningQP = 1


unary_observed, pairwise_raw, w_raw, y_inferred, z_inferred, e = loadTestInf()
options = Options()
# float[:, :, :] observed_unary, float[:, :] pairwise,
# int[:, :] clique_indexes,
# int[:, :] inferred_label, int[:, :] inferred_z,
# double[:] w, options
observed_unary = np.zeros([128, 128, 2], dtype=np.double)
for i in range(128):
    for j in range(128):
        observed_unary[i][j][0] = unary_observed[i][j][0]
        observed_unary[i][j][1] = unary_observed[i][j][1]

pairwise = np.zeros(pairwise_raw.shape, order='C', dtype=np.double)
for i in range(pairwise_raw.shape[0]):
    for j in range(3):
        pairwise[i][j] = pairwise_raw[i][j]

indexes = instance.cliques.reshape([128, 128], order='C').astype(np.int32).transpose()
clique_indexes = np.zeros([128, 128], order='C', dtype=np.int32)
for i in range(128):
    for j in range(128):
        clique_indexes[i][j] = indexes[i][j]

inferred_label = np.zeros([128, 128], dtype=np.int32)
inferred_z = np.zeros([64, 9], dtype=np.int32)
theta = np.zeros(1 + 2 * 9, dtype=np.double)
theta[0] = w_raw[0, 0]
for i in range(1, 10):
    theta[i] = w_raw[i, 0] - w_raw[i - 1, 0]
    theta[i + 9] = w_raw[i, 1] - w_raw[i - 1, 1]

e_i = Inf_Algo(observed_unary, pairwise, clique_indexes, inferred_label, inferred_z, theta, options)

for i in range(128):
    for j in range(128):
        if inferred_label[i][j] != y_inferred[i][j]:
            print(str(i) + ' ' + str(j))

for i in range(64):
    for j in range(9):
        if inferred_z[i][j] != z_inferred.T[i][j]:
            print(str(i) + str(j))
