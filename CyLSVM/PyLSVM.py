# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import numpy as np
from CyInf.InfAlgo import linEnvInf
from Utils.ReadMat import loadCheckboard
import numpy as np
from pprint import pprint as pp
from linEnvLearn import linEnvLearn

# initialize random number generator
np.random.seed(0)

# Generate checkboard data
def checkboardHelper(H, W):
    cliques = np.zeros([H, W],dtype=np.int32)  # mapping of variables to cliques
    y = np.zeros([H, W], dtype=np.int32)  # ground-truth labels

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
    gridStep = 16  # grid size for defining cliques
    maxIters = 100  # maximum learning iterations
    eps = 1.0e-16  # constraint violation threshold
    learningQP = 1  # encoding for learning QP (1, 2, or 3)
    figWnd = 0  # figure for showing results


class Instance:
    W = 128  # image width
    H = 128  # image height
    numCliques = (W / Options.gridStep) ** 2  # number of cliques

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

instance = Instance()


def loss_function(y_hat, instance):
    return np.sum(y_hat != instance.y) / instance.N


def feature_vector(ground_truth, latent_var, instance, options):
    """

    :param ground_truth: y labels
    :param latent_var: 64 X K Matrix, each row stands for a vector
    :param instance:
    :param options:
    :return:
    """
    kStarArray = np.sum(latent_var, 1)  # Length = K. The largest index of z_k = 1

    highOrderPhi = np.zeros(2 * options.K - 1)
    for cliqueId in range(1, int(instance.numCliques + 1)):
        indx = instance.cliques == cliqueId
        p = np.sum(ground_truth[indx]) / np.sum(indx)
        k = kStarArray[cliqueId - 1]
        highOrderPhi[:k] = highOrderPhi[:k] + p


def joint_inference(params, instance):
    y_hat, z_hat, e_hat = linEnvInf(params.unaryWeight * instance.unary, instance.pairwise,
                                    params.linEnvCoeffs, instance.cliques)
    return y_hat, z_hat, e_hat


def latent_inference(params, instance):
    y_hat, z_hat, e_hat = linEnvInf(params.unaryWeight * instance.unary, instance.pairwise,
                                    params.linEnvCoeffs, instance.cliques)
    return y_hat, z_hat, e_hat


def most_violated_constraint(params, instance, y_star):
    loss = loss_function(y_star, instance.y)
    y_hat, z_hat, e_hat = linEnvInf(params.unaryWeight * instance.unary - loss, instance.pairwise,
                                    params.linEnvCoeffs, instance.cliques)
    return y_hat, z_hat, e_hat


def decode_weights(higher_order_weight):
    K = (higher_order_weight.size + 1) / 2
    a = np.zeros(K)
    a[0] = higher_order_weight[0]
    b = np.zeros(K)

    for i, da in enumerate(higher_order_weight[1:K]):
        a[i] = da + a[i - 1]

    for i, db in enumerate(higher_order_weight[K:2 * K - 1]):
        b[i] = db + b[i + 1 - K]

    return a, b
