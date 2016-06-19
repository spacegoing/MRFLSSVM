# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import numpy as np
from CyInf.InfAlgo import linEnvInf

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
    kStarArray = np.sum(latent_var,1)  # Length = K. The largest index of z_k = 1

    highOrderPhi = np.zeros(2 * options.K - 1)
    for cliqueId in range(1, int(instance.numCliques + 1)):
        indx = instance.cliques == cliqueId
        p = np.sum(ground_truth[indx]) / np.sum(indx)
        k = kStarArray[cliqueId-1]
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