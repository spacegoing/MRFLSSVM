# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import numpy as np
from scipy.linalg import toeplitz
from scipy import optimize
import matlab.engine

from CyInf.testCases import linEnvInf

eng = matlab.engine.start_matlab()


class Params:
    def __init__(self, options):
        self.unaryWeight = 1.0
        self.pairwiseWeight = 0.0
        self.linEnvCoeffs = np.zeros([options.K, 2])


def quadprog(P, q, A, b, theta, solverName):
    def loss(x, sign=1.):
        return sign * (0.5 * np.dot(x.T, np.dot(P, x)) + np.dot(q, x))

    def jac(x, sign=1.):
        return sign * (np.dot(x.T, P) + q)

    cons = {'type': 'ineq',
            'fun': lambda x: b - np.dot(A, x),
            'jac': lambda x: -A}

    opt = {'disp': False}

    res_cons = optimize.minimize(loss, theta, jac=jac, constraints=cons,
                                 method=solverName, options=opt)

    return res_cons['x']


def quadprog_matlab(P, q, A, b, theta):
    P, q, A, b, theta = [matlab.double(i.tolist())
                         for i in [P, q, A, b, theta]]
    null = matlab.double([])
    theta = eng.quadprog(P, q, A, b, null, null, null, null, theta)

    return np.array(theta)


def featureVector(instance, y_hat, options):
    unaryPhi = np.sum(instance.unary[y_hat == 0.0, 0]) + \
               np.sum(instance.unary[y_hat == 1.0, 1])

    pairwisePhi = 0
    if instance.pairwise:
        pairwisePhi = np.sum(y_hat(instance.pairwise[:, 0]) != \
                             y_hat(instance.pairwise[:, 1]))

    highOrderPhi = np.zeros(options.K + 1)
    for cliqueId in range(1, int(instance.numCliques + 1)):
        indx = instance.cliques == cliqueId
        p = np.sum(y_hat[indx]) / np.sum(indx)
        k = int(np.floor(p * options.K))
        highOrderPhi[k] = highOrderPhi[k] + k - p * options.K + 1
        if k < options.K:
            highOrderPhi[k + 1] = highOrderPhi[k + 1] + p * options.K - k

    # todo: QP = 2 and 3

    return np.insert(np.insert(highOrderPhi, 0, pairwisePhi), 0, unaryPhi)


def linEnvLearn(instance, options):
    history = list()

    # true feature vector
    truePhi = featureVector(instance, instance.y, options)

    ## initialize parameters
    params = Params(options)

    numVariables = instance.N
    numParameters = options.K + 3  # unary, pairwise and linear coeffs

    ## construct QP objective
    P = np.eye(numParameters + 1, numParameters + 1)
    P[-1, -1] = 0.0
    q = np.zeros(numParameters + 1)
    q[-1] = 1.0e3

    ## positivity constraint on pairwise weight
    A = np.append(np.array([0.0, 1.0]), np.zeros(numParameters - 1))
    b = np.array([0.0])
    if options.K > 1:
        # todo: other cases
        if options.learningQP == 1:
            Dxx = toeplitz(np.insert(np.zeros(options.K - 2), 0, -1),
                           np.insert(np.zeros(options.K - 2), 0, [-1, 2, -1]))
            A = np.concatenate((A[np.newaxis, :],
                                np.c_[np.zeros([options.K - 1, numParameters - options.K - 1]),
                                      Dxx,
                                      np.zeros([options.K - 1, 1])])
                               )
            b = np.r_[b[np.newaxis, :], np.zeros([options.K - 1, 1])]

    # positivity constraint on slack
    A = np.r_[A, np.c_[np.zeros([1, numParameters]), 1]]
    b = np.r_[b, np.zeros([1, 1])]
    b = b.flatten()

    ## iterate until convergence
    theta = np.zeros(numParameters + 1)

    for t in range(0, options.maxIters):

        theta = quadprog_matlab(P, q, -A, -b, theta)
        params.unaryWeight = theta[0]
        params.pairwiseWeight = max(0, theta[1])

        if instance.pairwise:
            instance.pairwise[:, 2] = params.pairwiseWeight

        params.linEnvCoeffs[0, 1] = theta[2]
        for k in range(options.K):
            params.linEnvCoeffs[k, 0] = theta[k + 3] - theta[k + 2]

        for k in range(1, options.K):
            params.linEnvCoeffs[k, 1] = params.linEnvCoeffs[k - 1, 1] + \
                                        k * (params.linEnvCoeffs[k - 1, 0] -
                                             params.linEnvCoeffs[k, 0])
        params.linEnvCoeffs[:, 0] = params.linEnvCoeffs[:, 0] * options.K

        y_hat, z_hat, e_hat = linEnvInf(params.unaryWeight * instance.unary, instance.pairwise,
                                        params.linEnvCoeffs, instance.cliques)

        history.append({'params': params, 'y_hat': y_hat})

        # infer most violated constraint

        # Todo: Is there any bug?
        lossUnary = np.zeros([instance.N, 2])
        lossUnary[instance.y == 1, 0] = 1.0 / instance.N
        # lossUnary[instance.y == 1, 1] = 1.0 / instance.N

        y_loss, z_loss, e_loss = linEnvInf(params.unaryWeight * instance.unary - lossUnary,
                                           instance.pairwise, params.linEnvCoeffs, instance.cliques)
        # add constraint
        phi = featureVector(instance, y_loss, options)
        loss = np.sum(y_loss != instance.y) / numVariables
        slack = loss - np.dot((phi - truePhi), theta[:-1])[0]
        violation = slack - theta[-1]

        if violation < options.eps:
            break

        A = np.r_[A, [np.r_[phi - truePhi, 1]]]
        b = np.r_[b, loss]

    return history, y_hat
