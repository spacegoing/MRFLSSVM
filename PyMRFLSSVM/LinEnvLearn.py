# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import toeplitz
from scipy import optimize
import matlab.engine
from MrfTypes import Example, Options
from Batch_MRF_Helpers import inf_label_latent_helper
import sys

__author__ = 'spacegoing'


def quadprog_matlab(P, q, A, b, eng):
    P, q, A, b = [matlab.double(i.tolist())
                  for i in [P, q, A, b]]
    null = matlab.double([])
    theta = eng.quadprog(P, q, A, b, null, null, null, null)

    return np.array(theta, dtype=np.double, order='C').reshape(len(theta))


def phi_helper(ex, label_inferred, options):
    '''

    :param ex:
    :type ex: Example
    :param label_inferred:
    :type label_inferred: np.ndarray
    :param options:
    :type options: Options
    :return:
    :rtype: np.ndarray
    '''

    # unary phi
    unary_phi = sum(ex.unary_observed[:, :, 0].flatten()[label_inferred.flatten() == 0]) + \
                sum(ex.unary_observed[:, :, 1].flatten()[label_inferred.flatten() == 1])

    # pairwise phi
    pairwise_phi = 0
    if ex.hasPairwise:
        label = label_inferred.flatten()
        for i1, i2, value in ex.pairwise:
            if label[int(i1)] != label[int(i2)]:
                pairwise_phi += value

    highOrderPhi = np.zeros(options.K + 1)
    for cliqueId in range(1, ex.numCliques + 1):
        indx = ex.clique_indexes == cliqueId
        p = np.sum(label_inferred[indx]) / np.sum(indx)
        k = int(np.floor(p * options.K))
        highOrderPhi[k] = highOrderPhi[k] + k - p * options.K + 1
        if k < options.K:
            highOrderPhi[k + 1] = highOrderPhi[k + 1] + p * options.K - k

    phi = np.zeros(options.sizePhi, dtype=np.double)
    phi[:options.sizeHighPhi] = highOrderPhi
    phi[-2] = unary_phi
    phi[-1] = pairwise_phi

    return phi


def augmented_loss(ex):
    '''
    :param ex:
    :type ex: Example
    :return:
    :rtype:
    '''
    # Loss augmented inference
    lossUnary = np.zeros([ex.rows, ex.cols, 2])
    lossUnary[ex.y == 1, 0] = 1.0 / ex.numVariables
    lossUnary[ex.y == 0, 1] = 1.0 / ex.numVariables

    return lossUnary


def decode_theta(theta, options):
    '''

    :param theta:
    :type theta: np.ndarray
    :param options:
    :type options: Options
    :return:
    :rtype:
    '''
    # decode theta
    a_b_array = np.zeros([options.K, 2])
    a_b_array[0, 1] = theta[0]
    for k in range(options.K):
        a_b_array[k, 0] = theta[k + 1] - theta[k]

    for k in range(1, options.K):
        a_b_array[k, 1] = a_b_array[k - 1, 1] + \
                          k * (a_b_array[k - 1, 0] - a_b_array[k, 0])
    a_b_array[:, 0] = a_b_array[:, 0] * options.K

    return a_b_array


def encode_latent_theta(theta_pami, options):
    '''

    :param theta:
    :type theta: np.ndarray
    :param options:
    :type options: Options
    :return:
    :rtype: np.ndarray
    '''

    theta = np.zeros([2 * options.K + 1 + 1], dtype=np.double)
    a_b_array = decode_theta(theta_pami, options)

    theta[0] = a_b_array[0, 0]
    theta[-3:] = theta_pami[-3:]
    for i in range(1, options.K):
        theta[i] = a_b_array[i, 0] - a_b_array[i - 1, 0]
        theta[i + options.K - 1] = a_b_array[i, 1] - a_b_array[i - 1, 1]

    return theta


def pami_cutting_plane_ssvm(examples_list, options, eng):
    '''

    :param examples_list:
    :type examples_list: list[Example]
    :param options:
    :type options: Options
    :param eng:
    :type eng:
    :return:
    :rtype:
    '''
    history = list()
    examples_num = len(examples_list)

    lossUnary_list = list()
    for ex in examples_list:
        lossUnary_list.append(augmented_loss(ex))

    vt_list = list()
    for ex in examples_list:
        vt_list.append(phi_helper(ex, ex.y, options))

    ## initialize parameters
    theta = np.zeros(options.sizePhi + 1, dtype=np.double)
    # unary weight = 1
    theta[-3] = 1.0

    ## construct QP objective
    P = np.eye(options.sizePhi + 1, options.sizePhi + 1)
    P[-1, -1] = 0.0  # for slack variable
    q = np.zeros(options.sizePhi + 1)
    q[-1] = 1.0e3  # for slack variable

    # convex constraint on higher-order weights
    A = np.zeros([options.K + 1, options.sizePhi + 1])
    b = np.zeros(options.K + 1)
    for i in range(options.K - 1):
        A[i, i:i + 3] = np.asarray([-1.0, 2.0, -1.0])
    # positivity constraint on pairwise weight
    A[-2, -2] = 1.0
    # positivity constraint on slack
    A[-1, -1] = 1.0

    y_hat_loss_list = list()

    y_hat_loss = 0
    for ex in examples_list:
        pairwise = np.copy(ex.pairwise)
        if ex.hasPairwise:
            pairwise[:, 2] = theta[-2] * ex.pairwise[:, 2]
        y_hat = inf_label_latent_helper(
            theta[-3] * ex.unary_observed, pairwise,
            ex.clique_indexes, encode_latent_theta(theta, options),
            options, ex.hasPairwise)[0]
        y_hat_loss += np.sum(y_hat != ex.y) / (ex.y.shape[0] * ex.y.shape[1])
    y_hat_loss_list.append(y_hat_loss / examples_num)

    counter = 0
    ## iterate until convergence
    for t in range(0, options.maxIters):
        print("inner iter %d" % t)
        sys.stdout.flush()

        y_hat_loss_old = y_hat_loss
        theta_old = theta
        theta = quadprog_matlab(P, q, -A, -b, eng)

        y_hat_loss = 0
        for ex in examples_list:
            pairwise = np.copy(ex.pairwise)
            if ex.hasPairwise:
                pairwise[:, 2] = theta[-2] * ex.pairwise[:, 2]
            y_hat = inf_label_latent_helper(
                theta[-3] * ex.unary_observed, pairwise,
                ex.clique_indexes, encode_latent_theta(theta, options),
                options, ex.hasPairwise)[0]
            y_hat_loss += np.sum(y_hat != ex.y) / (ex.y.shape[0] * ex.y.shape[1])
        y_hat_loss_list.append(y_hat_loss / examples_num)

        if np.abs(y_hat_loss - y_hat_loss_old) < options.eps:
            counter += 1
            if counter == 3:
                history.append({'theta': theta, 'loss_aug': loss, 'y_hat_loss_list': y_hat_loss_list})
                return theta, history
        else:
            counter = 0

        # Decode parameters
        unaryWeight = theta[-3]
        pairwiseWeight = max(0, theta[-2])

        loss_arr = np.zeros(examples_num)
        phi_loss_sum = np.zeros(options.sizePhi, dtype=np.double, order='C')
        violation_sum = 0.0
        for vt, ex, lossUnary, m in zip(vt_list, examples_list,
                                        lossUnary_list, range(examples_num)):

            pairwise = np.copy(ex.pairwise)
            if ex.hasPairwise:
                pairwise[:, 2] = pairwiseWeight * ex.pairwise[:, 2]

            # infer most violated constraint
            y_loss, z_loss, e_loss = \
                inf_label_latent_helper(unaryWeight * ex.unary_observed - lossUnary,
                                        pairwise, ex.clique_indexes,
                                        encode_latent_theta(theta, options),
                                        options, ex.hasPairwise)

            # add constraint
            phi = phi_helper(ex, y_loss, options)
            phi_loss_sum += phi - vt

            loss_arr[m] = np.sum(y_loss != ex.y) / ex.numVariables
            slack = loss_arr[m] - np.dot((phi - vt), theta[:-1])
            violation_sum += slack - theta[-1]

        loss = np.sum(loss_arr) / examples_num
        phi_loss = phi_loss_sum / examples_num
        A = np.r_[A, [np.r_[phi_loss, 1]]]
        b = np.r_[b, loss]

        if violation_sum / examples_num < options.eps:
            history.append({'theta': theta, 'loss_aug': loss, 'y_hat_loss_list': y_hat_loss_list})
            return theta, history

        if all(theta_old == theta):
            print("inner loop break at %d" % t)
            history.append({'theta': theta, 'loss_aug': loss, 'y_hat_loss_list': y_hat_loss_list})
            return theta, history

        history.append({'theta': theta, 'loss_aug': loss, 'y_hat_loss_list': y_hat_loss_list})

    return theta, history


if __name__ == "__main__":
    from Checkboard import Instance
    from MrfTypes import BatchExamplesParser
    from Utils.ReadMat import loadMatPairwise

    # Generate instance
    parser = BatchExamplesParser()
    instance = Instance()
    unary_observed, pairwise = loadMatPairwise()
    instance.unary_observed = unary_observed
    # Original is 0.0
    pairwise[:, 2] = 1.0
    instance.pairwise = pairwise
    examples_list = parser.parse_checkboard(instance)
    ex = examples_list[0]

    options = Options()
    options.sizeHighPhi = options.K + 1
    options.sizePhi = options.K + 3

    # label_inferred = ex.y

    eng = matlab.engine.start_matlab()

    theta, history = pami_cutting_plane_ssvm(examples_list, options, eng)
    print(decode_theta(theta, options))
