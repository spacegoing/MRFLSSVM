# -*- coding: utf-8 -*-
import numpy as np
import matlab.engine
import Batch_MRF_Helpers as mrf
from MrfTypes import Example, Options
import pickle
import sys

__author__ = 'spacegoing'


def quadprog_matlab(P, q, A, b, eng):
    P, q, A, b = [matlab.double(i.tolist())
                  for i in [P, q, A, b]]
    null = matlab.double([])
    theta = eng.quadprog(P, q, A, b, null, null, null, null)

    return np.array(theta, dtype=np.double, order='C').reshape(len(theta))


def cutting_plane_ssvm(theta, vt_list, examples_list, lossUnary_list, options, eng):
    '''

    :param theta:
    :type theta:
    :param vt_list:
    :type vt_list: list[np.ndarray]
    :param examples_list:
    :type examples_list: list[Example]
    :type lossUnary_list: list[np.ndarray]
    :param options:
    :type options: Options
    :return:
    :rtype: np.ndarray
    :rtype: list[dict]
    '''
    # theta: first 2K - 1 are higher order params, then unary, pairwise and slack
    # vt = Inferred Cutting Plane = truePhi
    # instance = Instance()
    # options = Options()

    history = list()
    examples_num = len(examples_list)

    ## construct QP objective
    P = np.eye(options.sizePhi + 1, options.sizePhi + 1)
    P[-1, -1] = 0.0  # for slack variable
    q = np.zeros(options.sizePhi + 1)
    q[-1] = 1.0e3  # for slack variable

    ################## Adding Constraints ##########################
    # positivity constraint on pairwise weight
    A = np.zeros([2 * options.K, options.sizePhi + 1], dtype=np.double, order='C')
    b = np.zeros(2 * options.K, dtype=np.double, order='C')
    # options.K-1s' positive constraints for a_k - a_{k+1} ---------
    # A[0] = [0 -1 0 0...]
    # A[1] = [0 0 -1 0...]
    # A[K-2] = [0 0 0 -1...]
    for i in range(0, options.K - 1):
        A[i, i + 1] = -1
    # options.K-1s' positive constraints for b_{k+1} - b_k ---------
    # A[K-1] = [0 0 0 0 ... 1 0 0 0 ...]
    # A[K]   = [0 0 0 0 ... 0 1 0 0 ...]
    # A[2*K-3] = [0 0 0 0 ... 0 0 0 1 ...]
    for i in range(options.K - 1, options.sizeHighPhi - 1):
        A[i, i + 1] = 1
    # 1 positive constraints for pairwise
    A[options.sizeHighPhi - 1, options.sizePhi - 1] = 1
    # 1 positive constraints for slack
    A[options.sizeHighPhi, options.sizePhi] = 1

    ################## iterate until convergence ####################
    for t in range(0, options.maxIters):
        print("inner iter %d" % t)
        sys.stdout.flush()

        theta_old = theta
        theta = quadprog_matlab(P, q, -A, -b, eng)

        # Decode parameters
        unaryWeight = theta[options.sizeHighPhi]
        pairwiseWeight = max(0, theta[options.sizeHighPhi + 1])

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
                mrf.inf_label_latent_helper(unaryWeight * ex.unary_observed - lossUnary,
                                            pairwise, ex.clique_indexes,
                                            theta, options, ex.hasPairwise)

            # add constraint
            phi = mrf.phi_helper(ex, y_loss, z_loss, options)
            phi_loss_sum += phi - vt

            loss_arr[m] = np.sum(y_loss != ex.y) / ex.numVariables
            slack = loss_arr[m] - np.dot((phi - vt), theta[:-1])
            violation_sum += slack - theta[-1]

        if violation_sum / examples_num < options.eps:
            break

        loss = np.sum(loss_arr) / examples_num
        phi_loss = phi_loss_sum / examples_num
        A = np.r_[A, [np.r_[phi_loss, 1]]]
        b = np.r_[b, loss]

        history.append({'theta': theta, 'loss_aug': loss})

        if all(theta_old == theta):
            print("inner loop break at %d" % t)
            break

    return theta, history


def cccp_outer_loop(examples_list, options, init_method='', inf_latent_method='', batch_name=''):
    '''


    # inf_latent_method = 'remove_redund'
    # init_method = 'clique_by_clique'

    :param examples_list:
    :type examples_list: list[Example]
    :param options:
    :type options: Options
    :param init_method:
    :type init_method: str
    :param inf_latent_method:
    :type inf_latent_method: str
    :return:
    :rtype:
    '''
    eng = matlab.engine.start_matlab()

    outer_history = list()  # type: list[dict]

    examples_num = len(examples_list)

    if init_method == 'clique_by_clique':
        theta = mrf.init_theta_concave(examples_list[0], options)
    elif init_method == 'zeros':
        theta = np.zeros(options.sizePhi + 1, dtype=np.double, order='C')
        theta[options.sizeHighPhi] = 1  # set unary weight to 1
    elif init_method == 'ones':
        theta = np.ones(options.sizePhi + 1, dtype=np.double, order='C')
    else:
        theta = np.asarray([np.random.uniform(-1, 1, 1)[0]] +
                           # a_0
                           list(-1 * np.random.rand(1, options.K - 1)[0, :]) + \
                           # a_2 -> a_K
                           list(np.random.rand(1, options.K - 1)[0, :]) + \
                           # b_2 -> b_K
                           [np.random.uniform(-1, 1, 1)[0]] + list(np.random.rand(1, 2)[0, :]),
                           # unary + [pairwise slack]
                           dtype=np.double, order='C')

    lossUnary_list = list()
    for ex in examples_list:
        lossUnary_list.append(mrf.augmented_loss(ex))

    for t in range(20):
        print("%s outer iter %d" % (batch_name, t))
        theta_old = theta

        if inf_latent_method == 'remove_redundancy':
            theta = mrf.remove_redundancy_theta(theta, options)

        latent_inferred_list = list()
        for ex in examples_list:
            latent_inferred_list.append(
                mrf.inf_latent_helper(ex, theta, options))

        vt_list = list()
        for ex, latent_inferred in zip(examples_list, latent_inferred_list):
            vt_list.append(mrf.phi_helper(ex, ex.y, latent_inferred, options))

        theta, inner_history = cutting_plane_ssvm(theta, vt_list, examples_list,
                                                  lossUnary_list, options, eng)

        outer_history.append({"inner_history": inner_history,
                              "latent_inferred_list": latent_inferred_list})

        with open('./expData/batchResult/temp/%s_outer_iter%d.pickle'
                          % (batch_name, t), 'wb') as f:
            pickle.dump(outer_history, f)

        sys.stdout.flush()

        if all(theta == theta_old):
            print('stop converge at iter: %d' % t)
            sys.stdout.flush()
            break

    eng.exit()

    return outer_history


if __name__ == '__main__':
    from MrfTypes import BatchExamplesParser
    from Utils.IOhelpers import _load_grabcut_unary_pairwise_cliques
    import time

    time_list = list()

    raw_example_list = _load_grabcut_unary_pairwise_cliques()
    parser = BatchExamplesParser()
    examples_list_all = parser.parse_grabcut_pickle(raw_example_list)
    options = Options()

    inf_latent_method = 'remove_redundancy'
    init_method = 'clique_by_clique'

    for i in range(50):
        time_list.append(time.time())
        examples_list = examples_list_all[:i] + examples_list_all[i + 1:]

        outer_history = cccp_outer_loop(examples_list, options, inf_latent_method, init_method)

        with open('./expData/batchResult/training_result/'
                  'image_%d_%s_outer_history.pickle' % (i, examples_list_all[i].name), 'wb') as f:
            pickle.dump([examples_list, outer_history, examples_list_all[i].name, time_list], f)
            # miu = 0
            # outer_history = cccp_outer_loop([examples_list_all[0]], options, inf_latent_method, init_method)
            # with open('./expData/batchResult/training_result/'
            #           'image%d_outer_history.pickle' % miu, 'wb') as f:
            #     pickle.dump([outer_history, examples_list_all[miu].name], f)
            # ex = examples_list_all[0]
            # theta = outer_history[-1]['inner_history'][-1]['theta']
            # y_hat,z_hat,e_hat = mrf.inf_label_latent_helper(ex.unary_observed,
            #                                     ex.pairwise,
            #                                     ex.clique_indexes,
            #                                     theta,options,ex.hasPairwise)
            # np.sum(y_hat!=ex.y)/ex.numVariables
            # plot_linfunc_converged('./hahaha',outer_history,options)
            # plot_colormap('./hehehe',outer_history,examples_list_all[0],options)
