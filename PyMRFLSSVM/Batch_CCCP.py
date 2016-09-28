# -*- coding: utf-8 -*-
import numpy as np
import matlab.engine
import Batch_MRF_Helpers as mrf
from MrfTypes import Example, Options
from Utils.ReadMat import loadTestInf, loadMatPairwise
from Utils.IOhelpers import dump_pickle

__author__ = 'spacegoing'

eng = matlab.engine.start_matlab()
__DEBUG__ = 0
__plot__ = 1


def quadprog_matlab(P, q, A, b):
    P, q, A, b = [matlab.double(i.tolist())
                  for i in [P, q, A, b]]
    null = matlab.double([])
    theta = eng.quadprog(P, q, A, b, null, null, null, null)

    return np.array(theta, dtype=np.double, order='C').reshape(len(theta))


def cutting_plane_ssvm(theta, vt_list, examples_list, lossUnary_list, options):
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
    :rtype:
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

        # Decode parameters
        unaryWeight = theta[options.sizeHighPhi]
        pairwiseWeight = max(0, theta[options.sizeHighPhi + 1])

        # todo: log data
        y_hat, z_hat, e_hat = \
            mrf.inf_label_latent_helper(unaryWeight * examples_list[0].unary_observed,
                                        examples_list[0].pairwise,
                                        examples_list[0].clique_indexes,
                                        theta, options, examples_list[0].hasPairwise)
        history.append({'theta': theta, 'y_hat': y_hat})

        loss_arr = np.zeros(examples_num)
        phi_loss_sum = np.zeros(options.sizePhi, dtype=np.double, order='C')
        violation_sum = 0.0
        for vt, ex, lossUnary, m in zip(vt_list, examples_list,
                                        lossUnary_list, range(examples_num)):
            print(m)
            # todo: Errors: Pairwise features
            # todo: Errors: Unary features
            if ex.hasPairwise:
                ex.pairwise[:, 2] = pairwiseWeight

            # infer most violated constraint
            y_loss, z_loss, e_loss = \
                mrf.inf_label_latent_helper(unaryWeight * ex.unary_observed - lossUnary,
                                            ex.pairwise, ex.clique_indexes,
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

        theta = quadprog_matlab(P, q, -A, -b)

    return theta, history


def cccp_outer_loop(examples_list, options, init_method='', inf_latent_method=''):
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

    for t in range(10):
        theta_old = theta

        if inf_latent_method == 'remove_redund':
            theta = mrf.remove_redundancy_theta(theta, options)

        latent_inferred_list = list()
        for ex in examples_list:
            latent_inferred_list.append(
                mrf.inf_latent_helper(ex, theta, options))

        vt_list = list()
        for ex, latent_inferred in zip(examples_list, latent_inferred_list):
            vt_list.append(mrf.phi_helper(ex, ex.y, latent_inferred, options))

        theta, inner_history = cutting_plane_ssvm(theta, vt_list, examples_list,
                                                  lossUnary_list, options)

        total_error = np.sum(np.abs(examples_list[0].y - inner_history[-1]['y_hat']))
        outer_history.append({"inner_history": inner_history,
                              "latent_inferred": latent_inferred_list[0],
                              "total_error": total_error})

        if all(theta == theta_old):
            print(latent_inferred_list[0])
            print('stop converge at iter: %d' % t)
            print('classification error: %d' % total_error)
            break

        if np.sum(np.abs(examples_list[0].y - inner_history[-1]['y_hat'])) / examples_list[0].numVariables < \
                options.eps:
            print(latent_inferred_list[0])
            print('converge at iter: %d' % t)
            print('classification error: %d' % total_error)
            break

    return outer_history


if __name__ == '__main__':
    import numpy as np
    import matlab.engine
    import Batch_MRF_Helpers as mrf
    from MrfTypes import Example, Options
    from Utils.ReadMat import loadTestInf, loadMatPairwise
    from Utils.IOhelpers import dump_pickle

    __author__ = 'spacegoing'

    eng = matlab.engine.start_matlab()
    __DEBUG__ = 0
    __plot__ = 1


    def quadprog_matlab(P, q, A, b):
        P, q, A, b = [matlab.double(i.tolist())
                      for i in [P, q, A, b]]
        null = matlab.double([])
        theta = eng.quadprog(P, q, A, b, null, null, null, null)

        return np.array(theta, dtype=np.double, order='C').reshape(len(theta))


    from Utils.IOhelpers import _load_grabcut_unary_pairwise_cliques
    from MrfTypes import BatchExamplesParser

    raw_example_list = _load_grabcut_unary_pairwise_cliques()
    parser = BatchExamplesParser()
    examples_list = parser.parse_grabcut_pickle(raw_example_list)
    options = Options

    inf_latent_method = 'remove_redund'
    init_method = 'clique_by_clique'

    examples_list = [examples_list[0]]

    # outer_history = cccp_outer_loop([examples_list[0]], options, inf_latent_method, init_method)

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

    import pickle

    with open('theta_test.pickle', 'wb') as f:
        pickle.dump(theta, f)
    for i in theta:
        print(i, end=", ")
    print()

    lossUnary_list = list()
    for ex in examples_list:
        lossUnary_list.append(mrf.augmented_loss(ex))

    theta_old = theta

    if inf_latent_method == 'remove_redund':
        theta = mrf.remove_redundancy_theta(theta, options)

    print('latent')
    latent_inferred_list = list()
    for ex in examples_list:
        latent_inferred_list.append(
            mrf.inf_latent_helper(ex, theta, options))
    print('vt')
    vt_list = list()
    for ex, latent_inferred in zip(examples_list, latent_inferred_list):
        vt_list.append(mrf.phi_helper(ex, ex.y, latent_inferred, options))
    print('vt done')

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
        # Decode parameters
        unaryWeight = theta[options.sizeHighPhi]
        pairwiseWeight = max(0, theta[options.sizeHighPhi + 1])

        # todo: log data
        # if options.log_history:
        #     y_hat, z_hat, e_hat = \
        #         mrf.inf_label_latent_helper(unaryWeight * instance.unary_observed, instance.pairwise,
        #                                     instance.clique_indexes, theta, options)
        #     history.append({'theta': theta, 'y_hat': y_hat, 'z_hat': z_hat, 'e_hat': e_hat})

        loss_arr = np.zeros(examples_num)
        phi_loss_sum = np.zeros(options.sizePhi, dtype=np.double, order='C')
        violation_sum = 0.0
        vt = vt_list[0]
        ex = examples_list[0]
        lossUnary = lossUnary_list[0]
        m = 0

        # for vt, ex, lossUnary, m in zip(vt_list, examples_list,
        #                                 lossUnary_list, range(examples_num)):
        print(m)
        # todo: Errors: Pairwise features
        if ex.hasPairwise:
            ex.pairwise[:, 2] = pairwiseWeight

        u = unaryWeight * ex.unary_observed - lossUnary
        p = ex.pairwise
        c = ex.clique_indexes
        t = theta
        with open('ex_test.pickle', 'wb') as f:
            pickle.dump([u, p, c, t],f)

        # infer most violated constraint
        y_loss, z_loss, e_loss = \
            mrf.inf_label_latent_helper(unaryWeight * ex.unary_observed - lossUnary,
                                        ex.pairwise, ex.clique_indexes,
                                        theta, options, ex.hasPairwise)
        with open('loss_test.pickle', 'wb') as f:
            pickle.dump([y_loss, z_loss, e_loss],f)

        # add constraint
        phi = mrf.phi_helper(ex, y_loss, z_loss, options)
        phi_loss_sum += phi - vt

        loss_arr[m] = np.sum(y_loss != ex.y) / ex.numVariables
        slack = loss_arr[m] - np.dot((phi - vt), theta[:-1])
        violation_sum += slack - theta[-1]

        if violation_sum / examples_num < options.eps:
            print("violation_sum / examples_num < options.eps")

        loss = np.sum(loss_arr) / examples_num
        phi_loss = phi_loss_sum / examples_num
        A = np.r_[A, [np.r_[phi_loss, 1]]]
        b = np.r_[b, loss]
        #
        theta = quadprog_matlab(P, q, -A, -b)
