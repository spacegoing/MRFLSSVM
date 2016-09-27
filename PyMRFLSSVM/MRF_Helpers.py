# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
from CyInf.WllepGraphCut import Inf_Algo
import numpy as np


def phi_helper(unary_observed, pairwise, labels, latent_var, clique_indexes, options):
    # unary_observed = instance.unary_observed
    # pairwise = instance.pairwise
    # labels = instance.y
    # latent_var = instance.latent_var
    # clique_indexes = instance.clique_indexes. Note: clique id starts from 1
    #
    # options = Options()

    # unary phi
    unary_phi = sum(unary_observed[:, :, 0].flatten()[labels.flatten() == 0]) + \
                sum(unary_observed[:, :, 1].flatten()[labels.flatten() == 1])

    # pairwise phi
    pairwise_phi = 0
    if options.hasPairwise:
        pairwise_phi = sum(labels.flatten()[pairwise[:, 0].astype(np.int)] !=
                           labels.flatten()[pairwise[:, 1].astype(np.int)])

    # higher order phi
    higher_order_phi = np.zeros(2 * options.K - 1, dtype=np.double, order='C')
    max_latent_index = [int(sum(latent_var[i, :])) for i in range(options.numCliques)]

    # clique index starts from 1
    cliques_size = [sum(sum(clique_indexes == i + 1)) for i in range(options.numCliques)]
    cliques_value = [sum(labels.flatten()[clique_indexes.flatten() == i + 1]) /
                     cliques_size[i] for i in range(options.numCliques)]

    higher_order_phi[0] = sum(cliques_value)
    # 1 < i < K
    for i in range(options.numCliques):
        # if max_latent_index[i] = 0 < 1
        # then higher_order_phi[1:max_latent_index[i]] returns empty
        higher_order_phi[1:max_latent_index[i] + 1] += cliques_value[i]

    # sum of [[ i-K <= k^* ]] by clique_index
    # where k^* is the max_latent_index
    db_z = np.sum(latent_var, axis=0)

    # K <= i < 2K - 1
    for i in range(options.K, 2 * options.K - 1):
        higher_order_phi[i] = db_z[i - options.K]

    phi = np.zeros(options.sizePhi, dtype=np.double, order='C')
    phi[:options.sizeHighPhi] = higher_order_phi
    phi[options.sizeHighPhi] = unary_phi
    phi[options.sizeHighPhi + 1] = pairwise_phi

    return phi


def inf_latent_helper(labels, clique_indexes, theta_full, options):
    # np.double[:] theta_full contains unary & pairwise params
    # Inf_Algo only accepts higher-order params

    # # code for debugging
    # theta_full = theta
    # clique_indexes = instance.clique_indexes
    # labels = instance.y

    theta = theta_full[:options.sizeHighPhi]

    cliques_size = [sum(sum(clique_indexes == i + 1)) for i in range(options.numCliques)]  # clique index starts from 1
    cliques_value = [sum(labels.flatten()[clique_indexes.flatten() == i + 1]) /
                     cliques_size[i] for i in range(options.numCliques)]

    # # code for debugging
    # cliques_unary_value = [sum(instance.unary_observed[:, :, 1].
    #                            flatten()[clique_indexes.flatten() == i + 1]) / cliques_size[i]
    #                        for i in range(options.numCliques)]
    # a = np.reshape(cliques_value, [8, 8])
    # b = np.reshape(cliques_unary_value, [8, 8])

    inferred_latent = np.zeros([options.numCliques, options.K - 1], dtype=np.int32, order='C')
    for i in range(options.numCliques):
        for j in range(options.K - 1):
            # z_k = 1 only if (a_{k+1}-a_k)W_c(y_c) + b_{k+1}-b_k) < 0
            inferred_latent[i][j] = 1 if (theta[1 + j] * cliques_value[i] +
                                          theta[j + options.K] < 0) else 0

    return inferred_latent


def inf_label_latent_helper(unary_observed, pairwise, clique_indexes, theta_full, options):
    # np.double[:] theta_full contains unary & pairwise params
    # Inf_Algo only accepts higher-order params
    theta = theta_full[:options.sizeHighPhi]

    # inferred_label & inferred_z are assigned inside Inf_Algo()
    inferred_label = np.zeros([options.H, options.W], dtype=np.int32, order='C')
    inferred_z = np.zeros([options.numCliques, options.K - 1], dtype=np.int32, order='C')

    e_i = Inf_Algo(unary_observed, pairwise, clique_indexes, inferred_label, inferred_z, theta, options)

    return inferred_label, inferred_z, e_i


def init_theta_concave(instance, options):
    '''
    Initialize theta to encode a set of concave linear equations
    according to training data "instance".

    It first calculate W(y) of each cliques then determine how
    many different W(y) exists (namely linear equations needed).
    If options.K < desired number, this function will print a
    warning message then quit. User should increase options.K then
    run again.

    Then it sample a concave linear function equals the estimated
    (by W(y)) number of cliques. For extra linear functions
    (options.K> number of cliques) it simply initialize them to
    redundant functions.

    :param instance:
    :param options:
    :return:
    '''

    # clique index starts from 1
    cliques_size = [sum(sum(instance.clique_indexes == i + 1)) for i in range(options.numCliques)]
    cliques_value = [sum(instance.y.flatten()[instance.clique_indexes.flatten() == i + 1]) /
                     cliques_size[i] for i in range(options.numCliques)]
    unique_value_array = np.unique(cliques_value)

    # Check if Current K < potentially best number
    print("Potentially best K: %d" % unique_value_array.shape[0])
    if options.K < unique_value_array.shape[0]:
        print("Warning: Current K: %d < potentially best %d\n "
              "unique_value_array is shortened to fit options.K \n"
              "User may consider increase options.K"
              % (options.K, unique_value_array.shape[0]))
        shorten_indexes = [int(i) for i in
                           np.linspace(0, len(unique_value_array) - 1, options.K)]
        unique_value_array = unique_value_array[shorten_indexes]
        # print("Warning: Current K: %d < potentially best %d, please increase K then run again"
        #       % (options.K, unique_value_array.shape[0]))
        # raise ValueError("see warning info")

    # Mid points between unique values.
    mid_points_array = np.zeros([unique_value_array.shape[0] - 1])
    for i in range(1, unique_value_array.shape[0]):
        mid_points_array[i - 1] = (unique_value_array[i - 1] + unique_value_array[i]) / 2

    # sample a set of concave linear functions based on those points
    # initialize a_b parameters matrix (a,b) and sampled points matrix (x,y)
    a_b = np.zeros([options.K, 2])
    sampled_points = np.zeros([mid_points_array.shape[0] + 2, 2])
    sampled_points[1:mid_points_array.shape[0] + 1, 0] = mid_points_array
    sampled_points[mid_points_array.shape[0] + 1, 0] = 1
    if sampled_points.shape[0] < options.K + 1:
        redund_points = np.zeros([options.K + 1 - sampled_points.shape[0], 2])
        redund_points[:, 0] = np.linspace(1.1,
                                          1 + 0.1 * (options.K + 1 - sampled_points.shape[0]),
                                          options.K + 1 - sampled_points.shape[0])
        sampled_points = np.r_[sampled_points, redund_points]

    # Sample the first point
    sampled_points[1, 1] = np.random.uniform(sampled_points[1, 0], 1, 1)[0]
    a_b[0, 0] = (sampled_points[1, 1] - sampled_points[0, 1]) / \
                (sampled_points[1, 0] - sampled_points[0, 0])
    # Sample other points
    for i in range(1, options.K):
        up_bound = a_b[i - 1, 0] * sampled_points[i + 1, 0] + a_b[i - 1, 1] - 1e-9
        sampled_points[i + 1, 1] = np.random.uniform(up_bound - 0.5, up_bound, 1)[0]

        if (sampled_points[i + 1, 0] - sampled_points[i, 0]) != 0:
            a_b[i, 0] = (sampled_points[i + 1, 1] - sampled_points[i, 1]) / \
                        (sampled_points[i + 1, 0] - sampled_points[i, 0])
            a_b[i, 1] = sampled_points[i + 1, 1] - a_b[i, 0] * sampled_points[i + 1, 0]
        else:
            a_b[i, 0] = 0
            a_b[i, 1] = sampled_points[i + 1, 1]

    # encode a_b into theta
    theta = [a_b[0, 0]]
    # a_{k}-a{k-1}
    for i in range(1, options.K):
        theta.append(a_b[i, 0] - a_b[i - 1, 0])
    # b{k}-b{k-1}
    for i in range(1, options.K):
        theta.append(a_b[i, 1] - a_b[i - 1, 1])
    # unary, pairwise and slack
    theta += [np.random.uniform(-1, 1, 1)[0]] + list(np.random.rand(1, 2)[0, :])

    return theta


def remove_redundancy_theta(theta, options, eps=1e-14):
    '''

    :param theta:
    :param options:
    :param eps:
    :return:
    '''

    def intersect(a_1, b_1, a_2, b_2, func_idx, i):
        if a_1 - a_2 == 0:
            raise ValueError('Intersection Equals 0!\ntheta: %d and %d' % (func_idx, i))
        x = (b_2 - b_1) / (a_1 - a_2)
        y = (a_1 * b_2 - a_2 * b_1) / (a_1 - a_2)
        # Can't exceed 1
        if x > 1:
            x = 1
            y = a_1 + b_1
        return x, y

    # decode theta
    a_b_array = np.zeros([options.K, 2])
    a_b_array[0, 0] = theta[0]
    for i in range(1, options.K):
        a_b_array[i, 0] = theta[i] + a_b_array[i - 1, 0]
        a_b_array[i, 1] = theta[i + options.K - 1] + a_b_array[i - 1, 1]

    active_inter_points_list = [[0, 0]]
    active_func_idx_list = [0]
    func_idx = 0  # next index of active function
    while func_idx < options.K - 1:
        inter_points = list()
        a_1 = a_b_array[func_idx, 0]
        b_1 = a_b_array[func_idx, 1]
        # generate intersection points between i and all the other
        for i in range(func_idx + 1, options.K):
            a_2 = a_b_array[i, 0]
            b_2 = a_b_array[i, 1]
            inter_points.append(intersect(a_1, b_1, a_2, b_2, func_idx, i))
        inter_points = np.asarray(inter_points)

        # Which function is lower (inter point nearest to original point)
        active_inter_point_idx = np.argmin(inter_points[:, 0])
        active_point = inter_points[active_inter_point_idx, :]
        active_inter_points_list.append(active_point)
        if active_point[0] == 1:  # if x already exceeds 1, others are redund
            active_func_idx = active_inter_point_idx + func_idx
            active_func_idx_list.append(active_func_idx)
            break
        else:
            # else the next active function is the following:
            active_func_idx = active_inter_point_idx + func_idx + 1
            active_func_idx_list.append(active_func_idx)
            func_idx = active_func_idx

    # np.unique return values from smallest to largest
    active_func_idxs = np.unique(active_func_idx_list)
    active_func_no = active_func_idxs.shape[0]

    # Remove redundancies (Otherwise there will be 0s between 1s in
    # latent_inferred like: [1 0 1 0 0 0 0...])
    # todo: direct calc theta to improve performance
    a_b_array[:active_func_no, :] = a_b_array[active_func_idxs, :]
    a_b_array[active_func_no:, :] = a_b_array[active_func_idxs[-1], :]

    for i in range(1, options.K):
        theta[i] = a_b_array[i, 0] - a_b_array[i - 1, 0]
        theta[i + options.K - 1] = a_b_array[i, 1] - a_b_array[i - 1, 1]

    return theta


if __name__ == '__main__':
    from Checkboard import Instance, Options

    instance = Instance()
    unary_observed = instance.unary_observed
    pairwise = instance.pairwise
    labels = instance.y
    latent_var = instance.latent_var
    clique_indexes = instance.clique_indexes  # Note: clique id starts from 1

    options = Options()

    phi = phi_helper(unary_observed, pairwise, labels, latent_var, clique_indexes, options)

    theta_full = np.zeros(options.sizePhi, dtype=np.double, order='C')
    inferred_label, inferred_z, e_i = \
        inf_label_latent_helper(unary_observed, pairwise, clique_indexes, theta_full, options)
    inferred_latent = inf_latent_helper(labels, clique_indexes, theta_full, options)

    # Dev slack inf
    from Utils.IOhelpers import load_pickle

    outer_history, instance, options = \
        load_pickle("/Users/spacegoing/macCodeLab-MBP2015/"
                    "Python/MRFLSVM/PyMRFLSSVM/expData/"
                    "unbalaced_portions/more_black_3339active")
    theta = outer_history[-1]['inner_history'][-1]['theta']
    latent_inferred = inf_latent_helper(instance.y, instance.clique_indexes, theta, options)
    for i in latent_inferred:
        print(i)
    print('\n############Remove#############\n')
    theta = remove_redundancy_theta(theta, options)
    latent_inferred = inf_latent_helper(instance.y, instance.clique_indexes, theta, options)
    for i in latent_inferred:
        print(i)
