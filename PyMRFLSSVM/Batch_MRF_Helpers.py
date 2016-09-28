# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
from CyInf.WllepGraphCut import Inf_Algo
from MrfTypes import Example, Options
import numpy as np


def phi_helper(ex, label_inferred, latent_inferred, options):
    '''

    :param ex:
    :type ex: Example
    :param options:
    :type options: Options
    :return:
    :rtype:
    '''
    # unary_observed = instance.unary_observed
    # pairwise = instance.pairwise
    # labels = instance.y
    # latent_var = instance.latent_var
    # clique_indexes = instance.clique_indexes. Note: clique id starts from 1
    #
    # options = Options()

    # unary phi
    unary_phi = sum(ex.unary_observed[:, :, 0].flatten()[label_inferred.flatten() == 0]) + \
                sum(ex.unary_observed[:, :, 1].flatten()[label_inferred.flatten() == 1])

    # pairwise phi
    pairwise_phi = 0
    if ex.hasPairwise:
        pairwise_phi = sum(label_inferred.flatten()[ex.pairwise[:, 0].astype(np.int)] !=
                           label_inferred.flatten()[ex.pairwise[:, 1].astype(np.int)])

    # higher order phi
    higher_order_phi = np.zeros(2 * options.K - 1, dtype=np.double, order='C')
    max_latent_index = [int(sum(latent_inferred[i, :])) for i in range(ex.numCliques)]

    # clique index starts from 1
    cliques_size = [sum(sum(ex.clique_indexes == i + 1)) for i in range(ex.numCliques)]
    cliques_value = [sum(label_inferred.flatten()[ex.clique_indexes.flatten() == i + 1]) /
                     cliques_size[i] for i in range(ex.numCliques)]

    higher_order_phi[0] = sum(cliques_value)
    # 1 < i < K
    for i in range(ex.numCliques):
        # if max_latent_index[i] = 0 < 1
        # then higher_order_phi[1:max_latent_index[i]] returns empty
        higher_order_phi[1:max_latent_index[i] + 1] += cliques_value[i]

    # sum of [[ i-K <= k^* ]] by clique_index
    # where k^* is the max_latent_index
    db_z = np.sum(latent_inferred, axis=0)

    # K <= i < 2K - 1
    for i in range(options.K, 2 * options.K - 1):
        higher_order_phi[i] = db_z[i - options.K]

    phi = np.zeros(options.sizePhi, dtype=np.double, order='C')
    phi[:options.sizeHighPhi] = higher_order_phi
    phi[options.sizeHighPhi] = unary_phi
    phi[options.sizeHighPhi + 1] = pairwise_phi

    return phi


def inf_latent_helper(ex, theta_full, options):
    '''

    :param ex:
    :type ex: Example
    :param theta_full:
    :type theta_full:
    :param options:
    :type options: Options
    :return:
    :rtype:
    '''
    # np.double[:] theta_full contains unary & pairwise params
    # Inf_Algo only accepts higher-order params

    # # code for debugging
    # theta_full = theta
    # clique_indexes = instance.clique_indexes
    # labels = instance.y

    theta = theta_full[:options.sizeHighPhi]

    cliques_size = [sum(sum(ex.clique_indexes == i + 1)) for i in range(ex.numCliques)]  # clique index starts
    # from 1
    cliques_value = [sum(ex.y.flatten()[ex.clique_indexes.flatten() == i + 1]) /
                     cliques_size[i] for i in range(ex.numCliques)]

    # # code for debugging
    # cliques_unary_value = [sum(instance.unary_observed[:, :, 1].
    #                            flatten()[clique_indexes.flatten() == i + 1]) / cliques_size[i]
    #                        for i in range(options.numCliques)]
    # a = np.reshape(cliques_value, [8, 8])
    # b = np.reshape(cliques_unary_value, [8, 8])

    inferred_latent = np.zeros([ex.numCliques, options.K - 1], dtype=np.int32, order='C')
    for i in range(ex.numCliques):
        for j in range(options.K - 1):
            # z_k = 1 only if (a_{k+1}-a_k)W_c(y_c) + b_{k+1}-b_k) < 0
            inferred_latent[i][j] = 1 if (theta[1 + j] * cliques_value[i] +
                                          theta[j + options.K] < 0) else 0

    return inferred_latent


class Old_Option:
    def __init__(self, rows, cols, numCliques,
                 K, hasPairwise, learningQP):
        self.H = rows
        self.W = cols
        self.numCliques = numCliques
        self.K = K
        self.hasPairwise = hasPairwise
        self.learningQP = learningQP


def inf_label_latent_helper(unary_observed, pairwise, clique_indexes, theta_full, options, hasPairwise=False):
    '''

    :param unary_observed:
    :type unary_observed:
    :param pairwise:
    :type pairwise:
    :param clique_indexes:
    :type clique_indexes:
    :param theta_full:
    :type theta_full:
    :param options:
    :type options: Options
    :return:
    :rtype:
    '''

    # unary_observed = u
    # pairwise = p
    # clique_indexes = c
    # theta_full = t
    # hasPairwise = True

    rows = unary_observed.shape[0]
    cols = unary_observed.shape[1]
    numCliques = len(np.unique(clique_indexes))

    # np.double[:] theta_full contains unary & pairwise params
    # Inf_Algo only accepts higher-order params
    theta = theta_full[:options.sizeHighPhi]

    # inferred_label & inferred_z are assigned inside Inf_Algo()
    inferred_label = np.zeros([rows, cols], dtype=np.int32, order='C')
    inferred_z = np.zeros([numCliques, options.K - 1], dtype=np.int32, order='C')

    old_option = Old_Option(rows, cols, numCliques,
                            options.K, hasPairwise, 1)

    # print("%d %d %d %d %d %d" % (old_option.H,
    #                              old_option.W,
    #                              old_option.numCliques,
    #                              old_option.K,
    #                              old_option.hasPairwise,
    #                              old_option.learningQP))

    e_i = Inf_Algo(unary_observed, pairwise, clique_indexes,
                   inferred_label, inferred_z, theta, old_option)

    return inferred_label, inferred_z, e_i


def init_theta_concave(example, options):
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

    :param example:
    :type example: Example
    :param options:
    :type options: Options
    :return:
    '''

    # clique index starts from 1
    cliques_size = [sum(sum(example.clique_indexes == i + 1)) for i in range(example.numCliques)]
    cliques_value = [sum(example.y.flatten()[example.clique_indexes.flatten() == i + 1]) /
                     cliques_size[i] for i in range(example.numCliques)]
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

    return np.asarray(theta, dtype=np.double, order='C')


def remove_redundancy_theta(theta, options, eps=1e-5):
    '''

    :param theta:
    :type theta: np.ndarray
    :param options:
    :type options:Options
    :param eps:
    :return:
    '''

    def intersect(a_1, b_1, a_2, b_2, func_idx, i):
        if a_1 - a_2 == 0:
            print(theta)
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
    active_inter_points_arr = np.asarray(active_inter_points_list)
    active_func_idxs = np.unique(active_func_idx_list)

    # if diff < eps between inter_points, let its (a_i,b_i) equal
    # (a_{i+1},b_{i+1})
    for i in reversed(range(1, len(active_inter_points_list))):
        a_2, b_2 = active_inter_points_arr[i, :]
        a_1, b_1 = active_inter_points_arr[i - 1, :]
        if (abs(a_2 - a_1) < eps) \
                and (abs(b_2 - b_1) < eps):
            active_func_idxs[i - 1] = active_func_idxs[i]

    active_func_idxs = np.unique(active_func_idxs)
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
