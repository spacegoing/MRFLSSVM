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
        higher_order_phi[1:max_latent_index[i]+1] += cliques_value[i]

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
    theta = theta_full[:options.sizeHighPhi]

    cliques_size = [sum(sum(clique_indexes == i + 1)) for i in range(options.numCliques)]  # clique index starts from 1
    cliques_value = [sum(labels.flatten()[clique_indexes.flatten() == i + 1]) /
                     cliques_size[i] for i in range(options.numCliques)]

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
