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

    cliques_size = [sum(sum(clique_indexes == i + 1)) for i in range(options.numCliques)]  # clique index starts from 1
    cliques_value = [sum(labels.flatten()[clique_indexes.flatten() == i + 1]) / cliques_size[i] for i in range(int(
        options.numCliques))]

    higher_order_phi[0] = sum(cliques_value)
    # 1 < i < K
    for i in range(options.numCliques):
        # if max_latent_index[i] = 0 < 1
        # then higher_order_phi[1:max_latent_index[i]] returns empty
        higher_order_phi[1:max_latent_index[i]] += cliques_value[i]

    # sum of [[ i-K <= k^* ]] by cliqueID
    # where k^* is the max_latent_index
    db_z = np.sum(latent_var, axis=0)
    # K <= i < 2K - 1
    for i in range(options.K, 2 * options.K - 1):
        higher_order_phi[i] = db_z[i - options.K]

    phi = np.zeros(2 * options.K + 1, dtype=np.double, order='C')
    phi[:2 * options.K - 1] = higher_order_phi
    phi[2 * options.K - 1] = unary_phi
    phi[2 * options.K] = pairwise_phi

    return phi

# def inf_helper(observed_unary, pairwise, clique_indexes, theta, options):
#     inferred_label=
#     inferred_z=

if __name__ == '__main__':
    from Checkboard import Instance, Options

    instance = Instance()
    unary_observed = instance.unary_observed
    pairwise = instance.pairwise
    labels = instance.y
    latent_var = instance.latent_var
    cliques_indexes = instance.clique_indexes  # Note: clique id starts from 1

    options = Options()

    phi = phi_helper(unary_observed, pairwise, labels, latent_var, cliques_indexes, options)
