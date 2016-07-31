# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

from CyInf.GraphCy import GraphCy
import numpy as np


def linEnvInf(unaryWeights, pairWeights, envCoeffs, cliques):
    # unaryWeights = params.unaryWeight * instance.unary  # n x 2
    # pairWeights = instance.pairwise  #
    # envCoeffs = params.linEnvCoeffs  # K * 2
    # cliques = instance.cliques  # n x 1

    nVariables = unaryWeights.shape[0]
    K = envCoeffs.shape[0]
    nMaxCliquesPerVariable = 1  # prhs3.shape[1]

    g = GraphCy(nVariables, 8 * nVariables)
    g.add_node(nVariables)

    # Add unary terms
    for i, u in enumerate(unaryWeights):
        g.add_tweights(i, u[0], u[1])

    # Add pairwise terms
    if pairWeights.size == 0:  # not empty
        for p in pairWeights:
            u = p[0] - 1
            v = p[1] - 1
            w = p[2]

            assert w >= 0.0, "illegal pairwise weight"
            if w == 0.0:
                continue

            g.add_edge(u, v, w, w)

    # add auxiliary variables for each clique
    cliqueIDs = np.unique(cliques)
    z = list()

    if (K > 1):
        for i in cliqueIDs:
            z.append(g.add_node(K - 1))

    # add higher - order term for each clique
    # sets w_i = 1 / cliqueSize

    for i, c in enumerate(cliques):
        if nMaxCliquesPerVariable > 1:
            # one node belongs to more than 1 cliques
            for cc in c:
                w_i = 1.0 / np.sum(cliques == cc)
                a = envCoeffs[0]
                g.add_tweights(i, 0.0, a * w_i)

                for k in range(1, K):
                    da = envCoeffs[k - 1, 0] - envCoeffs[k, 0]
                    if (np.fabs(da) < 1.0e-6):
                        continue
                    assert da > 0, "invalid linear envelope coefficients"
                    g.add_edge(i, z[int(cc - 1)] + k - 1, 0.0, w_i * da)
                    # use cc here because the index of clique cc = cc-1 in z

        else:
            w_i = 1.0 / np.sum(cliques == c)
            a = envCoeffs[0, 0]
            g.add_tweights(i, 0.0, a * w_i)

            for k in range(1, K):
                da = envCoeffs[k - 1, 0] - envCoeffs[k, 0]
                if np.fabs(da) < 1.0e-6:
                    continue
                assert da > 0, "invalid linear envelope coefficients"
                g.add_edge(i, z[int(c - 1)] + k - 1, 0.0, w_i * da)

    # add edges between s and z_k and z_k and t
    for c in cliqueIDs:
        for k in range(1, K):
            da = envCoeffs[k - 1, 0] - envCoeffs[k, 0]
            if np.fabs(da) < 1.0e-6:
                continue
            db = envCoeffs[k, 1] - envCoeffs[k - 1, 1]
            assert db > 0, "invalid linear envelope coefficients"
            g.add_tweights(z[int(c - 1)] + k - 1, da, db)

    e = g.maxflow()

    y_hat = np.zeros(nVariables)

    for i in range(nVariables):
        y_hat[i] = g.what_segment(i)

    z_hat = np.zeros(len(cliqueIDs) * (K-1))
    for i,hi in enumerate(z):
        z_hat[i] = g.what_segment(hi)

    #todo: debug
    z_hat.reshape([64,9])
    print(z_hat)

    return y_hat, z_hat, e
