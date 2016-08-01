# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

from CyInf.GraphCy import GraphCy
from CyInf.WllepGraphCut import Inf_Algo
import numpy as np
from Utils.ReadMat import loadTestInf
from main import instance


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

class Options:
    H = 128
    W = 128
    numCliques = 64
    K = 10
    learningQP = 1


unary_observed, pairwise_raw, w_raw, y_inferred, z_inferred, e = loadTestInf()
options = Options()
# float[:, :, :]
# observed_unary, float[:, :]
# pairwise,
# int[:, :]
# clique_indexes,
# int[:, :]
# inferred_label, int[:, :]
# inferred_z,
# double[:]
# w, options
observed_unary = np.zeros([128,128,2],dtype=np.float32)
for i in range(128):
    for j in range(128):
        observed_unary[i][j][0]=unary_observed[i][j][0]
        observed_unary[i][j][1]=unary_observed[i][j][1]

pairwise=np.zeros(pairwise_raw.shape,order='C',dtype=np.float32)
for i in range(pairwise_raw.shape[0]):
    for j in range(3):
        pairwise[i][j] = pairwise_raw[i][j]

indexes = instance.cliques.reshape([128,128],order='C').astype(np.int32).transpose()
clique_indexes = np.zeros([128,128],order='C',dtype=np.int32)
for i in range(128):
    for j in range(128):
        clique_indexes[i][j] = indexes[i][j]

inferred_label = np.zeros(128*128,dtype=np.int32)
inferred_z = np.zeros(64*9,dtype=np.int32)
w=np.zeros(1+2*9,dtype=np.double)
w[0]=w_raw[0,0]
for i in range(1,10):
    w[i] = w_raw[i,0]-w_raw[i-1,0]
    w[i+9] = w_raw[i,1] -w_raw[i-1,1]

e_i = Inf_Algo(observed_unary, pairwise, clique_indexes, inferred_label, inferred_z, w, options)

inferred_label = inferred_label.reshape([128,128])
inferred_z = inferred_z.reshape([64,9])
for i in range(128):
    for j in range(128):
        if inferred_label[i][j] != y_inferred[i][j]:
            print(str(i)+' '+str(j))

for i in range(64):
    for j in range(9):
        if inferred_z[i][j] != z_inferred.T[i][j]:
            print(str(i) + str(j))
# z_inferred.T
# inferred_z
# for i in range(128):
#     for j in range(128):
