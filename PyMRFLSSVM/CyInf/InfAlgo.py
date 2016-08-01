# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

from CyInf.WllepGraphCut import Inf_Algo
import numpy as np

from Utils.ReadMat import loadTestInf
from main import instance


class Options:
    H = 128
    W = 128
    numCliques = 64
    K = 10
    learningQP = 1

def linEnvInf(unaryWeights, pairWeights, envCoeffs, cliques):
    # unaryWeights = params.unaryWeight * instance.unary  # n x 2
    # pairWeights = instance.pairwise  #
    # envCoeffs = params.linEnvCoeffs  # K * 2
    # cliques = instance.cliques  # n x 1



    return y_hat, z_hat, e




unary_observed, pairwise_raw, w_raw, y_inferred, z_inferred, e = loadTestInf()
options = Options()
# float[:, :, :] observed_unary, float[:, :] pairwise,
# int[:, :] clique_indexes,
# int[:, :] inferred_label, int[:, :] inferred_z,
# double[:] w, options
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

inferred_label = np.zeros([128,128],dtype=np.int32)
inferred_z = np.zeros([64,9],dtype=np.int32)
w=np.zeros(1+2*9,dtype=np.double)
w[0]=w_raw[0,0]
for i in range(1,10):
    w[i] = w_raw[i,0]-w_raw[i-1,0]
    w[i+9] = w_raw[i,1] -w_raw[i-1,1]

e_i = Inf_Algo(observed_unary, pairwise, clique_indexes, inferred_label, inferred_z, w, options)

# for i in range(128):
#     for j in range(128):
#         if inferred_label[i][j] != y_inferred[i][j]:
#             print(str(i)+' '+str(j))
#
# for i in range(64):
#     for j in range(9):
#         if inferred_z[i][j] != z_inferred.T[i][j]:
#             print(str(i) + str(j))
