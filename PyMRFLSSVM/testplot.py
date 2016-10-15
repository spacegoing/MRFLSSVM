# -*- coding: utf-8 -*-
from Batch_MRF_Helpers import inf_label_latent_helper
from MrfTypes import BatchExamplesParser, Options
from Utils.IOhelpers import _load_grabcut_unary_pairwise_cliques
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

__author__ = 'spacegoing'

path = './expData/batchResult/temp/bush.pickle_outer_iter0.pickle'

theta_list = list()
loss_list = list()
with open(path, 'rb') as f:
    data = pickle.load(f)
    inner_history = data[-1]['inner_history']
    for i in inner_history:
        theta_list.append(i['theta'])
        loss_list.append(i['loss_aug'])

batches_num = 25
raw_example_list = _load_grabcut_unary_pairwise_cliques()
parser = BatchExamplesParser()
examples_list_all = parser.parse_grabcut_pickle(raw_example_list)
options = Options()

inf_latent_method = ''
init_method = 'clique_by_clique'

Tasks = list()
# for i in range(batches_num):
i = 25
examples_list = examples_list_all[:i] + examples_list_all[i + 1:]

y_hat_loss = list()
for theta in theta_list:
    error = 0
    for ex in examples_list:
        y_hat = inf_label_latent_helper(
            ex.unary_observed, ex.pairwise,
            ex.clique_indexes, theta, options, ex.hasPairwise)[0]
        error += np.sum(y_hat != ex.y)
    y_hat_loss.append(error/len(examples_list))

for i in y_hat_loss:
    print('%.16f'%i,end=',\n')

no=20
plt.plot(sp.linspace(0, 100, len(y_hat_loss[:no])), y_hat_loss[:no])