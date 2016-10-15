# -*- coding: utf-8 -*-
import os
import numpy as np
from MrfTypes import BatchExamplesParser, Options
from Utils.IOhelpers import _load_grabcut_unary_pairwise_cliques, \
    _load_grabcut_train_results
from Batch_MRF_Helpers import inf_label_latent_helper
import pickle

__author__ = 'spacegoing'

options = Options()

######################## Load training results ##########################
train_results_path = "/Users/spacegoing/macCodeLab-MBP2015/" \
                     "Python/GrabCutPickleBatchResults/"
leave_out_train_results_dict = _load_grabcut_train_results(train_results_path)

######################## Load Examples List ##########################
raw_example_list = _load_grabcut_unary_pairwise_cliques()
parser = BatchExamplesParser()
examples_list_all = parser.parse_grabcut_pickle(raw_example_list)
# read list into dict
examples_name_ex_dict = dict()
for ex in examples_list_all:
    examples_name_ex_dict[ex.name.split('.')[0]] = ex

######################## Match training results to examples ##########
name_theta_example_dict = dict()
for name, theta_outerhist in leave_out_train_results_dict.items():
    name_theta_example_dict[name] = {'theta': theta_outerhist['theta'],
                                     'ex': examples_name_ex_dict[name]}


def test_results(name_theta_example_dict):
    '''

    :param name_theta_example_dict:
    :type name_theta_example_dict: dict
    :return:
    :rtype:
    '''

    name_infLabel_loss_y_dict = dict()
    for name, theta_ex_dict in name_theta_example_dict.items():
        theta = theta_ex_dict['theta']
        ex = theta_ex_dict['ex']
        inferred_label = \
            inf_label_latent_helper(ex.unary_observed, ex.pairwise,
                                    ex.clique_indexes, theta, options, ex.hasPairwise)[0]
        loss = np.sum(inferred_label != ex.y) / (ex.y.shape[0] * ex.y.shape[1])
        name_infLabel_loss_y_dict[name] = {'inferred_label': inferred_label,
                                           'loss': loss,
                                           'y': ex.y}
    return name_infLabel_loss_y_dict


if __name__ == '__main__':
    name_infLabel_loss_y_dict = test_results(name_theta_example_dict)

    stats_data_dir = './StatsData/'
    with open(stats_data_dir + '50_results.pickle', 'wb') as f:
        pickle.dump(name_infLabel_loss_y_dict, f)

    with open(stats_data_dir + '50_results.pickle', 'rb') as f:
        name_infLabel_loss_y_dict = pickle.load(f)

    avg_error = 0
    worst_loss_list = list()
    worst_name_list = list()
    for name, i in name_infLabel_loss_y_dict.items():
        print(i['loss'])
        if i['loss']>0.05:
            worst_loss_list.append(i['loss'])
            worst_name_list.append(str(name))
        else:
            avg_error += i['loss']

    print(1-avg_error / len(name_infLabel_loss_y_dict))
