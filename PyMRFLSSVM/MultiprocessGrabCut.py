# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')
from Batch_CCCP import cccp_outer_loop
from Checkboard import Instance
from MrfTypes import Example, Options, BatchExamplesParser
from Utils.IOhelpers import _load_grabcut_unary_pairwise_cliques
from Batch_MRF_Helpers import inf_label_latent_helper
import multiprocessing
import numpy as np
import sys
import pickle

__author__ = 'spacegoing'

inf_latent_method = ''
init_method = 'clique_by_clique'

parser = BatchExamplesParser()
root = './expData/unbalaced_portions/'


def calcFun(args):
    func, arg = args
    result = func(*arg)
    return result


def calcCheckboard(prefix_str, miu):
    instance = Instance('gaussian_portions', portion_miu=miu, is_gaussian=False)
    examples_list = parser.parse_checkboard(instance)

    options = Options()
    outer_history = cccp_outer_loop([examples_list[0]], options, init_method,
                                    inf_latent_method, prefix_str)

    with open('./expData/batchResult/training_result/'
              'image_%s_outer_history.pickle' % prefix_str, 'wb') as f:
        pickle.dump([outer_history, examples_list, prefix_str], f)

    sys.stdout.flush()


def calcGrabCut(examples_list, leave_out_name, inf_latent_method, init_method, options):
    outer_history = cccp_outer_loop(examples_list, options, inf_latent_method, init_method, leave_out_name)

    with open('./expData/batchResult/training_result/'
              'leaveout_image_%s_outer_history.pickle' % leave_out_name, 'wb') as f:
        pickle.dump([examples_list, outer_history, leave_out_name], f)


def multip_checkboard():
    data_list = [["more_black_3339", (0.3, 0.3, 0.3, 0.9)],
                 ["more_white_1777", (0.1, 0.7, 0.7, 0.7)],
                 ["balanced_portions_124678", (0.1, 0.2, 0.4, 0.6, 0.7, 0.8)]]
    Tasks = [(calcCheckboard, miu) for miu in data_list]
    pool = multiprocessing.Pool(3)
    r = pool.map(calcFun, Tasks)
    pool.close()
    pool.join()
    print(r)
    print('finished')


def multip_grabCut():
    raw_example_list = _load_grabcut_unary_pairwise_cliques()
    parser = BatchExamplesParser()
    selected_num = 50
    examples_list_all = parser.parse_grabcut_pickle(raw_example_list)
    options = Options()

    batches_num = len(examples_list_all)

    inf_latent_method = ''
    init_method = 'clique_by_clique'

    Tasks = list()
    for i in range(selected_num):
        examples_list = examples_list_all[:i] + examples_list_all[i + 1:]
        Tasks.append((calcGrabCut, (examples_list, examples_list_all[i].name,
                                    inf_latent_method, init_method, options)))

    process_num = min(selected_num, 20)
    pool = multiprocessing.Pool(process_num)
    r = pool.map(calcFun, Tasks)
    pool.close()
    pool.join()
    print(r)
    print('finished')


def laipiDog(examples_list, leave_out_name, inf_latent_method, init_method, options, ex_test):
    '''

    :param ex_test:
    :type ex_test: Example
    :return:
    :rtype:
    '''

    count = 0
    loss = 1.0
    outer_history = list()
    while loss > 0.5:
        outer_history = cccp_outer_loop(examples_list, options, inf_latent_method, init_method, leave_out_name)
        theta = outer_history[-1]['theta']
        y_hat = inf_label_latent_helper(ex_test.unary_observed, ex_test.pairwise,
                                        ex_test.clique_indexes, theta, options, ex_test.hasPairwise)[0]
        loss = np.sum(y_hat != ex_test.y) / (ex_test.y.shape[0] * ex_test.y.shape[1])
        count += 1

        with open('./expData/batchResult/training_result/'
                  'laipi_leaveout_image_%s_%d_%f_outer_history.pickle' %
                          (leave_out_name, count, float(1-loss)), 'wb') as f:
            pickle.dump([ex_test, outer_history, leave_out_name], f)


    with open('./expData/batchResult/training_result/'
              'laipichenggong_leaveout_image_%s_%d_outer_history.pickle' % (leave_out_name, count), 'wb') as f:
        pickle.dump([ex_test, outer_history, leave_out_name], f)


def laipi_grabCut():
    raw_example_list = _load_grabcut_unary_pairwise_cliques()
    parser = BatchExamplesParser()
    selected_num = 50
    examples_list_all = parser.parse_grabcut_pickle(raw_example_list)
    options = Options()

    batches_num = len(examples_list_all)

    inf_latent_method = 'remove_redundancy'
    init_method = 'clique_by_clique'

    laipi_name = ['tennis', 'person6', 'cross', '65019', 'elefant',
                  'bool', '37073', 'banana1', '153077', 'book', '153093', '24077',
                  '189080', '69020']

    Tasks = list()
    for i in range(selected_num):
        ex_test = examples_list_all[i]
        if ex_test.name.split('.')[0] in laipi_name:
            examples_list = examples_list_all[:i] + examples_list_all[i + 1:]
            Tasks.append((laipiDog, (examples_list, examples_list_all[i].name,
                                        inf_latent_method, init_method, options, ex_test)))
    if len(Tasks) != len(laipi_name):
        raise ValueError('Laipi doesn\'t match!')

    pool = multiprocessing.Pool(len(laipi_name))
    r = pool.map(calcFun, Tasks)
    pool.close()
    pool.join()
    print(r)
    print('laipi_finished')


if __name__ == "__main__":
    multiprocessing.freeze_support()

    import time

    time_be = time.time()

    laipi_grabCut()

    time_end = time.time()
    m, s = divmod(time_end - time_be, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))
