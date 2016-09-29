# -*- coding: utf-8 -*-
from Batch_CCCP import cccp_outer_loop
from Checkboard import Instance
from MrfTypes import Example, Options, BatchExamplesParser
from Utils.IOhelpers import _load_grabcut_unary_pairwise_cliques
import multiprocessing
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
    batches_num = 5
    raw_example_list = _load_grabcut_unary_pairwise_cliques()
    parser = BatchExamplesParser()
    examples_list_all = parser.parse_grabcut_pickle(raw_example_list)[:batches_num]
    options = Options()

    inf_latent_method = ''
    init_method = 'clique_by_clique'

    Tasks = list()
    for i in range(batches_num):
        examples_list = examples_list_all[:i] + examples_list_all[i + 1:]
        Tasks.append((calcGrabCut, (examples_list, examples_list_all[i].name,
                                    inf_latent_method, init_method, options)))

    process_num = min(batches_num,28)
    pool = multiprocessing.Pool(process_num)
    r = pool.map(calcFun, Tasks)
    pool.close()
    pool.join()
    print(r)
    print('finished')


if __name__ == "__main__":
    multiprocessing.freeze_support()

    import time
    time_be = time.time()

    multip_grabCut()

    time_end = time.time()
    m, s = divmod(time_end-time_be, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

