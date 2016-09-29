# -*- coding: utf-8 -*-
from Batch_CCCP import cccp_outer_loop
from Checkboard import Instance
from MrfTypes import Example, Options, BatchExamplesParser
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

if __name__ == "__main__":
    multiprocessing.freeze_support()
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
