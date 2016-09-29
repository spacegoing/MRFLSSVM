# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

# -*- coding: utf-8 -*-
from Batch_CCCP import cccp_outer_loop
from Checkboard import Instance
from ReportPlots import plot_linfunc_converged, plot_colormap
from MrfTypes import Options, BatchExamplesParser
from Utils.IOhelpers import dump_pickle
import multiprocessing
import sys


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
    cccp_outer_loop([examples_list[0]], options, init_method, inf_latent_method)

    # dump_pickle(prefix_str, outer_history, instance, options)
    # plot_colormap(prefix_str, outer_history, instance, options)
    # plot_linfunc_converged(prefix_str, outer_history, options)
    sys.stdout.flush()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    data_list = [[root + "more_black_3339", (0.3, 0.3, 0.3, 0.9)],
                 [root + "more_white_1777", (0.1, 0.7, 0.7, 0.7)],
                 [root + "balanced_portions_124678", (0.1, 0.2, 0.4, 0.6, 0.7, 0.8)]]
    Tasks = [(calcCheckboard, miu) for miu in data_list]
    pool = multiprocessing.Pool(3)
    r = pool.map(calcFun, Tasks)
    pool.close()
    pool.join()
    print(r)
    print('finished')
