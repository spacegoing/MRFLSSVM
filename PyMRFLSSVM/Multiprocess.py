# -*- coding: utf-8 -*-
from Batch_CCCP import cccp_outer_loop
import sys
from Checkboard import Instance
import numpy as np
from ReportPlots import plot_linfunc_converged, plot_colormap
import Batch_MRF_Helpers as mrf
from MrfTypes import Example, Options, BatchExamplesParser
from Utils.IOhelpers import dump_pickle
import multiprocessing

inf_latent_method = ''
init_method = 'clique_by_clique'


parser = BatchExamplesParser()
root = './expData/unbalaced_portions/'
def calcFun(args):
    func,arg=args
    result=func(*arg)
    return result


def calcfun(prefix_str, miu):
    instance = Instance('gaussian_portions', portion_miu=miu, is_gaussian=False)
    examples_list = parser.parse_checkboard(instance)

    options = Options()
    outer_history = cccp_outer_loop([examples_list[0]], options, init_method, inf_latent_method)

    dump_pickle(prefix_str, outer_history, instance, options)
    plot_colormap(prefix_str, outer_history, instance, options)
    plot_linfunc_converged(prefix_str, outer_history, options)
    sys.stdout.flush()

if __name__ =="__main__":
    multiprocessing.freeze_support()
    data_list = [[root + "more_black_3339", (0.3, 0.3, 0.3, 0.9)],
                 [root + "more_white_1777", (0.1, 0.7, 0.7, 0.7)],
                 [root + "balanced_portions_124678", (0.1, 0.2, 0.4, 0.6, 0.7, 0.8)]]
    Tasks = [(calcfun,i) for i in data_list]
    pool = multiprocessing.Pool(4)
    r=pool.map(calcFun,Tasks,1)
    pool.close()
    pool.join()
    print(r)
    print('finished')

