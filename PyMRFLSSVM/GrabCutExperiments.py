# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import pickle
import numpy as np
from MrfTypes import Example, Options
from BatchPlots import BatchPlotWrapper

options = Options()

def plot_checkboard():
    #
    # multiprocessing.freeze_support()
    # data_list = [["more_black_3339", (0.3, 0.3, 0.3, 0.9)],
    #              ["more_white_1777", (0.1, 0.7, 0.7, 0.7)],
    #              ["balanced_portions_124678", (0.1, 0.2, 0.4, 0.6, 0.7, 0.8)]]
    # Tasks = [(calcCheckboard, miu) for miu in data_list]
    # pool = multiprocessing.Pool(3)
    # r = pool.map(calcFun, Tasks)
    # pool.close()
    # pool.join()
    # print(r)
    # print('finished')

    image_name_list = ["more_black_3339", "more_white_1777", "balanced_portions_124678"]

    for image_name in image_name_list:
        batch_converge_pickle_dir = "./expData/batchResult/training_result/image_%s_outer_history.pickle"
        with open(batch_converge_pickle_dir % image_name, 'rb') as f:
            outer_history, examples_list, prefix_str = pickle.load(f)
        examples_list[0].name = image_name
        plot_wrapper = BatchPlotWrapper(examples_list, outer_history, options)
        plot_wrapper.plot_linfunc_converged(0)
        plot_wrapper.plot_color_map(0)
