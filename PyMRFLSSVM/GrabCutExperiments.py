# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import pickle
import numpy as np
from MrfTypes import Example, Options
from BatchPlots import BatchPlotWrapper

options = Options()
image_name_list = ["more_black_3339", "more_white_1777""balanced_portions_124678"]

image_name = image_name_list[0]

batch_converge_pickle_dir = "./expData/batchResult/training_result/image_%s_outer_history.pickle"
with open(batch_converge_pickle_dir % image_name, 'rb') as f:
    outer_history, examples_list, prefix_str = pickle.load(f)
plot_wrapper = BatchPlotWrapper(examples_list, outer_history, options)
