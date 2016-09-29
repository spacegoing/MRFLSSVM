# -*- coding: utf-8 -*-
from Checkboard import Instance
import numpy as np
from ReportPlots import plot_linfunc_converged, plot_colormap
import Batch_MRF_Helpers as mrf
from MrfTypes import Example, Options, BatchExamplesParser
from Utils.IOhelpers import dump_pickle



parser = BatchExamplesParser()
root = './expData/unbalaced_portions/'

# more black (1s)
prefix_str = "more_black_3339"
prefix_str = root + prefix_str

instance = Instance('gaussian_portions', portion_miu=(0.3, 0.3, 0.3, 0.9), is_gaussian=False)
examples_list = parser.parse_checkboard(instance)

options = Options()
outer_history = cccp_outer_loop([examples_list[0]], options, init_method, inf_latent_method)

dump_pickle(prefix_str, outer_history, instance, options)
plot_colormap(prefix_str, outer_history, instance, options)
plot_linfunc_converged(prefix_str, outer_history, options)

# more white (0s)
prefix_str = "more_white_1777"
prefix_str = root + prefix_str
instance = Instance('gaussian_portions', portion_miu=(0.1, 0.7, 0.7, 0.7), is_gaussian=False)
examples_list = parser.parse_checkboard(instance)

options = Options()
outer_history = cccp_outer_loop([examples_list[0]], options, init_method, inf_latent_method)
dump_pickle(prefix_str, outer_history, instance, options)
plot_colormap(prefix_str, outer_history, instance, options)
plot_linfunc_converged(prefix_str, outer_history, options)

prefix_str = "balanced_portions_124678"
prefix_str = root + prefix_str
instance = Instance('gaussian_portions',
                    portion_miu=(0.1, 0.2, 0.4,
                                    0.6, 0.7, 0.8), is_gaussian=False)
examples_list = parser.parse_checkboard(instance)

options = Options()
outer_history = cccp_outer_loop([examples_list[0]], options, init_method, inf_latent_method)
dump_pickle(prefix_str, outer_history, instance, options)
plot_colormap(prefix_str, outer_history, instance, options)
plot_linfunc_converged(prefix_str, outer_history, options)
