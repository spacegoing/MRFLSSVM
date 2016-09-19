# -*- coding: utf-8 -*-
import numpy as np
import pickle
from Utils.IOhelpers import dump_pickle

from Checkboard import Instance, Options
from CCCP import cccp_outer_loop
from ReportPlots import plot_linfunc_converged, plot_colormap

__author__ = 'spacegoing'

################################# Black_White Checkboard:#################################
# 1. Comparing to Prev. Results
# todo: 2. Show a_{n+1}-a{n}<-e problem
# todo: 3. Show init problem
root = "./expData/black_white/"
prefix_str = "bw_sym_concave_"
prefix_str = root + prefix_str
active = False
inactive = False

ina_counter = 0
a_counter = 0
while not (active and inactive):
    instance = Instance('black_white')
    options = Options()
    outer_history = cccp_outer_loop(instance, options)
    latent_inferred = outer_history[-1]["latent_inferred"]

    if np.sum(latent_inferred) == 0:
        if inactive:
            continue
        else:
            dump_pickle(prefix_str + 'inactive.pickle', outer_history, instance, options)
            inactive = True
            ina_counter += 1
            plot_colormap(prefix_str + 'inactive.pickle', outer_history, instance, options)
            plot_linfunc_converged(prefix_str + 'inactive.pickle', outer_history, options)
    else:
        if active:
            continue
        else:
            dump_pickle(prefix_str + 'active.pickle', outer_history, instance, options)
            active = True
            a_counter += 1
            plot_colormap(prefix_str + 'active.pickle', outer_history, instance, options)
            plot_linfunc_converged(prefix_str + 'active.pickle', outer_history, options)

print("active/total: %f" % (a_counter / (a_counter + ina_counter)))

# Asymmetric Noisy
prefix_str = "bw_asym_concave_"
prefix_str = root + prefix_str
active = False
inactive = False

ina_counter = 0
a_counter = 0
while not (active and inactive):
    instance = Instance('black_white', _eta=(0.1, 0.5))
    options = Options()
    outer_history = cccp_outer_loop(instance, options)
    latent_inferred = outer_history[-1]["latent_inferred"]

    if np.sum(latent_inferred) == 0:
        if inactive:
            continue
        else:
            dump_pickle(prefix_str + 'inactive.pickle', outer_history, instance, options)
            inactive = True
            ina_counter += 1
            plot_colormap(prefix_str + 'inactive.pickle', outer_history, instance, options)
            plot_linfunc_converged(prefix_str + 'inactive.pickle', outer_history, options)
    else:
        if active:
            continue
        else:
            dump_pickle(prefix_str + 'active.pickle', outer_history, instance, options)
            active = True
            a_counter += 1
            plot_colormap(prefix_str + 'active.pickle', outer_history, instance, options)
            plot_linfunc_converged(prefix_str + 'active.pickle', outer_history, options)

print("active/total: %f" % (a_counter / (a_counter + ina_counter)))

################################# Unbalanced Balanced Portions Checkboard:#################################
root = './expData/unbalaced_portions/'

prefix_str = "more_black_1and4"
prefix_str = root + prefix_str
instance = Instance('gaussian_portions', portion_miu=(0.1, 0.4))
options = Options()
outer_history = cccp_outer_loop(instance, options, init_method='clique_by_clique')
dump_pickle(prefix_str + '.pickle', outer_history, instance, options)
plot_colormap(prefix_str + '.pickle', outer_history, instance, options)
plot_linfunc_converged(prefix_str + '.pickle', outer_history, options)

prefix_str = "balanced_portions_6and9"
prefix_str = root + prefix_str
instance = Instance('gaussian_portions', portion_miu=(0.6, 0.9))
options = Options()
outer_history = cccp_outer_loop(instance, options, init_method='clique_by_clique')
dump_pickle(prefix_str + '.pickle', outer_history, instance, options)
plot_colormap(prefix_str + '.pickle', outer_history, instance, options)
plot_linfunc_converged(prefix_str + '.pickle', outer_history, options)

prefix_str = "balanced_portions_1to9"
prefix_str = root + prefix_str
instance = Instance('gaussian_portions',
                    portion_miu=(0.1, 0.2, 0.4,
                                 0.6, 0.7, 0.8), is_gaussian=False)
options = Options()
outer_history = cccp_outer_loop(instance, options, init_method='clique_by_clique')
dump_pickle(prefix_str + '.pickle', outer_history, instance, options)
plot_colormap(prefix_str + '.pickle', outer_history, instance, options)
plot_linfunc_converged(prefix_str + '.pickle', outer_history, options)
