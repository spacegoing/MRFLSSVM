# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

import numpy as np
import pickle
from Utils.IOhelpers import dump_pickle
from Checkboard import Instance, Options
from CCCP import cccp_outer_loop
from ReportPlots import plot_linfunc_converged, plot_colormap
from GrabCut.GrabCutInstance import GrabCutInstance, get_name_path_arr

image_dir = './GrabCut/Data/grabCut/images/'

mask_dir = './GrabCut/Data/grabCut/labels/'
mask_ground_truth_type = ''
mask_input_type = '_new'

name_image_mask_truemask = get_name_path_arr(image_dir, mask_dir,
                                             mask_ground_truth_type, mask_input_type)

i = 20
image_name = name_image_mask_truemask[i, 0]
image_path = name_image_mask_truemask[i, 1]
mask_path = name_image_mask_truemask[i, 2]
true_mask_path = name_image_mask_truemask[i, 3]

grabInstance = GrabCutInstance(image_path, mask_path, true_mask_path,
                               method='slic', numSegments=300)

instance = Instance()
options = Options()

options.H = grabInstance.true_mask.shape[0]  # rows image height
options.W = grabInstance.true_mask.shape[1]  # cols image width
options.numCliques = len(np.unique(grabInstance.cliques))  # number of clique_indexes
options.rowsPairwise = grabInstance.pairwise.shape[0]

instance.clique_indexes = grabInstance.cliques.astype(np.int32)
instance.pairwise = grabInstance.pairwise
instance.unary_observed = grabInstance.unary_observed
instance.y = grabInstance.true_mask

# Image Configs
options.gridStep = np.inf  # grid size for defining clique_indexes
options.numVariables = options.H * options.W
options.N = options.H * options.W  # number of variables

options.dimUnary = 2
options.dimPairwise = 3

# Learning Configs
options.K = 10  # number of lower linear functions
options.sizeHighPhi = 2 * options.K - 1
options.sizePhi = options.sizeHighPhi + 2
options.maxIters = 100  # maximum learning iterations
options.eps = 1.0e-16  # constraint violation threshold

# Other Configs
options.learningQP = 1  # encoding for learning QP (1, 2, or 3)
options.figWnd = 0  # figure for showing results
options.hasPairwise = True  # dimPairwise = 0 when it's false
options.log_history = True

instance.latent_var = np.zeros([Options.numCliques, Options.K - 1])

prefix_str = "./expData/grabCutRes/"+image_name
outer_history = cccp_outer_loop(instance, options)
dump_pickle(prefix_str, outer_history, instance, options)
plot_colormap(prefix_str, outer_history, instance, options)
plot_linfunc_converged(prefix_str, outer_history, options)