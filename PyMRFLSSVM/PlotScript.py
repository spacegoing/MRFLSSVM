# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Checkboard import Options, Instance, checkboardHelper

__author__ = 'spacegoing'
root_path = '/Users/spacegoing/macCodeLab-MBP2015/HonoursDoc/ExperimentsLatex/temp/'
checkboard_image_path = root_path + 'checkboard_images/'
asym_active_path = checkboard_image_path + 'sym_active.pickle'
sym_inactive_path = checkboard_image_path + 'sym_inactive.pickle'
asym_active_path = checkboard_image_path + 'asym_active.pickle'
asym_inactive_path = checkboard_image_path + 'asym_inactive.pickle'


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        latent_inferred = data[0]
        history = data[1]
        options = data[2]

    return latent_inferred, history, options


def plot_color_map():
    # Plot ground_truth and observed unary (symmetric and asym)
    instance = Instance()
    plt.imshow(instance.y, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'ground_truth.png', dpi=100)
    plt.close()

    sym_unary = checkboardHelper(0.1, 0.1)[2]
    plt.imshow(sym_unary[:, :, 1], cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'sym_unary.png', dpi=100)
    plt.close()

    asym_unary = checkboardHelper(0.1, 0.5)[2]
    plt.imshow(asym_unary[:, :, 1], cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'asym_unary.png', dpi=100)
    plt.close()

    # Load data
    sym_active_latent_inferred, sym_active_history, sym_active_options \
        = load_pickle(sym_active_path)
    sym_inactive_latent_inferred, sym_inactive_history, sym_inactive_options \
        = load_pickle(sym_inactive_path)
    asym_active_latent_inferred, asym_active_history, asym_active_options \
        = load_pickle(asym_active_path)
    asym_inactive_latent_inferred, asym_inactive_history, asym_inactive_options \
        = load_pickle(asym_inactive_path)

    # Plot color map at 10th iteration
    iter_no = 9
    sym_active_y_hat = sym_active_history[iter_no]['y_hat']
    plt.imshow(sym_active_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'sym_active_grey_map_%diter.png' % (iter_no + 1), dpi=100)
    plt.close()
    sym_inactive_y_hat = sym_inactive_history[iter_no]['y_hat']
    plt.imshow(sym_inactive_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'sym_inactive_grey_map_%diter.png' % (iter_no + 1), dpi=100)
    plt.close()
    asym_active_y_hat = asym_active_history[iter_no]['y_hat']
    plt.imshow(asym_active_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'asym_active_grey_map_%diter.png' % (iter_no + 1), dpi=100)
    plt.close()
    asym_inactive_y_hat = asym_inactive_history[iter_no]['y_hat']
    plt.imshow(asym_inactive_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'asym_inactive_grey_map_%diter.png' % (iter_no + 1), dpi=100)
    plt.close()

    # Plot converged color map
    iter_no = -1
    sym_active_y_hat = sym_active_history[iter_no]['y_hat']
    plt.imshow(sym_active_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'sym_active_grey_map_converge.png', dpi=100)
    plt.close()
    sym_inactive_y_hat = sym_inactive_history[iter_no]['y_hat']
    plt.imshow(sym_inactive_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'sym_inactive_grey_map_converge.png', dpi=100)
    plt.close()
    asym_active_y_hat = asym_active_history[iter_no]['y_hat']
    plt.imshow(asym_active_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'asym_active_grey_map_converge.png', dpi=100)
    plt.close()
    asym_inactive_y_hat = asym_inactive_history[iter_no]['y_hat']
    plt.imshow(asym_inactive_y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(checkboard_image_path + 'asym_inactive_grey_map_converge.png', dpi=100)
    plt.close()


def gen_plot_samples(theta, latent_plot, options):
    # decode theta
    a_b_array = np.zeros([options.K, 2])
    a_b_array[0, 0] = theta[0]
    for i in range(1, options.K):
        a_b_array[i, 0] = theta[i] + a_b_array[i - 1, 0]
        a_b_array[i, 1] = theta[i + options.K - 1] + a_b_array[i - 1, 1]

    # generate intersection points
    inter_points = np.zeros([options.K + 1, 2])
    for i in range(1, options.K):
        a_2 = a_b_array[i, 0]
        b_2 = a_b_array[i, 1]
        a_1 = a_b_array[i - 1, 0]
        b_1 = a_b_array[i - 1, 1]
        inter_points[i, 0] = (b_2 - b_1) / (a_1 - a_2)
        inter_points[i, 1] = (a_1 * b_2 - a_2 * b_1) / (a_1 - a_2)

    max_latent = np.max(np.sum(latent_plot, axis=1))

    unique_inter_points = list()
    for i in inter_points:
        if i[0] >= 1:
            i[0] = 1
            i[1] = a_b_array[max_latent, 0] + a_b_array[max_latent, 1]
        if tuple(i) not in unique_inter_points:
            unique_inter_points += [tuple(i)]
    unique_inter_points = np.asarray(unique_inter_points)
    if unique_inter_points[-1][0] < 1:
        np.r_['0,2', unique_inter_points, [1, a_b_array[max_latent, 0]
                                           + a_b_array[max_latent, 1]]]

    return unique_inter_points


def plot_linEnv():
    # Load data
    sym_active_latent_inferred, sym_active_history, sym_active_options \
        = load_pickle(sym_active_path)
    sym_inactive_latent_inferred, sym_inactive_history, sym_inactive_options \
        = load_pickle(sym_inactive_path)
    asym_active_latent_inferred, asym_active_history, asym_active_options \
        = load_pickle(asym_active_path)
    asym_inactive_latent_inferred, asym_inactive_history, asym_inactive_options \
        = load_pickle(asym_inactive_path)

    # Plot converged linear envelopes
    iter_no = -1

    sym_active_theta = sym_active_history[iter_no]['theta']
    sym_active_plot_samples = gen_plot_samples(sym_active_theta,
                                               sym_active_latent_inferred,
                                               sym_active_options)
    plt.plot(sym_active_plot_samples[:, 0], sym_active_plot_samples[:, 1], '-*')
    nonzero_idx = np.where(np.sum(sym_active_latent_inferred, axis=1) != 0)[0]
    nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
    latent_str = str(sym_active_latent_inferred[nonzero_idx, :])
    plt.title('active latent var: ' + latent_str)
    plt.savefig(checkboard_image_path + 'sym_active_theta_converge.png', dpi=100)
    plt.close()

    # # points for latex
    # for i in range(1,5):
    #     print('(%.16f, %.16f)\\\\' %(sym_active_plot_samples[i,0],sym_active_plot_samples[i,1]))


    sym_inactive_theta = sym_inactive_history[iter_no]['theta']
    sym_inactive_plot_samples = gen_plot_samples(sym_inactive_theta,
                                                 sym_inactive_latent_inferred,
                                                 sym_inactive_options)
    plt.plot(sym_inactive_plot_samples[:, 0], sym_inactive_plot_samples[:, 1], '-*')
    nonzero_idx = np.where(np.sum(sym_inactive_latent_inferred, axis=1) != 0)[0]
    nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
    latent_str = str(sym_inactive_latent_inferred[nonzero_idx, :])
    plt.title('inactive latent var: ' + latent_str)
    plt.savefig(checkboard_image_path + 'sym_inactive_theta_converge.png', dpi=100)
    plt.close()

    asym_active_theta = asym_active_history[iter_no]['theta']
    asym_active_plot_samples = gen_plot_samples(asym_active_theta,
                                                asym_active_latent_inferred,
                                                asym_active_options)
    plt.plot(asym_active_plot_samples[:, 0], asym_active_plot_samples[:, 1], '-*')
    nonzero_idx = np.where(np.sum(asym_active_latent_inferred, axis=1) != 0)[0]
    nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
    latent_str = str(asym_active_latent_inferred[nonzero_idx, :])
    plt.title('active latent var: ' + latent_str)
    plt.savefig(checkboard_image_path + 'asym_active_theta_converge.png', dpi=100)
    plt.close()

    asym_inactive_theta = asym_inactive_history[iter_no]['theta']
    asym_inactive_plot_samples = gen_plot_samples(asym_inactive_theta,
                                                  asym_inactive_latent_inferred,
                                                  asym_inactive_options)
    plt.plot(asym_inactive_plot_samples[:, 0], asym_inactive_plot_samples[:, 1], '-*')
    nonzero_idx = np.where(np.sum(asym_inactive_latent_inferred, axis=1) != 0)[0]
    nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
    latent_str = str(asym_inactive_latent_inferred[nonzero_idx, :])
    plt.title('inactive latent var: ' + latent_str)
    plt.savefig(checkboard_image_path + 'asym_inactive_theta_converge.png', dpi=100)
    plt.close()
