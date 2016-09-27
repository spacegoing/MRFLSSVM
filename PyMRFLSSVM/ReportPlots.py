# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
import matplotlib.pyplot as plt
import numpy as np
from Utils.IOhelpers import load_pickle

root_path = "./"

sym_concave_active_str = "sym_concave_active"
sym_concave_inactive_str = "sym_concave_inactive"
asym_concave_active_str = "asym_concave_active"
asym_concave_inactive_str = "asym_concave_inactive"

sym_random_active_str = "sym_random_active"
sym_random_inactive_str = "sym_random_inactive"
asym_random_active_str = "asym_random_active"
asym_random_inactive_str = "asym_random_inactive"


def gen_plot_samples(theta, latent_inferred, options):
    def intersect(a_1, b_1, a_2, b_2, func_idx, i):
        if a_1 - a_2 == 0:
            print('Intersection Equals 0!\ntheta: %d and %d' % (func_idx, i))
            return
        x = (b_2 - b_1) / (a_1 - a_2)
        y = (a_1 * b_2 - a_2 * b_1) / (a_1 - a_2)
        # Can't exceed 1
        if x > 1:
            x = 1
            y = a_1 + b_1
        return x, y

    # decode theta
    a_b_array = np.zeros([options.K, 2])
    a_b_array[0, 0] = theta[0]
    for i in range(1, options.K):
        a_b_array[i, 0] = theta[i] + a_b_array[i - 1, 0]
        a_b_array[i, 1] = theta[i + options.K - 1] + a_b_array[i - 1, 1]

    # generate intersection points
    active_inter_points_list = [[0, 0]]
    active_func_idx_list = [0]

    func_idx = 0
    while func_idx < options.K - 1:
        inter_points = list()
        a_1 = a_b_array[func_idx, 0]
        b_1 = a_b_array[func_idx, 1]
        for i in range(func_idx + 1, options.K):
            a_2 = a_b_array[i, 0]
            b_2 = a_b_array[i, 1]
            point = intersect(a_1, b_1, a_2, b_2, func_idx, i)
            if point:
                inter_points.append(point)
            else:
                continue

        if inter_points:
            inter_points = np.asarray(inter_points)

            # Which functions is lower (inter point nearest to original point)
            active_inter_point_idx = np.argmin(inter_points[:, 0])
            active_point = inter_points[active_inter_point_idx, :]
            active_inter_points_list.append(active_point)
            if active_point[0] == 1:
                active_func_idx = active_inter_point_idx + func_idx
                active_func_idx_list.append(active_func_idx)
                break
            else:
                active_func_idx = active_inter_point_idx + func_idx + 1
                active_func_idx_list.append(active_func_idx)
                func_idx = active_func_idx
        else:
            break

    if 1.0 not in np.asarray(active_inter_points_list)[:, 0]:
        x = 1
        max_latent = np.max(np.sum(latent_inferred, axis=1))
        y = a_b_array[max_latent, 0] + a_b_array[max_latent, 1]
        active_inter_points_list.append([x, y])

    active_inter_points = np.asarray(active_inter_points_list)
    active_func_idxs = np.unique(active_func_idx_list)

    return active_inter_points


def plot_linfunc_by_iters(save_name, outer_history, options):
    for t in range(len(outer_history)):
        latent_inferred = outer_history[t]['latent_inferred']
        inner_history = outer_history[t]['inner_history']

        for i in range(len(inner_history)):
            x_y_samples = gen_plot_samples(inner_history[i]['theta'], latent_inferred, options)
            plt.plot(x_y_samples[:, 0], x_y_samples[:, 1], '-*')
            # print order text
            plt.text(x_y_samples[np.argmax(x_y_samples[:, 1]), 0],
                     np.max(x_y_samples[:, 1]), str(i))

        nonzero_idx = np.where(np.sum(latent_inferred, axis=1) != 0)[0]
        nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
        latent_str = str(latent_inferred[nonzero_idx, :])
        plt.title('active latent var: ' + latent_str)

        plt.savefig(root_path + save_name + '_iteration_%d.png' % t, format='png', dpi=600)
        plt.close()


def plot_linfunc_converged(save_name, outer_history, options):
    theta = outer_history[-1]['inner_history'][-1]['theta']
    latent_inferred = outer_history[-1]['latent_inferred']

    plot_samples = gen_plot_samples(theta,
                                    latent_inferred,
                                    options)
    plt.plot(plot_samples[:, 0], plot_samples[:, 1], '-*')

    nonzero_idx = np.where(np.sum(latent_inferred, axis=1) != 0)[0]
    nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
    latent_str = str(latent_inferred[nonzero_idx, :])
    plt.title(save_name + ' latent var: ' + latent_str)

    plt.savefig(root_path + save_name + '_theta_converge.png', dpi=100)
    plt.close()


def plot_colormap(save_name, outer_history, instance, options):
    # Plot ground_truth and observed unary (symmetric and asym)
    plt.imshow(instance.y, cmap='Greys', interpolation='nearest')
    plt.savefig(root_path + save_name + '_ground_truth.png', dpi=100)
    plt.close()

    unary = instance.unary_observed
    plt.imshow(unary[:, :, 1], cmap='Greys', interpolation='nearest')
    plt.savefig(root_path + save_name + '_unary.png', dpi=100)
    plt.close()

    # # Plot color map at 2cd iteration
    # iter_no = 2
    # y_hat = outer_history[-1]['inner_history'][iter_no]['y_hat']
    # plt.imshow(y_hat, cmap='Greys', interpolation='nearest')
    # plt.savefig(root_path + save_name + '_grey_map_%diter.png' % (iter_no + 1), dpi=100)
    # plt.close()

    # Plot converged color map
    iter_no = len(outer_history[-1]['inner_history']) - 1
    y_hat = outer_history[-1]['inner_history'][iter_no]['y_hat']
    plt.imshow(y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(root_path + save_name + '_grey_map_converged_%diter.png' % (iter_no + 1), dpi=100)
    plt.close()


if __name__ == "__main__":
    filepath = sym_concave_active_str
    outer_history, instance, options = load_pickle(filepath)
    save_name = sym_concave_active_str

    plot_linfunc_by_iters(save_name, outer_history, options)
    plot_linfunc_converged(save_name, outer_history, options)
    plot_colormap(save_name, outer_history, instance, options)
