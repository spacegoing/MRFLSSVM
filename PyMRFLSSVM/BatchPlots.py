# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from MrfTypes import BatchExamplesParser, Options

__author__ = 'spacegoing'

root_path = "./expData/batchPlots/"


def gen_plot_samples(theta, max_latent, K):
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
    a_b_array = np.zeros([K, 2])
    a_b_array[0, 0] = theta[0]
    for i in range(1, K):
        a_b_array[i, 0] = theta[i] + a_b_array[i - 1, 0]
        a_b_array[i, 1] = theta[i + K - 1] + a_b_array[i - 1, 1]

    # generate intersection points
    active_inter_points_list = [[0, 0]]
    active_func_idx_list = [0]

    func_idx = 0
    while func_idx < K - 1:
        inter_points = list()
        a_1 = a_b_array[func_idx, 0]
        b_1 = a_b_array[func_idx, 1]
        for i in range(func_idx + 1, K):
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
        y = a_b_array[max_latent, 0] + a_b_array[max_latent, 1]
        active_inter_points_list.append([x, y])

    active_inter_points = np.asarray(active_inter_points_list)

    return active_inter_points


# def plot_linfunc_by_iters(save_name, outer_history, options):
#     for t in range(len(outer_history)):
#         latent_inferred = outer_history[t]['latent_inferred']
#         inner_history = outer_history[t]['inner_history']
#
#         for i in range(len(inner_history)):
#             x_y_samples = gen_plot_samples(inner_history[i]['theta'], latent_inferred, options)
#             plt.plot(x_y_samples[:, 0], x_y_samples[:, 1], '-*')
#             # print order text
#             plt.text(x_y_samples[np.argmax(x_y_samples[:, 1]), 0],
#                      np.max(x_y_samples[:, 1]), str(i))
#
#         nonzero_idx = np.where(np.sum(latent_inferred, axis=1) != 0)[0]
#         nonzero_idx = nonzero_idx[0] if nonzero_idx.shape[0] else 0
#         latent_str = str(latent_inferred[nonzero_idx, :])
#         plt.title('active latent var: ' + latent_str)
#
#         plt.savefig(root_path + save_name + '_iteration_%d.png' % t, format='png', dpi=600)
#         plt.close()


def plot_linfunc_converged(save_name, theta, max_latent, K):
    plot_samples = gen_plot_samples(theta,
                                    max_latent,
                                    K)
    plt.plot(plot_samples[:, 0], plot_samples[:, 1], '-*')
    plt.savefig(root_path + save_name + '_theta_converge.png', dpi=100)
    plt.close()


def plot_colormap(save_name, y, unary_observed, y_hat):
    # Plot ground_truth and observed unary (symmetric and asym)
    plt.imshow(y, cmap='Greys', interpolation='nearest')
    plt.savefig(root_path + save_name + '_ground_truth.png', dpi=100)
    plt.close()

    plt.imshow(unary_observed[:, :, 0], cmap='Greys', interpolation='nearest')
    plt.savefig(root_path + save_name + '_toSource_unary.png', dpi=100)
    plt.close()

    plt.imshow(unary_observed[:, :, 1], cmap='Greys', interpolation='nearest')
    plt.savefig(root_path + save_name + '_toSink_unary.png', dpi=100)
    plt.close()

    # Plot converged color map
    plt.imshow(y_hat, cmap='Greys', interpolation='nearest')
    plt.savefig(root_path + save_name + '_inferred_map_converged.png', dpi=100)
    plt.close()


from Batch_MRF_Helpers import inf_label_latent_helper


class BatchPlotWrapper:
    def __init__(self, examples_list, outer_history, options):
        self.examples_list = examples_list
        self.y_hat_list = list()
        self.name_list = list()
        self.theta = outer_history[-1]['inner_history'][-1]['theta']
        self.K = options.K

        self.max_latent = 0
        for latent_inferred in outer_history[-1]['latent_inferred_list']:
            self.max_latent = np.max([self.max_latent,
                                      np.max(np.sum(latent_inferred, axis=1))])

        for ex in examples_list:
            self.name_list = ex.name
            self.y_hat_list.append(inf_label_latent_helper(
                ex.unary_observed, ex.pairwise,
                ex.clique_indexes, self.theta, options, ex.hasPairwise)[0])

    def plot_linfunc_converged(self, image_no):
        plot_linfunc_converged(self.examples_list[image_no].name,
                               self.theta, self.max_latent, self.K)

    def plot_color_map(self, image_no):
        ex = self.examples_list[image_no]
        plot_colormap(ex.name, ex.y, ex.unary_observed, self.y_hat_list[image_no])

    # def match_train_test_pairs(self):
    #
    #     inferred_label_name_list = list()
    #     for i in inferred_label_name_loss_list:
    #         inferred_label, loss, test_name = i
    #         if test_name in ['banana1.pickle', '37073.pickle', '24077.pickle']:
    #             inferred_label_name_list.append([test_name, inferred_label])
    #
    #     image_dir = './GrabCut/Data/grabCut/images/'
    #
    #     name_list = ['banana1.bmp',
    #                  '37073.jpg',
    #                  '24077.jpg']
    #     import cv2
    #     img_dict = dict()
    #     for name in name_list:
    #         img = cv2.imread(image_dir + name)
    #         img_dict[name.split('.')[0]] = img
    #
    #     prd_img_list = list()
    #     for i in inferred_label_name_list:
    #         test_name, inferred_label = i
    #         prd_img = img_dict[test_name.split('.')[0]] * \
    #                   inferred_label[:, :, np.newaxis]
    #         prd_img_list.append([prd_img, test_name.split('.')[0]])
    #
    #     import pickle
    #     with open('oaijewopi.pickle', 'wb') as f:
    #         pickle.dump(prd_img_list, f)
    #
    #     with open('oaijewopi.pickle', 'rb') as f:
    #         prd_img_list = pickle.load(f)
    #
    #     for i in prd_img_list:
    #         img, name = i
    #         plt.imshow(img), plt.savefig(name,dpi=100)
    #
    #     1 - 0.13119733680481344
    #     1 - 0.34933593750000003
    #     1 - 0.39286662651148635
