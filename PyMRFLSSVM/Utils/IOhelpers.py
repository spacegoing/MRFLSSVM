# -*- coding: utf-8 -*-
import pickle
from Checkboard import Instance, Options
from GrabCut.GrabCutInstance import GrabCutInstance, get_name_path_arr
from MrfTypes import Example
import os

__author__ = 'spacegoing'


def dump_pickle(prefix_str, outer_history, instance, options):
    with open(prefix_str + '.pickle', 'wb') as f:
        pickle.dump({"outer_history": outer_history,
                     "instance": instance,
                     "options": options}, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    with open(filepath + ".pickle", 'rb') as f:
        data = pickle.load(f)
        instance = data['instance']
        options = data['options']
        outer_history = data['outer_history']

    return outer_history, instance, options


def __dump_grabcut_exps(dump_dir: str, mask_type: str, filename: str, grabInstance: GrabCutInstance):
    with open(dump_dir + mask_type + '/' +
                      grabInstance.superpixel_method + '/' + filename + '.pickle', 'wb') as f:
        pickle.dump(grabInstance, f, pickle.HIGHEST_PROTOCOL)


############save grabcut dataset unary pariwise and cliques features##################
def _dump_grabcut_unary_pairwise_cliques():
    image_dir = './GrabCut/Data/grabCut/images/'

    mask_dir = './GrabCut/Data/grabCut/labels/'
    mask_ground_truth_type = ''
    dump_dir = './GrabCut/Data/grabCut/UnaryPairwiseCliques/'

    mask_input_type_list = ['_new', '_rect', '_lasso']

    def inner_loop(mask_input_type):
        name_image_mask_truemask = get_name_path_arr(image_dir, mask_dir,
                                                     mask_ground_truth_type, mask_input_type)
        for i in range(len(name_image_mask_truemask)):
            image_name = name_image_mask_truemask[i, 0]
            image_path = name_image_mask_truemask[i, 1]
            mask_path = name_image_mask_truemask[i, 2]
            true_mask_path = name_image_mask_truemask[i, 3]
            print("start " + image_name + ' ' + mask_input_type)

            # slic
            grabInstance = GrabCutInstance(image_path, mask_path, true_mask_path,
                                           method='slic', numSegments=300)
            __dump_grabcut_exps(dump_dir, mask_input_type, image_name, grabInstance)

            # quickshift
            grabInstance = GrabCutInstance(image_path, mask_path, true_mask_path,
                                           method='quickshift')
            __dump_grabcut_exps(dump_dir, mask_input_type, image_name, grabInstance)

            # felzenszwalb
            grabInstance = GrabCutInstance(image_path, mask_path, true_mask_path,
                                           method='quickshift')
            __dump_grabcut_exps(dump_dir, mask_input_type, image_name, grabInstance)

    for m in mask_input_type_list:
        inner_loop(m)


def _load_grabcut_unary_pairwise_cliques(mask_type='_new',
                                         superpixel_method='slic'):
    """

    :param mask_type:
    :type mask_type: str
    :param superpixel_method:
    :type superpixel_method: str
    :return:
    :rtype: list[dict]
    """
    dump_dir = './GrabCut/Data/grabCut/UnaryPairwiseCliques/'
    dump_parsed_dir = dump_dir + mask_type + '/' + superpixel_method + '/'
    pickle_files = os.listdir(dump_parsed_dir)

    grabInstance_name_dict_list = list()
    for i in range(len(pickle_files)):
        filename = pickle_files[i]
        with open(dump_parsed_dir + filename, 'rb') as f:
            grabInstance = pickle.load(f)

        grabInstance_name_dict_list.append({'grabInstance': grabInstance,
                                            'filename': filename,
                                            'hasPairwise': True})

    return grabInstance_name_dict_list


def _load_grabcut_train_results(train_results_path="/Users/spacegoing/"
                                                   "macCodeLab-MBP2015/GrabCutResutls/"):
    pickle_files = os.listdir(train_results_path)
    name_theta_example_list = list()

    leave_out_train_results_dict = dict()
    empty_train_results_list = list()
    for filename in pickle_files:
        with open(train_results_path + filename, 'rb')as f:
            examples_list, outer_history, leave_out_name = pickle.load(f)
            try:
                theta_converged = outer_history[1]["inner_history"][-1]['theta']
                leave_out_train_results_dict[leave_out_name] = \
                    {'theta': theta_converged, 'outer_history': outer_history}
            except IndexError:
                empty_train_results_list.append(filename)

    return leave_out_train_results_dict
