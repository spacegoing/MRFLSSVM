# -*- coding: utf-8 -*-
import pickle
from Checkboard import Instance, Options
from GrabCut.GrabCutInstance import GrabCutInstance, get_name_path_arr
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
                      grabInstance.superpixel_method + '/' + filename+'.pickle','wb') as f:
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

def _load_grabcut_unary_pairwise_cliques(mask_type='_new', superpixel_method='slic'):
    dump_dir = './GrabCut/Data/grabCut/UnaryPairwiseCliques/'
    dump_parsed_dir = dump_dir + mask_type + '/' + superpixel_method + '/'
    image_files = os.listdir(dump_parsed_dir)

    instance_name_dict_list = list()
    for i in range(len(image_files)):
        filename = image_files[i].split('.')[0]
        with open(dump_parsed_dir + filename + '.pickle', 'rb') as f:
            grabInstance = pickle.load(f)

        instance_name_dict_list.append({'grabInstance': grabInstance,
                                        'filename': filename})

    return instance_name_dict_list