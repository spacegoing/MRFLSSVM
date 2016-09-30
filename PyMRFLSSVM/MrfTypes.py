# -*- coding: utf-8 -*-
import numpy as np
import Checkboard

# from pprint import pprint as pp
# from Utils.ReadMat import loadCheckboard

__author__ = 'spacegoing'


class Options:
    """
    :type K: int
    :type sizeHighPhi: int
    :type sizePhi: int
    :type maxIters: int
    :type eps: float
    """
    # Learning Configs
    K = 10  # number of lower linear functions
    sizeHighPhi = 2 * K - 1
    sizePhi = sizeHighPhi + 2
    maxIters = 100  # maximum learning iterations
    eps = 1.0e-7  # constraint violation threshold
    # Other Configs
    # # todo: encoding for learning QP (1, 2, or 3)
    # learningQP = 1


class Example:
    """
    :param name: name of graphcut image default:''
    :type rows: int
    :type cols: int
    :type numCliques: int
    :type numVariables: int
    :type dimPairwise: int
    :type rowsPairwise: int
    """

    # Image Configs

    def __init__(self, y: np.ndarray, unary_observed: np.ndarray,
                 clique_indexes: np.ndarray, hasPairwise: bool = True,
                 pairwise: np.ndarray = np.zeros([1, 3]), name: str = ''):
        self.y = y
        self.unary_observed = unary_observed
        self.clique_indexes = clique_indexes

        # H = rows
        self.rows = self.y.shape[0]
        # W = cols
        self.cols = self.y.shape[1]
        self.numCliques = len(np.unique(self.clique_indexes))
        self.numVariables = self.rows * self.cols

        self.pairwise = pairwise
        # np.([[index1, index2, pairwise],...])
        self.hasPairwise = hasPairwise  # dimPairwise = 0 when it's false
        self.dimPairwise = 3
        self.rowsPairwise = self.pairwise.shape[0]

        self.name = name


class BatchExamplesParser:
    def __init__(self):
        '''

        :param example_list:
        :type example_list: list[Example]
        '''

    def parse_grabcut_pickle(self, raw_example_list):
        '''

        :param raw_example_list:
        :type raw_example_list:
        :return:
        :rtype: list[Example]
        '''

        examples_list = list()
        for i in raw_example_list:
            ex = Example(i['grabInstance'].true_mask,
                         i['grabInstance'].unary_observed,
                         i['grabInstance'].cliques,
                         True, i['grabInstance'].pairwise,
                         i['filename'])

            examples_list.append(ex)

        return examples_list

    def parse_checkboard(self, instance):
        '''

        :param instance:
        :type instance: Checkboard.Instance
        :param options:
        :type options: Checkboard.Options
        :return:
        :rtype: list[Example]
        '''

        ex = Example(instance.y, instance.unary_observed,
                     instance.clique_indexes,
                     True, instance.pairwise, name='checkboard')
        examples_list = [ex]
        return examples_list
