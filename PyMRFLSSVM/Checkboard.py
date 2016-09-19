# -*- coding: utf-8 -*-
import numpy as np
from numpy.matlib import repmat

# from pprint import pprint as pp
# from Utils.ReadMat import loadCheckboard

__author__ = 'spacegoing'

# initialize random number generator
np.random.seed(0)


class Options:
    # Image Configs
    gridStep = 16  # grid size for defining clique_indexes
    H = 128  # rows image height
    W = 128  # cols image width
    numCliques = (int)(H / gridStep) ** 2  # number of clique_indexes
    numVariables = H * W
    N = H * W  # number of variables

    dimUnary = 2
    dimPairwise = 3

    # Learning Configs
    K = 10  # number of lower linear functions
    sizeHighPhi = 2 * K - 1
    sizePhi = sizeHighPhi + 2
    maxIters = 100  # maximum learning iterations
    eps = 1.0e-16  # constraint violation threshold

    # Other Configs
    learningQP = 1  # encoding for learning QP (1, 2, or 3)
    figWnd = 0  # figure for showing results
    hasPairwise = False  # dimPairwise = 0 when it's false
    rowsPairwise = H * W * 2 - H - W
    log_history = True


class Instance:
    def __init__(self, checkboard_type='black_white', _eta=(0.1, 0.1), **kwargs):

        self._eta = _eta
        self.functions_dict = {'black_white': self.checkboardHelper_bw,
                               'triang': self.checkboardHelper_triad,
                               'shuffle': self.checkboardHelper_shuffle,
                               'gaussian_portions': self.checkboardHelper_gaussian_portions}
        check_func = self.functions_dict.get(checkboard_type, False)

        try:
            if check_func == self.checkboardHelper_gaussian_portions:
                self.clique_indexes, self.y = check_func(**kwargs)
            else:
                self.clique_indexes, self.y = check_func()
        except:
            self.help()
            raise

        self.unary_observed = self.init_unary_feature()
        self.pairwise = self.init_pairwise_feature()
        self.latent_var = np.zeros([Options.numCliques, Options.K - 1])

        # self.unary = loadCheckboard().astype(np.double)

    def help(self):
        print("Help: Type  doesn't exists or parameters error!")
        print("Available checkboard types:")
        for i in self.functions_dict.keys():
            print(i)
        print("Note: `gaussian_portions()` has key word parameters "
              "`portion_miu`, `is_gaussian` and `sigma`")

    # Generate checkboard data
    def checkboardHelper_bw(self):
        H = Options.H
        W = Options.W

        # create checkboard data (clique and ground_truth y)
        clique_indexes = np.zeros([H, W], dtype=np.int32, order='C')  # mapping of variables to clique_indexes
        y = np.zeros([H, W], dtype=np.int32, order='C')  # ground-truth labels
        _black = True  # indicate _black True or white False
        _cliqueID = 1.0  # clique index starts from 1
        for _rowIndx in range(0, H, Options.gridStep):
            for _colIndx in range(0, W, Options.gridStep):
                clique_indexes[_rowIndx:_rowIndx + Options.gridStep,
                _colIndx:_colIndx + Options.gridStep] = _cliqueID
                _cliqueID += 1.0

                y[_rowIndx:_rowIndx + Options.gridStep,
                _colIndx:_colIndx + Options.gridStep] = 0.0 if _black else 1.0
                _black = not _black

            _black = not _black

        return clique_indexes, y

    # Generate checkboard data
    def checkboardHelper_triad(self):
        H = Options.H
        W = Options.W

        # create checkboard data (clique and ground_truth y)
        clique_indexes = np.zeros([H, W], dtype=np.int32, order='C')  # mapping of variables to clique_indexes
        _cliqueID = 1.0  # clique index starts from 1
        for _rowIndx in range(0, H, Options.gridStep):
            for _colIndx in range(0, W, Options.gridStep):
                clique_indexes[_rowIndx:_rowIndx + Options.gridStep,
                _colIndx:_colIndx + Options.gridStep] = _cliqueID
                _cliqueID += 1.0

        y = np.zeros([H, W], dtype=np.int32, order='C')  # ground-truth labels
        avg_no = H // 2

        def line_seg(x, y):
            return 1 if y - 2 * x > 0 else 0

        for r in range(H):
            for c in range(avg_no):
                y[r, c] = line_seg(c, 127 - r)

            y[r, avg_no:] = y[r, avg_no - 1::-1]

        return clique_indexes, y

    # Generate checkboard data
    def checkboardHelper_shuffle(self):
        H = Options.H
        W = Options.W

        # create checkboard data (clique and ground_truth y)
        clique_indexes = np.zeros([H, W], dtype=np.int32, order='C')  # mapping of variables to clique_indexes
        _cliqueID = 1.0  # clique index starts from 1
        for _rowIndx in range(0, H, Options.gridStep):
            for _colIndx in range(0, W, Options.gridStep):
                clique_indexes[_rowIndx:_rowIndx + Options.gridStep,
                _colIndx:_colIndx + Options.gridStep] = _cliqueID
                _cliqueID += 1.0

        y = np.zeros([H, W], dtype=np.int32, order='C')  # ground-truth labels
        cliques_row_no = Options.W // Options.gridStep
        black_indexes = np.floor(np.linspace(0, Options.gridStep, cliques_row_no))
        for i in range(cliques_row_no):
            # np.random.shuffle(black_indexes)
            full_list = []

            for j in np.nditer(black_indexes):
                ind = int(j)
                assign_array = [1] * ind + [0] * (Options.gridStep - ind)
                full_list += assign_array
            full_list = np.asarray(full_list)

            y[i * Options.gridStep:(i + 1) * Options.gridStep, :] = full_list[np.newaxis, :]

        return clique_indexes, y

    # Generate checkboard data
    def checkboardHelper_gaussian_portions(self, portion_miu=(0.3, 0.9), sigma=0.05, is_gaussian=True):
        H = Options.H
        W = Options.W

        portions_num = len(portion_miu)
        avg_portion_cliques_n = Options.numCliques // portions_num
        last_portion_cliques_n = avg_portion_cliques_n + \
                                 Options.numCliques % portions_num

        # white labels portions of each clique
        clique_portions = np.zeros([Options.numCliques])
        for i in range(portions_num - 1):
            if is_gaussian:
                clique_portions[i * avg_portion_cliques_n:
                (i + 1) * avg_portion_cliques_n] = np.random.normal(portion_miu[i], sigma,
                                                                    avg_portion_cliques_n)
            else:
                clique_portions[i * avg_portion_cliques_n:
                (i + 1) * avg_portion_cliques_n] = portion_miu[i]
        clique_portions[-last_portion_cliques_n:] = \
            np.random.normal(portion_miu[-1], sigma, last_portion_cliques_n)

        # white labels' quantity in each clique
        clique_white_num_array = np.zeros([Options.numCliques], dtype=np.int)
        for i in range(Options.numCliques):
            if clique_portions[i] < 0:
                clique_portions[i] = 0
            elif clique_portions[i] > 1:
                clique_portions[i] = 1
            clique_white_num_array[i] = np.floor(Options.gridStep ** 2
                                                 * clique_portions[i])

        # create checkboard data (clique and ground_truth y)
        clique_indexes = np.zeros([H, W], dtype=np.int32, order='C')  # mapping of variables to clique_indexes
        _cliqueID = 1.0  # clique index starts from 1
        y = np.zeros([H, W], dtype=np.int32, order='C')  # ground-truth labels
        for _rowIndx in range(0, H, Options.gridStep):
            for _colIndx in range(0, W, Options.gridStep):
                white_labels_num = clique_white_num_array[int(_cliqueID) - 1]
                labels = np.array([0] * white_labels_num + \
                                  [1] * (Options.gridStep ** 2 - white_labels_num))
                np.random.shuffle(labels)
                y[_rowIndx:_rowIndx + Options.gridStep,
                _colIndx:_colIndx + Options.gridStep] = \
                    np.reshape(labels, [Options.gridStep, Options.gridStep])

                clique_indexes[_rowIndx:_rowIndx + Options.gridStep,
                _colIndx:_colIndx + Options.gridStep] = _cliqueID
                _cliqueID += 1.0

        return clique_indexes, y

    def init_unary_feature(self):
        W = Options.W
        H = Options.H
        # create unary features
        unary_observed = np.zeros([H, W, 2], dtype=np.double, order='C')
        for i in range(H):
            for j in range(W):
                unary_observed[i][j][1] = 2 * (np.random.rand(1, 1)[0, 0] - 0.5) + \
                                          self._eta[0] * (1 - self.y[i][j]) - self._eta[1] * self.y[i][j]
        return unary_observed

    def init_pairwise_feature(self):
        W = Options.W
        H = Options.H
        # create pairwise features
        pairwise = np.zeros([Options.rowsPairwise, 3], dtype=np.double, order='C')
        if (Options.hasPairwise):
            u = repmat(np.arange(W), H - 1, 1) * H + \
                repmat(np.arange(H - 1).reshape([H - 1, 1]), 1, W)
            v = repmat(np.arange(W - 1), H, 1) * H + \
                repmat(np.arange(H).reshape([H, 1]), 1, W - 1)

            for i, (u_i, v_i) in enumerate(zip(u.flatten('F'), v.flatten('F'))):
                pairwise[i, 0] = u_i
                pairwise[i, 1] = u_i + 1
                pairwise[i + u.shape[0] * u.shape[1], 0] = v_i
                pairwise[i + u.shape[0] * u.shape[1], 1] = v_i + H

        return pairwise


if __name__ == '__main__':
    instance = Instance('gaussian_portions', portion_miu=(0.1, 0.9))
    options = Options()
    for i in range(Options.H):
        for j in range(Options.W):
            print(instance.y[i, j], end=' ')
        print()
