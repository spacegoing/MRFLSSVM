from cython.operator cimport dereference as deref, preincrement as inc
# cython: boundscheck=False
# cython: wraparound=False

cdef extern from "/Users/spacegoing/macCodeLab-MBP2015/Python/MRFLSVM/PyMRFLSSVM/CyInf/GraphCut.h":
    ctypedef struct OPTIONS:
        # Image Configs
        int cols;  # image width (checkboard cols)
        int rows;  # image height (checkboard rows)
        int numCliques;  # number of clique_indexes

        # Learning Configs
        int K;  # number of lower linear functions
        int n_pairwise_rows;

        # Other Configs
        int learningQP;  # encoding for learning QP (1, 2, or 3)

    double graph_cut_method(double *observed_unary, double * pairwise,
                            int * clique_indexes,
                            int * inferred_label, int * inferred_z,
                            double *w, OPTIONS options)

def Inf_Algo(double[:,:,::1]observed_unary, double [:,::1]pairwise,
                            int [:,::1]clique_indexes,
                            int [:,::1]inferred_label, int [:,::1]inferred_z,
                            double [:]theta, options):
    # w only contains parameters for higher order terms

    cdef OPTIONS c_options

    c_options.rows = options.H
    c_options.cols = options.W
    c_options.numCliques = options.numCliques
    c_options.K = options.K

    c_options.n_pairwise_rows = pairwise.shape[0]
    c_options.learningQP = options.learningQP

    return graph_cut_method(&observed_unary[0,0,0], &pairwise[0,0], &clique_indexes[0,0],
                            &inferred_label[0,0], &inferred_z[0,0], &theta[0], c_options)