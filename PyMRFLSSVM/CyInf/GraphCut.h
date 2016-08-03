//
// Created by spacegoing on 7/30/16.
//

#ifndef CYINF_GRAPHCUT_H
#define CYINF_GRAPHCUT_H

#include "maxflow-v3.03.src/graph.h"

typedef struct options {
    // Image Configs
    int rows; // image height (checkboard rows)
    int cols; // image width (checkboard cols)
    int numCliques; // number of cliques

    // Learning Configs
    int K;  // number of lower linear functions
    int n_pairwise_rows;

    // Other Configs
    int learningQP;  // encoding for learning QP (1, 2, or 3)
} OPTIONS;

double graph_cut_method(double *observed_unary, double *pairwise,
                        int *clique_indexes,
                        int *inferred_label, int *inferred_z,
                        double *w, OPTIONS options);

#endif //CYINF_GRAPHCUT_H
