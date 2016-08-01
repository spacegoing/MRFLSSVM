//
// Created by spacegoing on 7/30/16.
//
#include "GraphCut.h"
#include <stdio.h>
#include <algorithm>
#include <iostream>

#define DEBUG_LEVEL 13
/**
 * -1 all
 * 0  none
 * 10 z_index
 * 11 unary
 * 12 pairwise & cliques_index
 * 13 weights
 */

using namespace std;

double graph_cut_method(float *observed_unary, float *pairwise,
                        int *clique_indexes,
                        int *inferred_label, int *inferred_z,
                        double *w, OPTIONS options) {
    // The valid index of sm->w starts from 1. Thus the length is sm.sizePsi+1
    int nVariables = options.rows * options.cols;
    int K = options.K;
//    todo: multiple cliques
//    int nMaxCliquesPerVariable = 1;

    // remove redundancy-------------------------------------------------------
    // compute (b_k-b_{k-1})/a_{k-1}-a_k
//    double b_a_ratio[K - 1];
//    for (int i = 0; i < K - 1; ++i) {
//        b_a_ratio[i] = w[i + K] / (-1.0 * w[i + 1]);
//    }
//
//    // ensure (b_k-b_{k-1})/a_{k-1}-a_k < (b_{k+1}-b_{k})/a_k-a_{k+1}
//    // if not then use w[i-1] & w[i+K-1] replace w[i] & w[i+K]
//    for (int i = 1; i < K - 1; ++i) {
//        if (b_a_ratio[i] < b_a_ratio[i - 1]) {
//            w[i + 1] = w[i];
//            w[i + K] = w[i + K - 1];
//        }
//    }

    float *unaryWeights = observed_unary;

    typedef Graph<double, double, double> GraphType;
    GraphType *g = new GraphType(nVariables, 8 * nVariables);
    g->add_node(nVariables);

    // Add unary edges------------------------------------------------------------
#if ((DEBUG_LEVEL == 11) || (DEBUG_LEVEL == -1))
    printf("inspect unary edges: inspect unary edges: inspect unary edges: inspect unary edges: \n");
#endif
    int counter = 0;
    for (int i = 0; i < options.rows; ++i) {
        for (int j = 0; j < options.cols; ++j) {
            g->add_tweights(counter, unaryWeights[i * options.cols * 2 + j * 2],
                            unaryWeights[i * options.cols * 2 + j * 2 + 1]);
#if ((DEBUG_LEVEL == 11) || (DEBUG_LEVEL == -1))
            printf("%f ", unaryWeights[i * options.cols * 2 + j * 2 + 1]);
#endif
            counter++;
        }
#if ((DEBUG_LEVEL == 11) || (DEBUG_LEVEL == -1))
        printf("\n");
#endif
    }

    // Add pairwise edges
    if (options.n_pairwise_rows) {
        printf("pairlen: %d", options.n_pairwise_rows);
        for (int i = 0; i < options.n_pairwise_rows; ++i) {
            g->add_edge((int) pairwise[i * 3], (int) pairwise[i * 3 + 1], pairwise[i * 3 + 2],
                        pairwise[i * 3 + 2]);
#if ((DEBUG_LEVEL == 12) || (DEBUG_LEVEL == -1))
            printf("%d: %d %d %f\n", i, (int) pairwise[i * 3], (int) pairwise[i * 3 + 1],
                   pairwise[i * 3 + 2]);
#endif
        }
    }

    // Add higher-order terms (a,z between s and t)--------------------------------

    // Add auxiliary vars z for each clique
#if ((DEBUG_LEVEL == 10) || (DEBUG_LEVEL == -1))
    printf("inspect auxiliary vars: inspect auxiliary vars: inspect auxiliary vars: ");
#endif
    int z_index[options.numCliques];
    for (int k = 0; k < options.numCliques; ++k) {
        z_index[k] = g->add_node(K - 1);
#if((DEBUG_LEVEL == 10) || (DEBUG_LEVEL == -1))
        cout << z_index[k] << " ";
#endif
    }

#if ((DEBUG_LEVEL == 12) || (DEBUG_LEVEL == -1))
    for (int m = 0; m < options.rows; ++m) {
        for (int i = 0; i < options.cols; ++i) {
            printf("%d", clique_indexes[m * options.cols + i]);
        }
        printf("\n");
    }
#endif

    // compute clique size
    int *clique_size = (int *) calloc(options.numCliques, sizeof(int));
    for (int l = 0; l < options.rows; ++l) {
        for (int i = 0; i < options.cols; ++i) {
            // clique_index starts from 1
            clique_size[clique_indexes[l * options.cols + i] - 1] += 1;
        }
    }
#if ((DEBUG_LEVEL == 12) || (DEBUG_LEVEL == -1))
    printf("\n");
    for (int m = 0; m < options.numCliques; ++m) {
        printf("%d ", clique_size[m]);
    }
    printf("\n");
#endif

    // Add higher-order terms for each clique (w_i = 1/cliqueSize)
    // Edges between y_i z_k and y_i t
    // Edges between z_k and s and t
    // todo: nMaxCliquesPerVariable>1
    counter = 0;
    for (int i = 0; i < options.rows; ++i) {
        for (int j = 0; j < options.cols; ++j) {
            int clique_index = clique_indexes[i * options.cols + j] - 1;
            double w_i = 1.0 / clique_size[clique_index];

            // edge between y_i and t
            g->add_tweights(counter, 0.0, w[0] * w_i);

            for (int k = 0; k < K - 1; ++k) {
                // edge between y_i and z_k
                // in w, w[0] element is a1, the followings are a_k+1 - a_k
                // to w[K-1]. So the edge should be -w[k+1]
                g->add_edge(counter, z_index[clique_index] + k,
                            0.0, w_i * -1.0 * w[k + 1]);
#if ((DEBUG_LEVEL == 13) || (DEBUG_LEVEL == -1))
                printf("nodeID: %d, axID: %d, value: %f\n", counter,
                       z_index[clique_index] + k, w_i * -1.0 * w[k + 1]);
#endif
            }
            counter++;
        }
    }

    for (int i = 0; i < options.numCliques; ++i) {
        for (int k = 0; k < K - 1; ++k) {
            // edge between z_k and s and t
            g->add_tweights(z_index[i] + k, -1.0 * w[k + 1], w[K + k]);
#if ((DEBUG_LEVEL == 13) || (DEBUG_LEVEL == -1))
            printf("nodeID: %d, da: %f, db: %f\n",
                   z_index[i] + k, -1.0 * w[k + 1], w[K + k]);
#endif
        }
    }


    double e = g->maxflow();

    for (int m = 0; m < nVariables; ++m) {
        inferred_label[m] = (g->what_segment(m) == GraphType::SOURCE) ? 1 : 0;
    }

    int idx=0;
    for (int n = 0; n < options.numCliques; ++n) {
        for (int i = 0; i < K - 1; ++i) {
            inferred_z[idx] = (g->what_segment(z_index[n] + i) == GraphType::SOURCE) ? 1 : 0;
            idx++;
        }
    }

//    int row = 0;
//    int col = 0;
//    for (int i1 = 0; i1 < nVariables; ++i1) {
//        if (col == options.cols) {
//            col = 0;
//            row++;
//        }
//        inferred_label[row * options.cols + col] = (g->what_segment(i1) == GraphType::SOURCE) ? 1 : 0;
//        col++;
//    }
//
//    for (int l1 = 0; l1 < options.numCliques; ++l1) {
//        for (int k1 = 0; k1 < K - 1; ++k1) {
//            inferred_z[l1 * (K - 1) + k1] = (g->what_segment(z_index[l1] + k1) == GraphType::SOURCE) ? 1 : 0;
//        }
//    }

#if ((DEBUG_LEVEL == 2) || (DEBUG_LEVEL == -1))
    cout << "ybar##############################" << endl;
    for (int m1 = 0; m1 < options.rows; ++m1) {
        for (int i = 0; i < options.cols; ++i) {
            cout << inferred_label[m1 * options.cols + i] << " ";
        }
        cout << endl;
    }

    cout << "hbar##############################" << endl;
    for (int m1 = 0; m1 < options.numCliques; ++m1) {
        for (int i = 0; i < K - 1; ++i) {
            cout << inferred_z[m1 * (K - 1) + i] << " ";
        }
        cout << endl;
    }
#endif

    delete (g);

    return e;
}