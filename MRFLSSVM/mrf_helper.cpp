//
// Created by spacegoing on 6/30/16.
//

#include "mrf_helper.h"
#include "Checkboard/Checkboard.h"
#include "maxflow-v3.03.src/graph.h"
#include <iostream>
#include <stdio.h>

#define DEBUG_LEVEL 1

/**
 * -1 for all
 * 0 for null
 * 1 for psi
 * 2 for graph-cut
 */

void copy_check_options(STRUCT_LEARN_PARM *sparm, Options *options);

inline int *argmax_hidden_var(LATENT_VAR h);

void write_pairwise(float **pairwise, int n_rows) {
/*
  Writes the learned weight vector sm->w to file after training.
*/
    FILE *modelfl;
    long i;

    modelfl = fopen("pairwise.txt", "w");
    if (modelfl == NULL) {
        printf("Cannot open model file %s for output!", "pairwise.txt");
        exit(1);
    }

//    fprintf(modelfl, "# sizePsi:%ld\n", sm->sizePsi);
    for (i = 0; i < n_rows; i++) {
        fprintf(modelfl, "%f, %f, %f\n", pairwise[i][0] + 1, pairwise[i][1] + 1, pairwise[i][2]);
    }
    fclose(modelfl);

}

SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm) {
    SAMPLE sample;
    Checkboard checkboard;
    sample.n = 1; // Only 1 example: the checkboard itself.
    sample.examples = (EXAMPLE *) malloc(sizeof(EXAMPLE) * sample.n);

    sample.examples[0].x.n_rows = checkboard.options.H;
    sample.examples[0].x.n_cols = checkboard.options.W;
    sample.examples[0].x.dim_unary = checkboard.options.dimUnary;
    sample.examples[0].x.dim_pairwise = checkboard.options.dimPairwise;
    sample.examples[0].x.observed_unary = checkboard.cube_to_float(checkboard.unary);
    sample.examples[0].x.pairwise = checkboard.pairwise;

    sample.examples[0].y.n_rows = checkboard.options.H;
    sample.examples[0].y.n_cols = checkboard.options.W;
    sample.examples[0].y.clique_indexes = checkboard.mat_to_std_vec(checkboard.cliques);
    sample.examples[0].y.ground_truth_label = checkboard.mat_to_std_vec(checkboard.y);

    sample.examples[0].h.n_rows = checkboard.options.numCliques;
    sample.examples[0].h.n_cols = checkboard.options.K - 1;
    sample.examples[0].h.auxiliary_z = (int **) calloc(checkboard.options.numCliques, sizeof(int *));
    for (int i = 0; i < checkboard.options.numCliques; ++i)
        sample.examples[0].h.auxiliary_z[i] = (int *) calloc(checkboard.options.K - 1, sizeof(int));

    copy_check_options(sparm, &checkboard.options);

    int dim_pariwise = sparm->options.H * sparm->options.W * 2
                       - sparm->options.H - sparm->options.W;
    write_pairwise(checkboard.pairwise, dim_pariwise);

    return sample;
}

SVECTOR *psi_helper(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
    int *argmax_z_array = argmax_hidden_var(h); // numCliques X K-1

    // 1st higher order feature + number of non-zero higher-order features +
    // unary + pairwise + the terminate word (wnum = 0 & weight = 0 required by package)
    int higher_order_length = 2 * sparm->options.K - 1;
    WORD words[higher_order_length + 1 + 1 + 1];

    // Calculate W(y)-------------------------------------------------------------------
    int *clique_size_vector = (int *) calloc(sparm->options.numCliques, sizeof(int));
    float *clique_value_vector = (float *) calloc(sparm->options.numCliques, sizeof(float));
    int clique_id = 0;
    float w_y = 0.0;
    for (int i = 0; i < y.n_rows; ++i) {
        for (int j = 0; j < y.n_cols; ++j) {
            clique_id = y.clique_indexes[i][j] - 1; // clique_indexes start from 1
            clique_size_vector[clique_id]++;
            clique_value_vector[clique_id] += y.ground_truth_label[i][j];
        }
    }

#if ((DEBUG_LEVEL == 1) || (DEBUG_LEVEL == -1))
    cout << "clique value vector\n$$$$$$$$$$$$$$$$$$$$$$$#############" << endl;
    for (int n = 0; n < sparm->options.numCliques; ++n) {
        cout << clique_value_vector[n] << " ";
    }
    cout << endl;
#endif

    for (int k = 0; k < sparm->options.numCliques; ++k) {
        if (clique_size_vector[k]) {
            clique_value_vector[k] /= clique_size_vector[k];
            w_y += clique_value_vector[k];
        }
    }
#if ((DEBUG_LEVEL == 1) || (DEBUG_LEVEL == -1))
    cout << "clique value vector\n$$$$$$$$$$$$$$$$$$$$$$$#############" << endl;
    for (int n = 0; n < sparm->options.numCliques; ++n) {
        cout << clique_value_vector[n] << " ";
    }
    cout << endl;
#endif

    // Assign higher order vector to words[]--------------------------------------------
    for (int l = 0; l < higher_order_length; ++l) {
        words[l].wnum = l + 1;
        words[l].weight = 0;
    }

    words[0].weight = w_y;

    for (int m = 0; m < sparm->options.numCliques; ++m) {
        // 1< l <= K
        for (int l = 1; l < argmax_z_array[m] + 1; ++l) {
            words[l].weight += clique_value_vector[m];
        }
        // K< l <= 2K - 1
        for (int l = sparm->options.K; l < sparm->options.K + argmax_z_array[m]; ++l) {
            words[l].weight += clique_value_vector[m];
        }
    }

    //-------------------------------------------------------------------------
    // unary & pairwise features

    // calculate unary psi
    int unary_key = sparm->options.K * 2 - 1;
    float unary_psi = 0;
    for (int i = 0; i < y.n_rows; ++i) {
        for (int j = 0; j < y.n_cols; ++j) {
            if (y.ground_truth_label[i][j] == 1)
                unary_psi += x.observed_unary[i][j][x.dim_unary - 1];
        }
    }
#if ((DEBUG_LEVEL == 1) || (DEBUG_LEVEL == -1))
    FILE *modelfl;

    modelfl = fopen("pairwise.txt", "w");
    if (modelfl == NULL) {
        printf("Cannot open model file %s for output!", "pairwise.txt");
        exit(1);
    }
#endif
    // calculate pariwise psi
    int pairwise_key = unary_key + 1;
    float pairwise_psi = 0.0;
    if (sparm->options.dimPairwise) {
        int dim_pariwise = sparm->options.H * sparm->options.W * 2
                           - sparm->options.H - sparm->options.W;
        for (int i = 0; i < dim_pariwise; ++i) {
            div_t ind0 = div((int) x.pairwise[i][0], sparm->options.H);
            div_t ind1 = div((int) x.pairwise[i][1], sparm->options.H);

            if (y.ground_truth_label[ind0.rem][ind0.quot] !=
                y.ground_truth_label[ind1.rem][ind1.quot]) {
                pairwise_psi += 1;
#if ((DEBUG_LEVEL == 1) || (DEBUG_LEVEL == -1))
                fprintf(modelfl, "%d\n", i);
#endif
            }
        }
    }
#if ((DEBUG_LEVEL == 1) || (DEBUG_LEVEL == -1))
    fclose(modelfl);
#endif


    // Assign unary & pairwise
    // Notice: the wnum of unary & pairwise are always
    // options.K*2 and options.K*2+1
    words[unary_key].wnum = sparm->options.K * 2;
    words[unary_key].weight = unary_psi;
    words[pairwise_key].wnum = sparm->options.K * 2 + 1;
    words[pairwise_key].weight = pairwise_psi;

    // Add the 0 term as termination of words[] (required by library)
    words[pairwise_key + 1].wnum = 0;
    words[pairwise_key + 1].weight = 0;


    return create_svector(words, (char *) "", 1);
}

LATENT_VAR infer_latent_variables_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
    LATENT_VAR h;
    h.n_rows = sparm->options.numCliques;
    h.n_cols = sparm->options.K - 1;
    h.auxiliary_z = (int **) calloc(h.n_rows, sizeof(int *));
    for (int i = 0; i < h.n_rows; ++i) {
        h.auxiliary_z[i] = (int *) malloc(h.n_cols * sizeof(int));
        for (int j = 0; j < h.n_cols; ++j) {
            h.auxiliary_z[i][j] = (-1.0 * sm->w[j + 1] >= sm->w[sparm->options.K + j]) ? 1 : 0;
        }
    }
    return h;
}

void find_most_violated_constraint_marginrescaling_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar,
                                                          STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
    // The valid index of sm->w starts from 1. Thus the length is sm.sizePsi+1
    int nVariables = sparm->options.H * sparm->options.W;
    int K = sparm->options.K;
//    todo: multiple cliques
//    int nMaxCliquesPerVariable = 1;
    double *w = sm->w;

    // remove redundancy-------------------------------------------------------
    // compute (b_k-b_{k-1})/a_{k-1}-a_k
    double b_a_ratio[K - 1];
    for (int i = 0; i < K - 1; ++i) {
        b_a_ratio[i] = w[i + K] / (-1.0 * w[i + 1]);
    }

    // ensure (b_k-b_{k-1})/a_{k-1}-a_k < (b_{k+1}-b_{k})/a_k-a_{k+1}
    // if not then use w[i-1] & w[i+K-1] replace w[i] & w[i+K]
    for (int i = 1; i < K - 1; ++i) {
        if (b_a_ratio[i] < b_a_ratio[i - 1]) {
            w[i + 1] = w[i];
            w[i + K] = w[i + K - 1];
        }
    }

    float ***unaryWeights = x.observed_unary;

    typedef Graph<double, double, double> GraphType;
    GraphType *g = new GraphType(nVariables, 8 * nVariables);
    g->add_node(nVariables);

    // Add unary edges------------------------------------------------------------
    int counter = 0;
    for (int i = 0; i < x.n_rows; ++i) {
        for (int j = 0; j < x.n_cols; ++j) {
            g->add_tweights(counter, unaryWeights[i][j][0], unaryWeights[i][j][1]);
            counter++;
        }
    }

    // Add pairwise edges
    if (sparm->options.dimPairwise) {
        int dim_pariwise = sparm->options.H * sparm->options.W * 2
                           - sparm->options.H - sparm->options.W;
        for (int i = 0; i < dim_pariwise; ++i) {
            g->add_edge((int) x.pairwise[i][0], (int) x.pairwise[i][1], x.pairwise[i][2], x.pairwise[i][2]);
        }
    }

    // Add higher-order terms (a,z between s and t)--------------------------------

    // Add auxiliary vars z for each clique
#if (DEBUG_LEVEL==10)
    printf("pfweijpawjefpajifpoaiwejfopaijeopfijaoepjfpaje");
#endif
    int z_index[sparm->options.numCliques];
    for (int k = 0; k < sparm->options.numCliques; ++k) {
        z_index[k] = g->add_node(K - 1);
#if(DEBUG_LEVEL==10)
        cout<<z_index[k]<<" ";
#endif
    }

    // compute clique size
    int *clique_size = (int *) calloc(sparm->options.numCliques, sizeof(int));
    for (int l = 0; l < y.n_rows; ++l) {
        for (int i = 0; i < y.n_cols; ++i) {
            // clique_index starts from 1
            clique_size[y.clique_indexes[l][i] - 1] += 1;
        }
    }

    // Add higher-order terms for each clique (w_i = 1/cliqueSize)
    // Edges between y_i z_k and y_i t
    // Edges between z_k and s and t
    // todo: nMaxCliquesPerVariable>1
    counter = 0;
    for (int i = 0; i < y.n_rows; ++i) {
        for (int j = 0; j < y.n_cols; ++j) {
            int clique_index = y.clique_indexes[i][j] - 1;
            double w_i = 1.0 / clique_size[clique_index];

            // edge between y_i and t
            g->add_tweights(counter, 0.0, w[0] * w_i);

            for (int k = 0; k < K - 1; ++k) {
                // edge between y_i and z_k
                // in w, w[0] element is a1, the followings are a_k+1 - a_k
                // to w[K-1]. So the edge should be -w[k+1]
                g->add_edge(counter, z_index[clique_index] + k,
                            0.0, w_i * -1.0 * w[k + 1]);

                // edge between z_k and s and t
                g->add_tweights(z_index[clique_index] + k,
                                w_i * -1.0 * w[k + 1], w[K + k]);
            }
            counter++;
        }
    }


    ybar->ground_truth_label = (int **) malloc(sparm->options.H * sizeof(int *));
    ybar->clique_indexes = (int **) malloc(sparm->options.H * sizeof(int *));
    for (int m = 0; m < sparm->options.H; ++m) {
        ybar->ground_truth_label[m] = (int *) malloc(sparm->options.W * sizeof(int));
        ybar->clique_indexes[m] = (int *) malloc(sparm->options.W * sizeof(int));
    }
    ybar->n_rows = y.n_rows;
    ybar->n_cols = y.n_cols;

    hbar->auxiliary_z = (int **) malloc(sparm->options.numCliques * sizeof(int *));
    for (int n = 0; n < sparm->options.numCliques; ++n)
        hbar->auxiliary_z[n] = (int *) malloc((K - 1) * sizeof(int));
    hbar->n_rows = sparm->options.numCliques;
    hbar->n_cols = sparm->options.K - 1;

    double e = g->maxflow();

    int row = 0;
    int col = 0;
    for (int i1 = 0; i1 < nVariables; ++i1) {
        if (col == y.n_cols) {
            col = 0;
            row++;
        }
        ybar->ground_truth_label[row][col] = (g->what_segment(i1) == GraphType::SOURCE) ? 1 : 0;
        ybar->clique_indexes[row][col] = y.clique_indexes[row][col];
        col++;
    }

    for (int l1 = 0; l1 < sparm->options.numCliques; ++l1) {
        for (int k1 = 0; k1 < K - 1; ++k1) {
            hbar->auxiliary_z[l1][k1] = (g->what_segment(z_index[l1] + k1) == GraphType::SOURCE) ? 1 : 0;
        }
    }

#if ((DEBUG_LEVEL == 2) || (DEBUG_LEVEL==-1))
    cout << "ybar##############################" << endl;
    for (int m1 = 0; m1 < y.n_rows; ++m1) {
        for (int i = 0; i < y.n_cols; ++i) {
            cout << ybar->ground_truth_label[m1][i] << " ";
        }
        cout << endl;
    }

    cout << "hbar##############################" << endl;
    for (int m1 = 0; m1 < hbar->n_rows; ++m1) {
        for (int i = 0; i < hbar->n_cols; ++i) {
            cout << hbar->auxiliary_z[m1][i] << " ";
        }
        cout << endl;
    }
#endif
    delete (g);
}

void copy_check_options(STRUCT_LEARN_PARM *sparm, Options *options) {
    sparm->options.gridStep = options->gridStep;  // grid size for defining cliques
    sparm->options.W = options->W; // image width
    sparm->options.H = options->H; // image height
    sparm->options.numCliques = options->numCliques; // number of cliques
    sparm->options.N = options->N; // number of variables

    sparm->options.dimUnary = options->dimUnary;
    sparm->options.dimPairwise = options->dimPairwise;

    sparm->options.K = options->K;  // number of lower linear functions
    sparm->options.maxIters = options->maxIters;  // maximum learning iterations
    sparm->options.eps = options->eps;  // constraint violation threshold

    sparm->options.learningQP = options->learningQP;  // encoding for learning QP (1, 2, or 3)
    sparm->options.figWnd = options->figWnd;  // figure for showing results

}

inline int *argmax_hidden_var(LATENT_VAR h) {
    // return number of non-zero hidden vars
    // notice the non-zero state of h is continuous
    int counter = 0;
    int *argmax_z_array = (int *) malloc(h.n_rows * sizeof(int));
    for (int i = 0; i < h.n_rows; ++i) { // numCliques
        for (int j = 0; j < h.n_cols; ++j) {
            if (h.auxiliary_z[i][j] < 1) break;
            counter++;
        }
        argmax_z_array[i] = counter;
        counter = 0;
    }

    return argmax_z_array;
}

//int main(int argc, char **argv) {
//
//    // test read_example()
//    STRUCT_LEARN_PARM sparm;
//    SAMPLE sample = read_struct_examples_helper((char *) "", &sparm);
//
//    EXAMPLE example0 = sample.examples[0];
//    PATTERN x0 = example0.x;
//    LABEL y0 = example0.y;
//    LATENT_VAR h0 = example0.h;
//    STRUCTMODEL *sm0;
//
////    // calculate [i][j] from index
////    float a = 130.0;
////    div_t q = div((int) a, 128);
////    cout << "r: " << q.rem << " q: " << q.quot << endl;
//
////    // Check pairwise
////    cout<<"127th: "<<x0.pairwise[126][0]<<" "<<x0.pairwise[126][1]<<"\n";
////    cout<<"128th: "<<x0.pairwise[127][0]<<" "<<x0.pairwise[127][1]<<"\n";
////    cout<<"1015th: "<<x0.pairwise[1014][0]<<" "<<x0.pairwise[1014][1]<<"\n";
////    cout<<"1016th: "<<x0.pairwise[1015][0]<<" "<<x0.pairwise[1015][1]<<"\n";
////    cout<<"1017th: "<<x0.pairwise[1016][0]<<" "<<x0.pairwise[1016][1]<<"\n";
////    cout<<"16255th: "<<x0.pairwise[16254][0]<<" "<<x0.pairwise[16254][1]<<"\n";
////    cout<<"16256th: "<<x0.pairwise[16255][0]<<" "<<x0.pairwise[16255][1]<<"\n";
////    cout<<"16257th: "<<x0.pairwise[16256][0]<<" "<<x0.pairwise[16256][1]<<"\n";
////    cout<<"32512th: "<<x0.pairwise[32511][0]<<" "<<x0.pairwise[32511][1]<<"\n";
//
////     // print checkboard matrix value--------------------------------------------
////    for (int i = 0; i < sample.examples[0].y.n_rows; ++i) {
////        for (int j = 0; j < sample.examples[0].y.n_cols; ++j) {
////            std::cout << sample.examples[0].y.clique_indexes[i][j];
////        }
////        std::cout << "\n";
////    }
//
////    for (int i = 0; i < x0.n_rows; ++i) {
////        for (int j = 0; j < x0.n_cols; ++j) {
////            cout << x0.observed_unary[i][j][0] << ":" << x0.observed_unary[i][j][1] << " ";
////        }
////        cout<<"\n";
////    }
//
//
//    // test psi()---------------------------------------------------------------
////    h0.auxiliary_z[1][0] = 1;
////    h0.auxiliary_z[1][1] = 1;
////    h0.auxiliary_z[1][2] = 1;
////    for (int i = 0; i < h0.n_rows; ++i) {
////        for (int j = 0; j < h0.n_cols; ++j) {
////            h0.auxiliary_z[i][j] = 1;
////        }
////    }
////
////    SVECTOR *svec0 = psi_helper(x0, y0, h0, sm0, &sparm);
////    WORD *index = svec0->words;
////    while (index->wnum) {
////        cout << "Key: " << index->wnum << ", Value: " << index->weight << "\n";
////        index++;
////    }
//
//    // test psi() with / without pariwise---------------------------------------
//
//
//    return 0;
//}