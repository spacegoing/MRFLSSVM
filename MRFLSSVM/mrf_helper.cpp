//
// Created by spacegoing on 6/30/16.
//

#include "mrf_helper.h"
#include "Checkboard/Checkboard.h"
#include <iostream>


void copy_check_options(STRUCT_LEARN_PARM *sparm, Options *options);

inline int argmax_hidden_var(LATENT_VAR h);

SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm) {
    SAMPLE sample;
    Checkboard checkboard;
    sample.n = 1; // Only 1 example: the checkboard itself.
    sample.examples = (EXAMPLE *) malloc(sizeof(EXAMPLE) * sample.n);

    sample.examples[0].x.n_rows = checkboard.options.H;
    sample.examples[0].x.n_cols = checkboard.options.W;
    sample.examples[0].x.observed_unary = checkboard.mat_to_float_vec(checkboard.unary.slice(1));

    sample.examples[0].y.n_rows = checkboard.options.H;
    sample.examples[0].y.n_cols = checkboard.options.W;
    sample.examples[0].y.clique_indexes = checkboard.mat_to_std_vec(checkboard.cliques);
    sample.examples[0].y.ground_truth_label = checkboard.mat_to_std_vec(checkboard.y);

    sample.examples[0].h.n_rows = checkboard.options.K - 1;
    sample.examples[0].h.auxiliary_z = (int *) calloc(checkboard.options.K - 1, sizeof(int));

    copy_check_options(sparm, &checkboard.options);

    return sample;
}

SVECTOR *psi_helper(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
    int max_h = argmax_hidden_var(h); // max_h = max non zero index + 1

    // 1st higher order feature + number of non-zero higher-order features +
    // unary + pairwise + the terminate word (wnum = 0 & weight = 0 required by package)
    WORD words[1 + 2 * max_h + 1 + 1 + 1];

    // Calculate W(y)
    int clique_size_vector[sparm->options.numCliques]{0};
    float clique_value_vector[sparm->options.numCliques]{0.0};
    int clique_id = 0;
    float w_y = 0.0;
    for (int i = 0; i < y.n_rows; ++i) {
        for (int j = 0; j < y.n_cols; ++j) {
            clique_id = y.clique_indexes[i][j] - 1; // clique_indexes start from 1
            clique_size_vector[clique_id]++;
            clique_value_vector[clique_id] += y.ground_truth_label[i][j];
        }
    }
    for (int k = 0; k < sparm->options.numCliques; ++k) {
        if (clique_size_vector[k]) {
            clique_value_vector[k] /= clique_size_vector[k];
            w_y += clique_value_vector[k];
        }
    }

    // Assign higher order vector to words[]
    words[0].wnum = 1; // index has to start from 1, 0 used as terminate
    words[0].weight = w_y;

    // 1< l <= K
    for (int l = 1; l < max_h + 1; ++l) {
        words[l].wnum = l + 1;
        words[l].weight = w_y;
    }
    // K< l <= 2K - 1
    int counter = 1;
    for (int l = max_h + 1; l < 2 * max_h + 1; ++l) {
        words[l].wnum = sparm->options.K + counter;
        words[l].weight = 1;
        counter++;
    }


    //-------------------------------------------------
    // unary & pairwise features
    int unary_key = 1 + 2 * max_h;
    float unary_psi = 0;
    for (int i = 0; i < y.n_rows; ++i) {
        for (int j = 0; j < y.n_cols; ++j) {
            if (y.ground_truth_label[i][j] == 1)
                unary_psi += x.observed_unary[i][j];
        }
    }

    //todo: pariwise psi
    int pairwise_key = unary_key + 1;
    float pairwise_psi = 0.0;

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

inline int argmax_hidden_var(LATENT_VAR h) {
    // return number of non-zero hidden vars
    // notice the non-zero state of h is continuous
    int counter = 0;
    for (int i = 0; i < h.n_rows; ++i) {
        if (h.auxiliary_z[i] < 1) break;
        counter++;
    }

    return counter;
}


int main(int argc, char **argv) {

    // test read_example()
    STRUCT_LEARN_PARM sparm;
    SAMPLE sample = read_struct_examples_helper((char *) "", &sparm);

//     // print checkboard matrix value--------------------------------------------
//    for (int i = 0; i < sample.examples[0].y.n_rows; ++i) {
//        for (int j = 0; j < sample.examples[0].y.n_cols; ++j) {
//            std::cout << sample.examples[0].y.clique_indexes[i][j];
//        }
//        std::cout << "\n";
//    }

    // test psi()---------------------------------------------------------------
    EXAMPLE example0 = sample.examples[0];
    PATTERN x0 = example0.x;
    LABEL y0 = example0.y;
    LATENT_VAR h0 = example0.h;
    STRUCTMODEL *sm0;
//    h0.auxiliary_z[0] = 1;
//    h0.auxiliary_z[1] = 1;
//    h0.auxiliary_z[2] = 1;

    SVECTOR *svec0 = psi_helper(x0, y0, h0, sm0, &sparm);
    WORD *index = svec0->words;
    while (index->wnum) {
        cout << "Key: " << index->wnum << ", Value: " << index->weight << "\n";
        index++;
    }



    return 0;
}