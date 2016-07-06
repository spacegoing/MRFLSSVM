//
// Created by spacegoing on 6/30/16.
//

#include "mrf_helper.h"
#include "Checkboard/Checkboard.h"
#include <iostream>

void copy_check_options(STRUCT_LEARN_PARM *sparm, Options *options);

SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm) {
    SAMPLE sample;
    Checkboard checkboard;
    sample.n = 1; // Only 1 example: the checkboard itself.
    sample.examples = (EXAMPLE *) malloc(sizeof(EXAMPLE) * sample.n);

    sample.examples[0].x.n_rows = checkboard.options.H;
    sample.examples[0].x.n_cols = checkboard.options.W;
    sample.examples[0].x.observed_label = checkboard.mat_to_std_vec(checkboard.unary.slice(1));

    sample.examples[0].y.n_rows = checkboard.options.H;
    sample.examples[0].y.n_cols = checkboard.options.W;
    sample.examples[0].y.clique_indexes = checkboard.mat_to_std_vec(checkboard.cliques);
    sample.examples[0].y.ground_truth_label = checkboard.mat_to_std_vec(checkboard.y);

    sample.examples[0].h.n_rows = checkboard.options.K-1;
    sample.examples[0].h.auxiliary_z = (int *) calloc(checkboard.options.K-1, sizeof(int));

    copy_check_options(sparm, &checkboard.options);

    return sample;
}

void copy_check_options(STRUCT_LEARN_PARM *sparm, Options *options){
    sparm->options.gridStep=options->gridStep;  // grid size for defining cliques
    sparm->options.W=options->W; // image width
    sparm->options.H=options->H; // image height
    sparm->options.numCliques=options->numCliques; // number of cliques
    sparm->options.N=options->N; // number of variables

    sparm->options.dimUnary=options->dimUnary;
    sparm->options.dimPairwise=options->dimPairwise;

    sparm->options.K=options->K;  // number of lower linear functions
    sparm->options.maxIters=options->maxIters;  // maximum learning iterations
    sparm->options.eps=options->eps;  // constraint violation threshold

    sparm->options.learningQP=options->learningQP;  // encoding for learning QP (1, 2, or 3)
    sparm->options.figWnd=options->figWnd;  // figure for showing results

}


int main(int argc, char **argv) {
    STRUCT_LEARN_PARM sparm;
    SAMPLE sample = read_struct_examples_helper("", &sparm);

    for (int i = 0; i < sample.examples[0].y.n_rows; ++i) {
        for (int j = 0; j < sample.examples[0].y.n_cols; ++j) {
            std::cout<<sample.examples[0].y.clique_indexes[i][j];
        }
        std::cout<<"\n";
    }

    return 0;
}