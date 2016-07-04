//
// Created by spacegoing on 6/30/16.
//

#include "mrf_helper.h"
#include "Checkboard/Checkboard.h"

struct LSSVM {
    Checkboard checkboard;
    int unaryWeight = 1;
    int pairwiseWeight = 0;
    int featureLength = 2 * checkboard.options.K - 1;
    mat linEnvCoeffs = zeros<mat>(featureLength, featureLength);
};


SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm) {
    SAMPLE sample;
    Checkboard checkboard;
    sample.n = 1; // Only 1 example: the checkboard itself.
    sample.examples = (EXAMPLE *) malloc(sizeof(EXAMPLE) * sample.n);

    sample.examples[0].x.row_no = checkboard.options.H;
    sample.examples[0].x.col_no = checkboard.options.W;
    sample.examples[0].x.observed_label = &checkboard.mat_to_std_vec(checkboard.unary.slice(1))[0][0];


    return sample;
}

