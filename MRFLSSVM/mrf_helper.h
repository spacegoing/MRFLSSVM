//
// Created by spacegoing on 6/30/16.
//

#ifndef LATENTSSVM_V0_12_MRF_HELPER_H
#define LATENTSSVM_V0_12_MRF_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif


#include "svm_struct_latent_api_types.h"

SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm);

SVECTOR *psi_helper(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

//void classify_struct_example_helper(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm,
//                                    STRUCT_LEARN_PARM *sparm);

void find_most_violated_constraint_marginrescaling_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar,
                                                          STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

LATENT_VAR infer_latent_variables_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);


#ifdef __cplusplus
}
#endif

#endif //LATENTSSVM_V0_12_MRF_HELPER_H
