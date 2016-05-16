#ifdef __cplusplus
extern "C" {
#endif
#include "svm_struct_latent_api_types.h"

void classify_struct_example_helper(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

void find_most_violated_constraint_marginrescaling_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

LATENT_VAR infer_latent_variables_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm);

double helper_vilain(TOM_LABEL y, TOM_LABEL ybar, STRUCT_LEARN_PARM *param);

double helper_joachims(TOM_LABEL y, TOM_LABEL ybar, STRUCT_LEARN_PARM *param);

TOM_LABEL label2tomlabel(LABEL y);

void free_tomlabel(TOM_LABEL ty);

#ifdef __cplusplus
}
#endif
