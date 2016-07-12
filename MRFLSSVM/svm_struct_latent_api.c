/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include "svm_struct_latent_api_types.h"
#include "mrf_helper.h"


SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in 
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialized in this function or by calling init_latent_variables(). 
*/
    return read_struct_examples_helper(file, sparm);
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm,
                       KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

    sm->sizePsi = 2 * sparm->options.K + 1; /* replace with appropriate number */

    /* your code here*/

}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/

    /* your code here */
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
    return psi_helper(x, y, h, sm, sparm);
}

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/

    /* your code here */
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm,
                                                   STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/

    /* your code here */
    find_most_violated_constraint_marginrescaling_helper(x, y, ybar, hbar, sm, sparm);
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

    return infer_latent_variables_helper(x, y, sm, sparm);
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/
    double ans = 0;

    /* your code here */
    int *clique_size_vector = (int *) calloc(sparm->options.numCliques, sizeof(int));
    int clique_id = 0;
    for (int i = 0; i < y.n_rows; ++i) {
        for (int j = 0; j < y.n_cols; ++j) {
            clique_id = y.clique_indexes[i][j] - 1; // clique_indexes start from 1
            clique_size_vector[clique_id]++;
        }
    }

    for (int i = 0; i < y.n_rows; ++i) {
        for (int j = 0; j < y.n_cols; ++j) {
            if (y.ground_truth_label[i][j] != ybar.ground_truth_label[i][j])
                ans += 1.0 / clique_size_vector[y.clique_indexes[i][j] - 1];
        }
    }

    return ans;
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
*/
    FILE *modelfl;
    long i;

    modelfl = fopen(file, "w");
    if (modelfl == NULL) {
        printf("Cannot open model file %s for output!", file);
        exit(1);
    }

    fprintf(modelfl, "# sizePsi:%ld\n", sm->sizePsi);
    for (i = 1; i < sm->sizePsi + 1; i++) {
        fprintf(modelfl, "%ld:%.16g\n", i, sm->w[i]);
    }
    fclose(modelfl);

}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/
    STRUCTMODEL sm;

    /* your code here */
    FILE *modelfl;
    long sizePsi, i, fnum;
    double fweight;

    modelfl = fopen(file, "r");
    if (modelfl == NULL) {
        printf("Cannot open model file %s for input!", file);
        exit(1);
    }

    if (fscanf(modelfl, "# sizePsi:%ld", &sizePsi) != 1) {
        printf("Incorrect model file format for %s!\n", file);
        fflush(stdout);
    }

    sm.sizePsi = sizePsi;
    sm.w = (double *) malloc(sizeof(double) * (sizePsi + 1));
    for (i = 0; i < sizePsi + 1; i++) {
        sm.w[i] = 0.0;
    }

    while (!feof(modelfl)) {
        fscanf(modelfl, "%ld:%lf", &fnum, &fweight);
        sm.w[fnum] = fweight;
    }
    fclose(modelfl);

    return (sm);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/

    /* your code here */

    free(sm.w);
}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/

    /* your code here */
    for (int i = 0; i < x.n_rows; ++i) {
        for (int j = 0; j < x.n_cols; ++j) {
            free(x.observed_unary[i][j]);
        }
        free(x.observed_unary[i]);
    }

}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

    /* your code here */
    for (int i = 0; i < y.n_rows; ++i) {
        free(y.ground_truth_label[i]);
        free(y.clique_indexes[i]);
    }

}

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

    /* your code here */
    for (int i = 0; i < h.n_rows; ++i) {
        free(h.auxiliary_z[i]);
    }

}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
    int i;
    for (i = 0; i < s.n; i++) {
        free_pattern(s.examples[i].x);
        free_label(s.examples[i].y);
        free_latent_var(s.examples[i].h);
    }
    free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
    int i;

    /* set default */

    for (i = 0; (i < sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-'); i++) {
        switch ((sparm->custom_argv[i])[2]) {
            /* your code here */
            default:
                printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]);
                exit(0);
        }
    }
}

