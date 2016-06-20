/* svm_struct_latent_api.c */
/* 17 December 2008 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "np_helper.h"


SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
    return (read_struct_examples_helper(file, sparm));
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm,
                       KERNEL_PARM *kparm) {

    sm->sizePsi = sparm->max_feature_key;

}


SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
    LL_NODE *edge;
    SVECTOR *oldsum, *sum;
    WORD empty[2];

    empty[0].wnum = 0;
    sum = create_svector(empty, "", 1.0);

    edge = h.head;
    while (edge != NULL) {
        oldsum = sum;
        sum = add_ss(x.pair_features[edge->u][edge->v], oldsum);
        free_svector(oldsum);
        edge = edge->next;
    }

    return (sum);
}

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
    classify_struct_example_helper(x, y, h, sm, sparm);
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm,
                                                   STRUCT_LEARN_PARM *sparm) {
    find_most_violated_constraint_marginrescaling_helper(x, y, ybar, hbar, sm, sparm);
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
    return (infer_latent_variables_helper(x, y, sm, sparm));
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
    double counts;
    LL_NODE *edge;

    counts = y.num_nps - y.num_clusters;
    edge = hbar.head;
    while (edge != NULL) {
        assert(edge->u < y.num_nps);
        assert(edge->v < y.num_nps);
        if (y.cluster_id[edge->u] == y.cluster_id[edge->v]) {
            counts -= 1.0;
        } else {
            counts += sparm->cost_factor;
        }
        edge = edge->next;
    }
    assert(counts >= 0);
    return (counts);

}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
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
    STRUCTMODEL sm;
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
    free(sm.w);
}

void free_pattern(PATTERN x) {
    int i, j;
    for (i = 0; i < x.num_nps; i++) {
        for (j = 0; j < i; j++) {
            free_svector(x.pair_features[i][j]);
        }
        if (i > 0)
            free(x.pair_features[i]);
    }
    free(x.pair_features);
}

void free_label(LABEL y) {
    free(y.cluster_id);
}

void free_latent_var(LATENT_VAR h) {
    LL_NODE *ptr, *temp;
    ptr = h.head;
    while (ptr != NULL) {
        temp = ptr->next;
        free(ptr);
        ptr = temp;
    }
}

void free_struct_sample(SAMPLE s) {
    int i;
    for (i = 0; i < s.n; i++) {
        free_pattern(s.examples[i].x);
        free_label(s.examples[i].y);
        free_latent_var(s.examples[i].h);
    }
    free(s.examples);

}

void free_tomlabel(TOM_LABEL ty) {

    free(ty.first_member);
    free(ty.next_member);
    free(ty.number);

}


void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
    int i;

    /* set default */
    sparm->cost_factor = 1.0;

    for (i = 0; (i < sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-'); i++) {
        switch ((sparm->custom_argv[i])[2]) {
            case 'k':
                i++;
                sparm->cost_factor = atof(sparm->custom_argv[i]);
                break;
            default:
                printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]);
                exit(0);
        }
    }

    assert(sparm->cost_factor > -1E-8);
    assert(sparm->cost_factor < 1.0 + 1E-8);

}


double pairwise_loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm) {
    TOM_LABEL ty, tybar;
    double score;

    ty = label2tomlabel(y);
    tybar = label2tomlabel(ybar);

    score = helper_joachims(ty, tybar, sparm);
    free_tomlabel(ty);
    free_tomlabel(tybar);

    return (1.0 - score);

}

double mitre_loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm) {
    TOM_LABEL ty, tybar;
    double score;

    ty = label2tomlabel(y);
    tybar = label2tomlabel(ybar);

    score = helper_vilain(ty, tybar, sparm);
    free_tomlabel(ty);
    free_tomlabel(tybar);

    return (1.0 - score);

}
