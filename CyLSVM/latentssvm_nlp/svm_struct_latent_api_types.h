/* svm_struct_latent_api_types.h */
/* 30 September 2008 */


# include "svm_light/svm_common.h"

typedef struct pattern {
    /* noun phrases are numbered from 1 to num_nps.
       pair_features are the features describing the correlation between 2 noun phrases
     */
    int num_nps;
    SVECTOR ***pair_features;

} PATTERN;

typedef struct label {
    /* cluster_id is an array of dimension num_nps;
       if cluster_id[i]==cluster_id[j], then noun phrase i and noun phrase j
       are coreferent. There is no requirement that the cluster id has to be
       contiguous integers.
     */
    int num_nps;
    int num_clusters;
    int *cluster_id;
} LABEL;

typedef struct ll_node {
    /* linked list node in adjacency list representation.
       (u,v) is the edge in the graph, with u, v the identifiers for
       noun phrases
     */
    int u;
    int v;
    struct ll_node *next;
} LL_NODE;

typedef struct latent_var {
    LL_NODE *head;
} LATENT_VAR;

typedef struct example {
    PATTERN x;
    LABEL y;
    LATENT_VAR h;
} EXAMPLE;

typedef struct sample {
    int n;
    EXAMPLE *examples;
} SAMPLE;


typedef struct structmodel {
    double *w;
    /* pointer to the learned weights */
    MODEL *svm_model;
    /* the learned SVM model */
    long sizePsi;     /* maximum number of weights in w */
    /* other information that is needed for the stuctural model can be
       added here, e.g. the grammar rules for NLP parsing */
} STRUCTMODEL;


typedef struct struct_learn_parm {
    double epsilon;
    /* precision for which to solve
        quadratic program */
    long newconstretrain;
    /* number of new constraints to
              accumulate before recomputing the QP
              solution */
    double C;
    /* trade-off between margin and loss */
    char custom_argv[20][1000];
    /* string set with the -u command line option */
    int custom_argc;
    /* number of -u command line options */
    int slack_norm;
    /* norm to use in objective function
                           for slack variables; 1 -> L1-norm,
           2 -> L2-norm */
    int loss_type;
    /* selected loss function from -r
          command line option. Select between
          slack rescaling (1) and margin
          rescaling (2) */
    int loss_function;        /* select between different loss
				  functions via -l command line
				  option */
    /* add your own variables */
    long max_feature_key;
    double cost_factor;          // cost factor for penalizing false positive edges
} STRUCT_LEARN_PARM;


/* label definition from Tom for evaluation purpose;
   renamed to TOM_LABEL
 */
typedef struct tom_label {
    /* this defines the y-part (the label) of a training example,
       e.g. the parse tree of the corresponding sentence. */
    /* How man noun-phrases, and how many clusters do we have? */
    int number_of_nps, number_of_clusters;
    /* Arrays of length number_of_nps.  Each item in an array
       corresponds to a noun-phrase.

       Suppose we're looking at index i.  first_member[i] holds the
       index of the "first" member of the cluster that NP i is in, a
       sort of cluster ID.  Notice, the first member is NOT guaranteed
       to have the numerically lowest ID.

       next_member[i] holds the index of the next member in the cluster
       that i belongs to -- also, the last item of a cluster points to
       the first item, so this forms a cycle.  For an example of what I
       mean, if there are "n" items in the cluster that "i" belongs to,
       then "next_member[next_member[...[i]...]]" (with "n" references)
       should evaluate to "i".

       Finally, number[i] points to the number of items in this cluster,
       but only if member i is the first member for the cluster.
       Therefore, to get the number of members in the cluster of
       noun-phrase i, one could always get number[first_member[i]]. */
    unsigned short *next_member, *first_member, *number;
    /* Alternatively, one could set the above three fields as NULL and
       have the relaxation variables "relax".  If "relax" is not null,
       there are "number_of_nps" elements in "relax", and "relax[i]"
       points to an array of length i. */
    double **relax;
} TOM_LABEL;
