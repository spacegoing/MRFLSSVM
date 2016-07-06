/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api_types.h                                      */
/*                                                                      */
/*   API type definitions for Latent SVM^struct                         */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 30.Sep.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "svm_light/svm_common.h"

typedef struct pattern {
    /*
      Type definition for input pattern x
    */
    int n_rows;
    int n_cols;
    double **observed_label;
} PATTERN;

typedef struct label {
    /*
      Type definition for output label y
    */
    int n_rows;
    int n_cols;
    int **clique_indexes;
    int **ground_truth_label;
} LABEL;

typedef struct latent_var {
    /*
      Type definition for latent variable h
    */
    int n_rows;
    int *auxiliary_z;
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

struct check_options {
    // Image Configs
    int gridStep;  // grid size for defining cliques
    int W; // image width
    int H; // image height
    int numCliques; // number of cliques
    int N; // number of variables

    int dimUnary;
    int dimPairwise;

    // Learning Configs
    int K;  // number of lower linear functions
    int maxIters;  // maximum learning iterations
    double eps;  // constraint violation threshold

    // Other Configs
    int learningQP;  // encoding for learning QP (1, 2, or 3)
    int figWnd;  // figure for showing results
};

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
    check_options options;
    int length_feature_vector;
    double cost_factor;
} STRUCT_LEARN_PARM;



