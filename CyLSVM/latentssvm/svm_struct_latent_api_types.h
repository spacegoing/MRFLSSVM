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
    double *unary; // Float. observed unary terms (y destroyed by noisy)
} PATTERN;

typedef struct label {
    /*
      Type definition for output label y
    */
    int *y; // Binary{0,1}. ground-truth unary terms
} LABEL;

typedef struct latent_var {
    /*
      Type definition for latent variable h
    */
    int *z; // Binary{0,1}. Auxiliary variables
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
    /* add your own variables */
    long max_feature_key;
    double cost_factor;     // cost factor for penalizing false positive edges
} STRUCT_LEARN_PARM;

