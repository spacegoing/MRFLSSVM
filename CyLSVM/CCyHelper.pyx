
cdef extern from "./latentssvm/svm_struct_latent_api_types.h":
    ctypedef struct PATTERN:
        double *unary

    ctypedef struct LABEL:
        int *y

    ctypedef struct LATENT_VAR:
        int *z

    ctypedef struct EXAMPLE:
        PATTERN x;
        LABEL y;
        LATENT_VAR h;

    ctypedef struct SAMPLE:
        int n
        EXAMPLE *examples

