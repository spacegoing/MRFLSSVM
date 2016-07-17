# MRF-LSSVM Package #
This package is based on

1. Chun-Nam Yu LSSVM Package
2. Thorsten Joachims SVM^light Package
3. Yuri Boykov MAXFLOW Package

## Package Construction ##

The package is constructed as follows:

1. main file: svm_struct_latent_cccp.c
Main file from Yu's LSSVM package. `cutting_plane_algorithm` is modified for adding QP parameters' constraints.

2. APIs: svm_struct_latent_api.c and svm_struct_latent_api_types.h
These are APIs and Types called by main file.

3. MRF related functions: mrf_helper.cpp
This is the implementation file of those APIs.

4. Inputs and Configurations: Checkboard Directory
These are the equivalance of params and instance in Matlab code.
