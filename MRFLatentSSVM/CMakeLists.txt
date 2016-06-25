cmake_minimum_required(VERSION 3.5)
project(MRFLatentSSVM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    maxflow-v3.03.src/block.h
    maxflow-v3.03.src/graph.cpp
    maxflow-v3.03.src/graph.h
    maxflow-v3.03.src/maxflow.cpp
    svm_light/kernel.h
    svm_light/svm_classify.c
    svm_light/svm_common.c
    svm_light/svm_common.h
    svm_light/svm_hideo.c
    svm_light/svm_learn.c
    svm_light/svm_learn.h
    svm_light/svm_learn_main.c
    mrf_helper.cpp
    mrf_helper.h
    svm_struct_latent_api.c
    svm_struct_latent_api.h
    svm_struct_latent_api_types.h
    svm_struct_latent_cccp.c
    svm_struct_latent_classify.c)

add_executable(MRFLatentSSVM ${SOURCE_FILES})