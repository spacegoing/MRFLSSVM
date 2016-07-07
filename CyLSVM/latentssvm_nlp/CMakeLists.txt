cmake_minimum_required(VERSION 3.3)
project(latentnpcoref_v0_12)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fomit-frame-pointer -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fomit-frame-pointer -ffast-math")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -lm")

add_library(SSVM STATIC svm_light/svm_common.o svm_light/svm_hideo.o
        svm_light/svm_learn.o)
SET_SOURCE_FILES_PROPERTIES(
        svm_light/svm_common.o
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
        svm_light/svm_hideo.o
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
        svm_light/svm_learn.o
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
)
SET_TARGET_PROPERTIES(
        SSVM
        PROPERTIES
        LINKER_LANGUAGE CXX
)

add_library(Disjoint STATIC DisjointSets.cpp)
add_library(Nphelper STATIC np_helper.cpp)

add_library(lssvm STATIC svm_struct_latent_api.c)

add_executable(lssvm_learn svm_struct_latent_cccp.c)
SET_TARGET_PROPERTIES(
        lssvm_learn
        PROPERTIES
        LINKER_LANGUAGE CXX
)
target_link_libraries(lssvm_learn lssvm Disjoint Nphelper SSVM)

#add_executable(lssvm_pred svm_struct_latent_classify.c)
#target_link_libraries(lssvm_pred lssvm Disjoint Nphelper)

