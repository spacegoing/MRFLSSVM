cmake_minimum_required(VERSION 3.5)
project(svm_light)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    kernel.h
    svm_classify.c
    svm_common.c
    svm_common.h
    svm_hideo.c
    svm_learn.c
    svm_learn.h
    svm_learn_main.c)

add_executable(svm_light ${SOURCE_FILES})