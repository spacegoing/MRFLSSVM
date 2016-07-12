cmake_minimum_required(VERSION 3.5)
project(latentssvm_v0_12)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fomit-frame-pointer -ffast-math")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fomit-frame-pointer -ffast-math")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -lm")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

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

add_library(GraphCut STATIC maxflow-v3.03.src/maxflow.cpp maxflow-v3.03.src/graph.cpp)

add_library(CHBD STATIC Checkboard/Checkboard.cpp)

add_library(LSSVMY STATIC svm_struct_latent_api.c)

set(SOURCE_FILES mrf_helper.cpp svm_struct_latent_api_types.h svm_struct_latent_cccp.c)

add_executable(mrflssvm_learn ${SOURCE_FILES})
target_link_libraries(mrflssvm_learn CHBD GraphCut SSVM LSSVMY ${ARMADILLO_LIBRARIES})
