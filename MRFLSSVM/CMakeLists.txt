cmake_minimum_required(VERSION 3.5)
project(latentssvm_v0_12)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(SOURCE_FILES
        mrf_helper.cpp
        mrf_helper.h
        Checkboard/Checkboard.cpp
        Checkboard/Checkboard.h)

add_executable(mrflssvm ${SOURCE_FILES})
target_link_libraries(mrflssvm ${ARMADILLO_LIBRARIES})
