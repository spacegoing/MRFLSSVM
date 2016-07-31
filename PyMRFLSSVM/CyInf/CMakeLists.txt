cmake_minimum_required(VERSION 3.5)
project(CyInf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(GraphCut STATIC maxflow-v3.03.src/maxflow.cpp maxflow-v3.03.src/graph.cpp)

set(SOURCE_FILES GraphSimp.cpp GraphSimp.h GraphCut.cpp GraphCut.h)

add_executable(CyInf ${SOURCE_FILES})
target_link_libraries(CyInf GraphCut)
