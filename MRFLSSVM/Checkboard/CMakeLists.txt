cmake_minimum_required(VERSION 3.5)
project(Checkboard)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(SOURCE_FILES Checkboard.cpp)
add_executable(Checkboard ${SOURCE_FILES})

target_link_libraries(Checkboard ${ARMADILLO_LIBRARIES})
