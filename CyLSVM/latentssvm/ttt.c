//
// Created by spacegoing on 6/25/16.
//
#include "Python.h"
#include "CythonWrapper.h"
#include <stdio.h>

int main(int argc, char **argv) {
    Py_SetPythonHome(L"/Users/spacegoing/anaconda");
    Py_Initialize();
//    inittest(); // Python 2.x
    PyInit_CCyHelper(); // Python 3.x

    int *a;
    read_struct_examples_helper(a);

    int col = sizeof a / sizeof *a;
    for (int j = 0; j < col; j++) {
        printf("id: %d, int: %d", j, a[j]);
    }

    Py_Finalize();
    return 0;
}