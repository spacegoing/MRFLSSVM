//
// Created by spacegoing on 6/25/16.
//
#include "ttt.h"
int* crazytest(void);
int main(int argc, char **argv) {
    Py_SetPythonHome(L"/Users/spacegoing/anaconda");
    Py_Initialize();
    PyInit_CCyHelper(); // Python 3.x

    int* arr = crazytest();
    for(int j =0; j<20; j++){
        printf("int: %d", arr[j]);
    }

    Py_Finalize();
    return 0;
}

int* read_struct_examples_helper(void){
    int* a = read_struct_examples_py();
    return a;
}

int* crazytest(void){
    int* a = read_struct_examples();
    return a;
}