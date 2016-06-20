#include "Python.h"
#include "test.h"
#include <math.h>
#include <stdio.h>

void cAdd(int *);

int main(int argc, char **argv) {
    Py_SetPythonHome(L"/Users/spacegoing/anaconda");
    Py_Initialize();
//    inittest(); // Python 2.x
    PyInit_test(); // Python 3.x
    int a [2] = {0 , 0};
    cAdd(a);
    printf("fist: %d, second: %d", a[0], a[1]);

    int row = 3;
    int col = 4;
    int * arr = numpyArray(3,4);
    for (int i=0; i<row; i++){
        for(int j =0; j<col; j++){
            printf("int: %d", arr[i*col + j]);
        }
    }

    Py_Finalize();
    return 0;
}

void cAdd(int* a){
    pythonAdd(a);
}