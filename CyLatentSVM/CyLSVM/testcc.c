#include "Python.h"
#include "test.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    Py_Initialize();
    inittest();
    int a [2] = {0 , 0};
    pythonAdd(a);
    printf("fist: %d, second: %d", a[0], a[1]);
    Py_Finalize();
    return 0;
}