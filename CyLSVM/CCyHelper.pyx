from PyLSVM import instance, Options
import numpy as np

cdef api int* read_struct_examples_py():
    cdef int[:] y = instance.y

    return &y[0]