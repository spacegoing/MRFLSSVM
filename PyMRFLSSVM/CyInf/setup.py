from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("WllepInf", sources = ["WllepGraphCut.pyx","./maxflow-v3.03.src/maxflow.cpp",
                                      "./maxflow-v3.03.src/graph.cpp", "./GraphCut.cpp"], language = "c++")

setup(name = "WllepInf", ext_modules = cythonize(ext))
