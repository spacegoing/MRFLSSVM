from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("WllepGraphCut", sources = ["WllepGraphCut.pyx","./maxflow-v3.03.src/maxflow.cpp",
                                      "./maxflow-v3.03.src/graph.cpp", "./GraphCut.cpp"], language = "c++")

setup(name = "WllepGraphCut", ext_modules = cythonize(ext))
