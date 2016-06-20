from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("GraphCy", sources = ["GraphCy.pyx","./maxflow-v3.03.src/maxflow.cpp","./maxflow-v3.03.src/graph.cpp", "./GraphSimp.cpp"], language = "c++")

setup(name = "GraphCy", ext_modules = cythonize(ext))
