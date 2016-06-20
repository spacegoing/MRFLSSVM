# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("testc", sources=["test.pyx"],
                include_dirs=['.', get_include()])

setup(name="testc", ext_modules=cythonize(ext))
