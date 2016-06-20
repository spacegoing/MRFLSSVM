# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("testc", sources=["test.pyx"])

setup(name="testc", ext_modules=cythonize(ext))
