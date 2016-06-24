# -*- coding: utf-8 -*-
__author__ = 'spacegoing'
from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

extensionName = "testc"
sourcesFiles = ["CCyHelper.pyx"]

ext = Extension(extensionName, sources=sourcesFiles,
                include_dirs=['.', get_include()])

setup(name=extensionName, ext_modules=cythonize(ext))
