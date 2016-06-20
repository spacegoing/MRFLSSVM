# -*- coding: utf-8 -*-
__author__ = 'spacegoing'

from scipy.io import loadmat

def loadCheckboard():
    mat = loadmat('tmpData/linEnvLearn.mat')

    return mat['options'], mat['instance']

def loadCheckboard():
    mat = loadmat('tmpData/linEnvLearn.mat')

    return mat['instance']['unary'][0][0]
