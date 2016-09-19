# -*- coding: utf-8 -*-
import pickle
from Checkboard import Instance, Options

__author__ = 'spacegoing'

def dump_pickle(prefix_str, outer_history, instance, options):
    with open(prefix_str + 'active.pickle', 'wb') as f:
        pickle.dump({"outer_history": outer_history,
                     "instance": instance,
                     "options": options}, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath):
    with open(filepath + ".pickle", 'rb') as f:
        data = pickle.load(f)
        instance = data['instance']
        options = data['options']
        outer_history = data['outer_history']

    return outer_history, instance, options
