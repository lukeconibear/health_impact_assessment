#!/usr/bin/env python3
import numpy as np

def import_npz(npz_file, namespace):
    '''load all numpy arrays into global namespace'''
    '''ensure original arrays have the variable name you require'''
    data = np.load(npz_file)
    for var in data:
        if data[var].dtype == np.dtype('float64'):
            namespace[var] = data[var].astype('float32')
        else:
            namespace[var] = data[var]
