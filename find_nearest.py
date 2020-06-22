#!/usr/bin/env python3
import numpy as np

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx
