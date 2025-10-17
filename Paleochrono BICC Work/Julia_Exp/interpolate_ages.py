#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:51:54 2025

@author: parrenif
"""

import sys
import numpy as np

dir = sys.argv[1]

readarray = np.loadtxt(dir+'output.txt')
depth = readarray[:, 0]
age = readarray[:, 1]

new_depth = np.loadtxt(dir+'depths_outputs.txt')

new_age = np.interp(new_depth, depth, age)

output = np.vstack((new_depth, new_age))
np.savetxt(dir+'output_interpolated.txt', np.transpose(output), delimiter='\t')
