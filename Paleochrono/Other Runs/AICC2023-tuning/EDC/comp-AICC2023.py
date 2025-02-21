#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:35:51 2023

@author: parrenif
"""

import numpy as np
import matplotlib.pyplot as plt

readarray = np.loadtxt('output.txt')
depth = readarray[:, 0]
age = readarray[:, 1]
air_age = readarray[:, 3]

readarray = np.loadtxt('output-AICC2023.txt')
age_AICC2023 = readarray[:, 1]
air_age_AICC2023 = readarray[:, 3]

plt.plot(age_AICC2023, age - age_AICC2023)
plt.show()