import pandas as pd
import numpy as np
import os
import re

links_path = "E:\GitHub\BICC\Holcene Revision\GRIP_WDC_TephraLinks.xlsx"

links = pd.read_excel(links_path, usecols=[0,1,2,3,4,5,6,7,8], skiprows=29, names=["WDC(m)", "GRIP(m)", "WD2014 age (BC/AD iso)", "WD2014 age (yr BP)", "GICC05 age (yr BP)", "age diff", "GRIP dz/dt", "distance (yrs)", "type"])

print(links)