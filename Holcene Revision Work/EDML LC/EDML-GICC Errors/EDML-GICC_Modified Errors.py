import pandas as pd
import numpy as np
import os
import re

#set data paths
os.chdir("/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision")
links_path = "EDML-GICC Errors/NGRIP_EDML_match.xlsx"
GICC_GRIP_age = "GICC05-GICC21 Conversion.xlsx"
EDML_age = "EDML-GICC Errors/EDML_counted.xlsx"

#import data
links = pd.read_excel(links_path, skiprows=11, names=["GICC05 b2k", "EDML (m)", "NGRIP (m)"], usecols=[0,1,2])
links['GICC05 b1950'] = links['GICC05 b2k'] - 50
GICC_GRIP_age = pd.read_excel(GICC_GRIP_age, sheet_name=4, skiprows=7, names=["GICC21 Age b2k", "GICC21 dt", "GICC05 years b2k", "GICC05 offset"])
EDML = pd.read_excel(EDML_age, sheet_name=0, skiprows=1, usecols=[0,1])
EDML["age (b1950)"] = EDML["Year b2k"] - 50

EDML_GICC_compare = links[["EDML (m)"]].copy(deep=True) #create a new dataframe for WDC and GRIP depths using stratigraphic links

#get interp age for EDML depths from layer count
interp_edml_age = np.interp(EDML_GICC_compare["EDML (m)"], EDML["Depth (m)"], EDML["age (b1950)"]) #interpolate WDC depths to get ages

#filter out not possible ages (below layer count - DYE03 used for initial counting, but don't have rn & above GICC limits)
EDML_GICC_compare["EDML age (yr BP1950)"] = interp_edml_age #add interpolated ages to dataframe
EDML_GICC_compare = EDML_GICC_compare[EDML_GICC_compare["EDML age (yr BP1950)"] > 1250]
EDML_GICC_compare = EDML_GICC_compare[EDML_GICC_compare["EDML age (yr BP1950)"] < 3835]
EDML_GICC_compare = EDML_GICC_compare.reset_index(drop=True) # reset the index

#create arrays for interpolation
links_gicc = np.array(links["GICC05 b2k"])
gicc05_t = np.array(GICC_GRIP_age["GICC05 years b2k"])
gicc21_t = np.array(GICC_GRIP_age["GICC21 Age b2k"])

#interpolate to switch GICC05 to GICC21 and filter out bad years
gicc21_links = np.interp(links_gicc, gicc05_t, gicc21_t)
gicc21_links = gicc21_links[gicc21_links < 3835]
gicc21_links = gicc21_links[gicc21_links > 1250]

#switch to 1950
EDML_GICC_compare["GICC age (yr BP1950)"] = gicc21_links - 50

#calc diff
EDML_GICC_compare["difference (yr)"] = EDML_GICC_compare["EDML age (yr BP1950)"] - EDML_GICC_compare["GICC age (yr BP1950)"] #calculate difference between ages


# this computes the error from that given section, which is the difference between layer counts in GICC and WDC for that section
section_error = np.zeros(len(EDML_GICC_compare)) #create empty list for section error
for i in range(0, len(EDML_GICC_compare)):
    if i == 0:
        EDML_error = EDML_GICC_compare["EDML age (yr BP1950)"][i] - 0 #calculate section error for each row
        GICC_error = EDML_GICC_compare["GICC age (yr BP1950)"][i] - 0 #calculate section error for each row
    else:
        EDML_error = EDML_GICC_compare["EDML age (yr BP1950)"][i] - EDML_GICC_compare["EDML age (yr BP1950)"][i-1] #calculate section error for each row
        GICC_error = EDML_GICC_compare["GICC age (yr BP1950)"][i] - EDML_GICC_compare["GICC age (yr BP1950)"][i-1] #calculate section error for each row
    section_error[i] = EDML_error - GICC_error #calculate section error for each row

EDML_GICC_compare["section error (yr)"] = section_error #calculate difference between ages

output_path = "EDML-GICC Errors/Modified_WDC_GICC_Compare.xlsx"
EDML_GICC_compare.to_excel(output_path, index=False)

print(f"File saved to: {output_path}")