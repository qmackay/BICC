import pandas as pd
import numpy as np
import os
import re

#set data paths
os.chdir("/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision")
links_path = "EDML-WDC Errors/EDML_WDC_horizons.txt"
WD2014_path = "Updated_WD2014 Layer Count.tab"
EDML_age = "EDML-WDC Errors/EDML_counted.xlsx"

#import data
links = pd.read_csv(links_path, skiprows=1, names=["EDML", "WDC", "UNCERTAINTY"], delimiter='\t')
WD2014 = pd.read_csv(WD2014_path, comment="#", delimiter="\t", names=["Depth ice/snow [m]", "Cal age [ka BP] (ice age)"])
EDML = pd.read_excel(EDML_age, sheet_name=0, skiprows=1, usecols=[0,1])
EDML["age (b1950)"] = EDML["Year b2k"] - 50

WDC_EDML_compare = links[["WDC", "EDML"]].copy(deep=True) #create a new dataframe for WDC and GRIP depths using stratigraphic links

#get interp age for WDC depths from layer count
interp_wdcage = np.interp(WDC_EDML_compare["WDC"], WD2014["Depth ice/snow [m]"], WD2014["Cal age [ka BP] (ice age)"]) #interpolate WDC depths to get ages
interp_wdcage = interp_wdcage * 1000 #convert ka BP to yr BP

WDC_EDML_compare["WD2014 age (yr BP1950)"] = interp_wdcage #add interpolated ages to dataframe

#get interp age for GRIP depths from GICC layer count
interp_edml_age = np.interp(WDC_EDML_compare["EDML"], EDML["Depth (m)"], EDML["age (b1950)"]) #interpolate GRIP depths to get GICC ages
WDC_EDML_compare["EDML age (yr BP1950)"] = interp_edml_age

# Filter out rows where GICC age > 3800
WDC_EDML_compare = WDC_EDML_compare[~np.logical_or(WDC_EDML_compare['WD2014 age (yr BP1950)'] > 3800, WDC_EDML_compare['EDML age (yr BP1950)'] > 3800)]
WDC_EDML_compare = WDC_EDML_compare.reset_index(drop=True) #reset index

WDC_EDML_compare["difference (yr)"] = WDC_EDML_compare["WD2014 age (yr BP1950)"] - WDC_EDML_compare["EDML age (yr BP1950)"] #calculate difference between ages


# this computes the error from that given section, which is the difference between layer counts in GICC and WDC for that section

section_error = np.zeros(len(WDC_EDML_compare)) #create empty list for section error
for i in range(0, len(WDC_EDML_compare)):
    if i == 0:
        WDC_error = WDC_EDML_compare["WD2014 age (yr BP1950)"][i] - 0 #calculate section error for each row
        EDML_error = WDC_EDML_compare["EDML age (yr BP1950)"][i] - 0 #calculate section error for each row
    else:
        WDC_error = WDC_EDML_compare["WD2014 age (yr BP1950)"][i] - WDC_EDML_compare["WD2014 age (yr BP1950)"][i-1] #calculate section error for each row
        EDML_error = WDC_EDML_compare["EDML age (yr BP1950)"][i] - WDC_EDML_compare["EDML age (yr BP1950)"][i-1] #calculate section error for each row
    section_error[i] = WDC_error - EDML_error #calculate section error for each row

WDC_EDML_compare["section error (yr)"] = section_error #calculate difference between ages

output_path = "EDML-WDC Errors/Modified_WDC_EDML_Compare.xlsx"
WDC_EDML_compare.to_excel(output_path, index=False)

print(f"File saved to: {output_path}")