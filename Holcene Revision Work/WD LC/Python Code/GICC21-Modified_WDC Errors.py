import pandas as pd
import numpy as np
import os
import re

#set data paths
os.chdir("/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision")
links_path = "GRIP_WDC_TephraLinks.xlsx"
WD2014_path = "Updated_WD2014 Layer Count.tab"
GICC_GRIP_age = "GICC05-GICC21 Conversion.xlsx"

#import data
links = pd.read_excel(links_path, usecols=[0,1,2,3,4,5,6,7,8], skiprows=29, names=["WDC(m)", "GRIP(m)", "WD2014 age (BC/AD iso)", "WD2014 age (yr BP)", "GICC05 age (yr BP)", "age diff", "GRIP dz/dt", "distance (yrs)", "type"])
WD2014 = pd.read_csv(WD2014_path, comment="#", delimiter="\t", names=["Depth ice/snow [m]", "Cal age [ka BP] (ice age)"])
GICC_GRIP_age = pd.read_excel(GICC_GRIP_age, sheet_name=1, skiprows=1, names=["Age (b2k)", "dt years", "age (CE/BCE)", "EastGRIP (m)",	"NEEM (m)",	"NorthGRIP1 (m)",	"NorthGRIP2 (m)",	"NEEM-2011-S1 (m)",	"GRIP (m)",	"DYE-3 79 (m)",	"DYE-3 4B (m)",	"DYE-3 18C (m)"])

GICC_GRIP_age["Age (b1950)"] = GICC_GRIP_age["Age (b2k)"] - 50 #make b1950 column

WDC_GICC_compare = links[["WDC(m)", "GRIP(m)"]].copy(deep=True) #create a new dataframe for WDC and GRIP depths using stratigraphic links

#get interp age for WDC depths from layer count
interp_wdcage = np.interp(WDC_GICC_compare["WDC(m)"], WD2014["Depth ice/snow [m]"], WD2014["Cal age [ka BP] (ice age)"]) #interpolate WDC depths to get ages
interp_wdcage = interp_wdcage * 1000 #convert ka BP to yr BP

WDC_GICC_compare["WD2014 age (yr BP1950)"] = interp_wdcage #add interpolated ages to dataframe

#get interp age for GRIP depths from GICC layer count
interp_giccage = np.interp(WDC_GICC_compare["GRIP(m)"], GICC_GRIP_age["GRIP (m)"], GICC_GRIP_age["Age (b1950)"]) #interpolate GRIP depths to get GICC ages
WDC_GICC_compare["GICC age (yr BP1950)"] = interp_giccage

# Filter out rows where GICC age > 3800
WDC_GICC_compare = WDC_GICC_compare[0:31]

WDC_GICC_compare["difference (yr)"] = WDC_GICC_compare["WD2014 age (yr BP1950)"] - WDC_GICC_compare["GICC age (yr BP1950)"] #calculate difference between ages


# this computes the error from that given section, which is the difference between layer counts in GICC and WDC for that section

section_error = np.zeros(len(WDC_GICC_compare)) #create empty list for section error
for i in range(0, len(WDC_GICC_compare)):
    if i == 0:
        WDC_error = WDC_GICC_compare["WD2014 age (yr BP1950)"][i] - 0 #calculate section error for each row
        GICC_error = WDC_GICC_compare["GICC age (yr BP1950)"][i] - 0 #calculate section error for each row
    else:
        WDC_error = WDC_GICC_compare["WD2014 age (yr BP1950)"][i] - WDC_GICC_compare["WD2014 age (yr BP1950)"][i-1] #calculate section error for each row
        GICC_error = WDC_GICC_compare["GICC age (yr BP1950)"][i] - WDC_GICC_compare["GICC age (yr BP1950)"][i-1] #calculate section error for each row
    section_error[i] = WDC_error - GICC_error #calculate section error for each row

WDC_GICC_compare["section error (yr)"] = section_error #calculate difference between ages

output_path = "Modified_WDC_GICC_Compare.txt"
WDC_GICC_compare.to_csv(output_path, sep="\t", index=False)

print(f"File saved to: {output_path}")