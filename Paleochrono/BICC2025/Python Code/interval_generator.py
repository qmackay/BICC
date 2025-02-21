import pandas as pd
import numpy as np
import os

# Load the WD_AgeDepth file

if os.name == 'nt':
    agedepth_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_AgeDepth.txt"
elif os.name == 'posix':
    agedepth_file = '/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_AgeDepth.txt'
df = pd.read_csv(agedepth_file, delimiter="\t", comment="#", names=["depth", "age", "sigma"])

# Define year intervals
min_age = df["age"].min()
max_age = df["age"].max()
intervals = np.arange(min_age, max_age, 100) #set year intervals, ignoring the end if less than 100 years

depths = np.interp(intervals, df["age"], df["depth"])

sigma_start = np.interp(intervals, df["age"], df["sigma"])
sigma_end = np.interp(intervals+99, df["age"], df["sigma"])

sigma_diff = sigma_end - sigma_start

#this is here to test for an error
for i in range(len(sigma_diff)):
    if sigma_diff[i] == 0:
        sigma_diff[i] += 0.01

interval_df = pd.DataFrame({
    "#top depth (m)": depths[:-1],
    "bottom depth (m)": depths[1:],
    "duration (yr)": 100,
    "sigma duration (yr)": sigma_diff[:-1]
})

interval_df.dropna(inplace=True)

if os.name == 'nt':
    output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/ice_age_intervals.txt"
elif os.name == 'posix':
    output_file = "/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/ice_age_intervals.txt"


interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")