import pandas as pd
import numpy as np

# Load the WD_AgeDepth file
agedepth_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_AgeDepth.txt"
df = pd.read_csv(agedepth_file, delimiter="\t", comment="#", names=["depth", "age", "sigma"])

# Define year intervals
min_age = df["age"].min()
max_age = df["age"].max()
intervals = np.arange(min_age, max_age, 100) #set year intervals, ignoring the end if less than 100 years

# Interpolate depths for the defined age intervals. (this works because the original file doesn't skip depths, so doesn't need to interpolate)
depths = np.interp(intervals, df["age"], df["depth"])

sigma_start = np.interp(intervals, df["age"], df["sigma"])
sigma_end = np.interp(intervals+99, df["age"], df["sigma"])

sigma_diff = sigma_end - sigma_start

interval_df = pd.DataFrame({
    "#top depth (m)": depths[:-1],
    "bottom depth (m)": depths[1:],
    "duration (yr)": 100,
    "sigma duration (yr)": sigma_diff[:-1] #how should I calculate this?
})

output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/ice_age_intervals.txt"
interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")