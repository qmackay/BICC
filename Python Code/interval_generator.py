import pandas as pd
import numpy as np

# Load the WD_AgeDepth file
agedepth_file = "E:\GitHub\BICC\Paleochrono\EDML-WDC_test\WDC\WD_AgeDepth.txt"
df = pd.read_csv(agedepth_file, delimiter="\t", comment="#", names=["depth", "age", "sigma"])

# Define 1000-year intervals
min_age = df["age"].min()
max_age = df["age"].max()
intervals = np.arange(min_age, max_age, 1000)

# Interpolate depths for the defined age intervals. (this works because the original file doesn't skip depths, so doesn't need to interpolate)
depths = np.interp(intervals, df["age"], df["depth"])

interval_df = pd.DataFrame({
    "top depth (m)": depths[:-1],
    "bottom depth (m)": depths[1:],
    "duration (yr)": 1000,
    "sigma duration (yr)": 50 #how should I calculate this?
})

output_file = "E:\GitHub\BICC\Paleochrono\EDML-WDC_test\WDC\WD_AgeDepth_intervals.txt"
interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")