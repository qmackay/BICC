import pandas as pd
import numpy as np
import os

# Load the file
if os.name == 'nt':
    read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_Density.tab"
elif os.name == 'posix':
    read_file = '/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_Density.tab'
df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["depth", "relative_density", "sigma"])

interval_df = pd.DataFrame({
    "#depth": df["depth"],
    "relative density": df["relative_density"],
})

#remove rows with missing data
interval_df.dropna(inplace=True)

if os.name == 'nt':
    output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/density.txt"
elif os.name == 'posix':
    output_file = "/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/density.txt"

interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")