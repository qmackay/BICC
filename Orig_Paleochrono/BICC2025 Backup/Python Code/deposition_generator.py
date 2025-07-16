import pandas as pd
import numpy as np
import os

# Load the file
if os.name == 'nt':
    read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD2014_Accumulation.txt"
elif os.name == 'posix':
    read_file = '/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD2014_Accumulation.txt'

df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["depth", "ice_age", "Accumulation", "2sigma_uncertainty"])

print(df)

interval_df = pd.DataFrame({
    "#depth": df["depth"],
    "accumulation (m ice equiv)": df["Accumulation"],
    "error (%)": (df["2sigma_uncertainty"]/df["Accumulation"])
})

#remove rows with missing data
interval_df.dropna(inplace=True)

if os.name == 'nt':
    output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/deposition.txt"
elif os.name == 'posix':
    output_file = "/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/deposition.txt"

interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")