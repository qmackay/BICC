import pandas as pd
import numpy as np
import os

# Load the file

os_system = os.name
if os.name == 'nt':
    read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_LID.txt"
elif os.name == 'posix':
    read_file = '/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_LID.txt'

df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["depth", "LID"])

stdev = 0.10 # 10% uncertainty

interval_df = pd.DataFrame({
    "#depth": df["depth"],
    "LID": df["LID"],
    'stdev (%)': stdev
})

#remove rows with missing data
interval_df.dropna(inplace=True)

if os.name == 'nt':
    output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/lock_in_depth.txt"
elif os.name == 'posix':
    output_file = '/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/lock_in_depth.txt'

interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")