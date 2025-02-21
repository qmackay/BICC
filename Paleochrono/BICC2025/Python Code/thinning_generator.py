import pandas as pd
import numpy as np
import os

# Load the file
if os.name == 'nt':
    read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_Thinning.tab"
elif os.name == 'posix':
    read_file = '/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_Thinning.tab'

df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["depth", "thinning", "uncertainty"])

print(df)

interval_df = pd.DataFrame({
    "#depth": df["depth"],
    "thinning": df["thinning"],
    'uncertainty (%)': (df["uncertainty"]/df["thinning"])
})

interval_df.dropna(inplace=True)

if os.name == 'nt':
    output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/thinning.txt"
elif os.name == 'posix':
    output_file = "/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/thinning.txt"

interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")