import pandas as pd
import numpy as np
import os

# Load the file
if os.name == 'nt':
    read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/Tie Points/Methane Many-core tie-points.txt"
elif os.name == 'posix':
    read_file = '/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/Tie Points/Methane Many-core tie-points.txt'

df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["WD_depth", "WD_unc", "EDML_depth", "EDML_unc", "EDC_depth", "EDC_unc", "DF_depth", "DF_unc", "TAL_depth", "TAL_unc", "SP_depth", "SP_unc"])

core1 = "EDC"
core2 = "WD"

interval_df = pd.DataFrame({
    f"#{core1} Depth": df[f"{core1}_depth"],
    f"{core2} Depth": df[f"{core2}_depth"],
    'uncertainty': ((df[f"{core1}_unc"]+df[f"{core2}_unc"]))
})

interval_df.dropna(inplace=True)

if os.name == 'nt':
    output_file = f"E:/GitHub/BICC/Paleochrono/EDML-WDC_test/{core1}-WDC/airair_synchro_horizons.txt"
elif os.name == 'posix':
    output_file = f"/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/EDML-WDC_test/{core1}-WDC/airair_synchro_horizons.txt"

print(interval_df)

interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")