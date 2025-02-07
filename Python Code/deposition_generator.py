import pandas as pd
import numpy as np

# Load the file
read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD2014_Accumulation.txt"
df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["depth", "ice_age", "Accumulation", "2sigma_uncertainty"])

print(df)

interval_df = pd.DataFrame({
    "#depth": df["depth"],
    "accumulation (m ice equiv)": df["Accumulation"],
})

output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/deposition.txt"
interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")