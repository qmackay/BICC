import pandas as pd
import numpy as np

# Load the file
read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_Thinning.tab"
df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["depth", "thinning", "uncertainty"])

print(df)

interval_df = pd.DataFrame({
    "#depth": df["depth"],
    "thinning": df["thinning"],
    'uncertainty (%)': df["uncertainty"]
})

output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/thinning.txt"
interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")