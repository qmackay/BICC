import pandas as pd
import numpy as np

# Load the file
read_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/raw data/WDC/WD_LID.txt"
df = pd.read_csv(read_file, delimiter="\t", comment="#", names=["depth", "LID"])

stdev = 4/df["LID"] # 4% uncertainty

interval_df = pd.DataFrame({
    "#depth": df["depth"],
    "LID": df["LID"],
    'stdev (%)': stdev
})

output_file = "E:/GitHub/BICC/Paleochrono/EDML-WDC_test/WDC/lock_in_depth.txt"
interval_df.to_csv(output_file, sep="\t", index=False)

print(f"Done! Saved to: {output_file}")