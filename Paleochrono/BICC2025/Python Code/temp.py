import os
import pandas as pd

if os.name == 'nt':
    edc_file = "E:/GitHub/BICC/Paleochrono/BICC2025/raw data/Methane Data/EDC CH4.xlsx"
elif os.name == 'posix':
    edc_file = "/Users/quinnmackay/Documents/GitHub/BICC/Paleochrono/BICC2025/raw data/Methane Data/EDC CH4.xlsx"

edc_ch4 = pd.read_excel(edc_file, comment="#", skiprows=1, sheet_name=0, names=["Depth (m)", "Age (yr)", "CH4 (ppb)", "uncertainty"])

edc_ch4.dropna(inplace=True)

print(edc_ch4)