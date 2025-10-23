import pandas as pd
import numpy as np
import os
import openpyxl


#DF CH4 Import

df_file = "E:/GitHub/BICC/Paleochrono/BICC2025/raw data/Methane Data/DF CH4.xlsx"

df_ch4 = pd.read_excel(df_file, comment="#", skiprows=19, sheet_name=5, usecols=[0,1], names=["depth (m)", "CH4 (ppb)"])

df_ch4.dropna(inplace=True)

#DF CH4 Import

edc_file = "E:/GitHub/BICC/Paleochrono/BICC2025/raw data/Methane Data/EDC CH4.xlsx"

edc_ch4 = pd.read_excel(edc_file, comment="#", skiprows=9, sheet_name=0, usecols=[0,1,2], names=["depth (m)", "CH4 (ppb)", "uncertainty"])

edc_ch4.dropna(inplace=True)

#EDML CH4 Import

edml_file = "E:/GitHub/BICC/Paleochrono/BICC2025/raw data/Methane Data/EDML CH4.tab"

edml_ch4 = pd.read_csv(edml_file, delimiter="\t", comment="#", names=["depth (m)", "Gas age (ka bp)", "CH4 (ppb)"])

edml_ch4.dropna(inplace=True)

#WDC CH4 Import

wdc_file = "E:/GitHub/BICC/Paleochrono/BICC2025/raw data/Methane Data/WDC CH4.txt"

wdc_ch4 = pd.read_csv(wdc_file, delimiter="\t", comment="#", names=["depth (m)", "age (ka bp)", "CH4 (ppb)", "lab"])

wdc_ch4.dropna(inplace=True)

wdc_ch4 = pd.DataFrame({
    "depth": wdc_ch4["depth (m)"],
    "age (ka bp)": wdc_ch4["age (ka bp)"],
    "CH4 (ppb)": wdc_ch4["CH4 (ppb)"]

})

def paleochrono_import():
    cores = ['DF', 'EDC', 'EDML', 'WDC']
    paleo_output = {}

    for core in cores:
        core_df_path = f"E:/GitHub/BICC/Paleochrono/BICC2025/{core}/output.txt"
        core_df = pd.read_csv(core_df_path, delimiter="\t", comment="#", names=["depth", "ice_age", "sigma_ice_age", "air_age", "sigma_air_age", "sigma_delta_age", "deporate", "sigma_deporate", "thinning", "sigma_thinning", "LID", "sigma_LID", "delta_depth", "sigma_delta_depth", "deporate_model", "sigma_deporate_model", "thinning_model", "sigma_thinning_model", "LID_model", "sigma_LID_model", "icelayerthick", "sigma_icelayerthick", "airlayerthick", "sigma_airlayerthick"])
        paleo_output[core] = core_df
        return paleo_output


if __name__ == "__main__":
    ch4import()
    paleochrono_import()

print(paleo_output['WDC'])