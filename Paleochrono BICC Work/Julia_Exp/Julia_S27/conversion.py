import numpy as np
import pandas as pd
import os

# Get the directory of the current script and import files
script_dir = os.path.dirname(os.path.abspath(__file__))
import_folder = os.path.join(script_dir, 'Formatting')
import_files = [f for f in os.listdir(import_folder) if f.endswith('.xlsx')]  # only .xlsx files

# Loop through each file in the import folder
for file_name in import_files:

    base_name = os.path.splitext(file_name)[0]  # remove xlsx

    #read the excel file
    file_path_import = os.path.join(script_dir, 'Formatting', f"{base_name}.xlsx")
    df = pd.read_excel(file_path_import)

    #export to S27 folder
    file_path_export = os.path.join(script_dir, 'S27', f"{base_name}.txt")
    df.to_csv(file_path_export, sep='\t', index=False)
    print(f"Converted {file_name} to {base_name}.txt")