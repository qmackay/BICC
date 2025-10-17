import subprocess
import os

# Get current and parent directories
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Define scripts relative to their locations
convert_script = 'conversion.py'
clean_script = 'Clean.py'
paleo_script = 'paleochrono.py'

# Run local script
subprocess.run(['python', convert_script], cwd=script_dir, check=True)

# Run parent directory scripts
subprocess.run(['python', clean_script, os.path.join(parent_dir, 'Julia_S27')], cwd=parent_dir, check=True)
subprocess.run(['python', paleo_script, os.path.join(parent_dir, 'Julia_S27')], cwd=parent_dir, check=True)