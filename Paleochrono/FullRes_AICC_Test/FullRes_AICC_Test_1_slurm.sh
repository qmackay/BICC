#!/bin/bash -l
# BEGIN SLRUM OPTIONS DECLARATIONS
# Export all environment variables
#SBATCH --export=ALL
#SBATCH --export=TMPDIR=/tmp
#SBATCH --propagate=NONE
# Your job name
#SBATCH --job-name=FullRes_AICC_Test
#
# Use current working directory
#SBATCH --chdir=/fs1/home/ceoas/mackayq/BICC/Paleochrono
#
#Output files for stdout and stderr
#SBATCH --output=FullRes_AICC_Test/FullRes_AICC_Test.o
#SBATCH --error=FullRes_AICC_Test/FullRes_AICC_Test.e
#
# END SLURM OPTIONS DECLARATIONS
#
PATH=/home/ceoas/mackayq/miniconda3_x86/envs/bicc/bin:/home/ceoas/mackayq/miniconda3_x86/condabin:/local/cluster/sge/bin:/local/cluster/sge/bin/lx-amd64:/usr/local/cuda/bin:/home/ceoas/mackayq/scripts:/home/ceoas/mackayq/bin:/local/ceoas/bin:/local/ceoas/x86_64/bin:/local/cluster/slurm/bin:/bin:/usr/bin:/usr/local/bin:/usr/X11R6/bin:/usr/X/bin:.
LD_LIBRARY_PATH=/local/ceoas/x86_64/lib:/local/ceoas/x86_64/lib64
export PATH
export R_LIBS=/home/ceoas/mackayq/R
#
#
#The following auto-generated commands will be run by the execution node.
#We execute your command via /usr/bin/time with a custom format
#so that the memory usage and other stats can be tracked; note that
#GNU time v1.7 has a bug in that it reports 4X too much memory usage
source /home/ceoas/mackayq/.bashrc
echo "  Started on:           " `/bin/hostname -s` 
echo "  Started at:           " `/bin/date` 
/usr/bin/time -f " \\tFull Command:                      %C \\n\\tMemory (kb):                       %M \\n\\t# SWAP  (freq):                    %W \\n\\t# Waits (freq):                    %w \\n\\tCPU (percent):                     %P \\n\\tTime (seconds):                    %e \\n\\tTime (hh:mm:ss.ms):                %E \\n\\tSystem CPU Time (seconds):         %S \\n\\tUser   CPU Time (seconds):         %U " \
~/miniconda3_x86/envs/bicc/bin/python paleochrono.py AICC2023-FullRes
echo "  Finished at:           " `date` 
