#!/bin/bash
#SBATCH -w ltl-gpu05
#SBATCH --time=20:00:00


#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
# . /etc/profile.d/modules.sh                # Leave this line (enables the module command)
# module purge                               # Removes all modules still loaded
# module load rhel8/default-amp              # REQUIRED - loads the basic environment

conda init
# Source .bashrc to ensure conda is initialized
source ~/.bashrc
conda init

echo "===============Activating conda environment=================="
conda activate t5
echo ">Python version: $(which python)"
echo "===============Running script=================="

# cd /home/yz926/MUncertainty/lm-poly-test

bash run2.bash