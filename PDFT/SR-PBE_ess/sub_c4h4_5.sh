#!/bin/bash
#SBATCH -J c4h4_5
#SBATCH -p normal_plus
#SBATCH --mem-per-cpu=2gb
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o c4h4_5.out

# load the environment
module purge
module load apps/anaconda3/2021.05
export PATH=$PATH:/public/software/apps/anaconda3/2021.05/bin

python c4h4_5.py > c4h4_5.out
