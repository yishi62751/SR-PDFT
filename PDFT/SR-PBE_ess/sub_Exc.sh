#!/bin/bash
#SBATCH -J Exc
#SBATCH -p normal_plus
#SBATCH --mem-per-cpu=2gb
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o Exc.out

# load the environment
module purge
module load apps/anaconda3/2021.05
export PATH=$PATH:/public/software/apps/anaconda3/2021.05/bin

python Exc.py > Exc.out
