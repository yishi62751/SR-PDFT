#!/bin/bash
#SBATCH -J chk_pbe
#SBATCH -p normal_plus
#SBATCH --mem-per-cpu=2gb
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o chk_pbe.out

# load the environment
module purge
module load apps/anaconda3/2021.05
export PATH=$PATH:/public/software/apps/anaconda3/2021.05/bin

python chk_pbe.py > chk_pbe.out
