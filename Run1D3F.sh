#!/bin/bash
#SBATCH --job-name=1D3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --partition=scavenger-gpu
#SBATCH --output=1D3.out  # Direct output to a file named slurm-<job_id>.out
#SBATCH --error=1D3.err   # Direct errors to a file named slurm-<job_id>.err

# Activate your conda environment
source /hpc/home/rkm41/miniconda3/bin/activate myPINN

# Run your Python script
python 1D3Final.py