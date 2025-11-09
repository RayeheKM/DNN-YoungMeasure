#!/bin/bash
#SBATCH --job-name=Semi2D
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --partition=scavenger-gpu
#SBATCH --output=Semi2D.out  # Direct output to a file named slurm-<job_id>.out
#SBATCH --error=Semi2D.err   # Direct errors to a file named slurm-<job_id>.err

# Activate your conda environment
source /hpc/home/rkm41/miniconda3/bin/activate myPINN

# Run your Python script
python Semi2DFinal.py