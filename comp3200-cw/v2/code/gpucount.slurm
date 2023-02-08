#!/bin/bash

#SBATCH -p lyceum
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=36G
#SBATCH --job-name=jeap1g20_swarm_learning
#SBATCH --time=00:01:00
#SBATCH --output=run.out
#SBATCH --error=run.err

source ~/env/bin/activate

python3 gpucount.py
