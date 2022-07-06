#!/bin/bash
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --constraint=a100
#SBATCH --mem 220000
#SBATCH --cpus-per-task=10
#SBATCH --job-name=gendata
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python3 generatedata.py --yaml=artpop

