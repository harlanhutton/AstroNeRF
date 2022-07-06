#!/bin/bash
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --constraint=a100
#SBATCH --mem 220000
#SBATCH --cpus-per-task=10
#SBATCH --job-name=train_bp
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=hhutton@flatironinstitute.org


python3 train_new.py --yaml=barfplanar_new

