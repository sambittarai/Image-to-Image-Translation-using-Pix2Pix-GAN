#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="pix2pixtest"
#SBATCH --mail-user=sambitarai17@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load tensorflow/1.9.0
export PYTHONPATH=~/.local/lib/python3.6/site-packages/
python pix2pix_n.py

