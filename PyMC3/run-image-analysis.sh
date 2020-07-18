#!/bin/bash
#SBATCH --account=def-emoodie
#SBATCH --mem-per-cpu=8192M
#SBATCH --time=0-04:00
#SBATCH --cpus-per-task=16

export MKL_THREADING_LAYER=GNU
module load python/3.6
source ../../venv/bin/activate

python 5-image-analysis.py
