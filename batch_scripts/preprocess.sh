#!/bin/bash -l
# Configuration Options
#SBATCH --account=turbine
#SBATCH --partition=debug

#SBATCH --job-name=wildlife_adam
#SBATCH --output=batchlogs/preprocess_dataset1.out
#SBATCH --error=batchlogs/preprocess_dataset1.err
#SBATCH --mail-type=ALL
#SBATCH --time=0-12:00:00
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34 #16

# Load Software
source /home/hs7569/dsci601/bin/activate

# Move to repo root
# cd /home/hs7569/github/DSCI-601-Wildlife || exit 1

export PYTHONPATH=$(pwd)


# Run training
python3 data/resize_dataset.py
# python -m inference.manual_evaluate
