#!/bin/bash -l
# Configuration Options
#SBATCH --account=turbine
#SBATCH --partition=tier3 

#SBATCH --job-name=wildlife_adam
#SBATCH --output=batchlogs/deformable.out
#SBATCH --error=batchlogs/deformable.err
#SBATCH --mail-type=ALL
#SBATCH --time=4-00:00:00
#SBATCH --mem=256g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34 #16
#SBATCH --gres=gpu:a100:1

# Load Software
source /home/hs7569/dsci601/bin/activate

# Move to repo root
# cd /home/hs7569/github/DSCI-601-Wildlife || exit 1

export PYTHONPATH=$(pwd)


# Run training
python3 modeling/detr/train_detr.py
# python -m inference.manual_evaluate
