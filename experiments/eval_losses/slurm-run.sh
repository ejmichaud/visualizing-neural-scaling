#!/bin/bash
#SBATCH --job-name=eval_losses
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=2
#SBATCH --time=0-20:00:00
#SBATCH --output=/om/user/ericjm/results/visualizing-neural-scaling/eval_losses/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/visualizing-neural-scaling/eval_losses/logs/slurm-%A_%a.err
#SBATCH --mem=50GB
#SBATCH --array=6

python /om2/user/ericjm/visualizing-neural-scaling/experiments/eval_losses/eval.py $SLURM_ARRAY_TASK_ID

