#!/bin/bash
#SBATCH -c 16
#SBATCH -t 1-00:00
#SBATCH -p kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-15
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads

module load python
mamba activate /n/home09/wlt/scratch/Lab/wlt/ensemble_env
python run.py ${SLURM_ARRAY_TASK_ID}

