#!/bin/bash
#SBATCH -A nvr_elm_llm                 #account
#SBATCH -p polar3,polar,grizzly,polar4 #partition
#SBATCH -t 04:00:00                    #wall time limit, hr:min:sec
#SBATCH -N 1                           #number of nodes
#SBATCH -J t2i_image_generation        #job name
#SBATCH --array=0-0%64
#SBATCH --output=exp_wenkunh/supir/4k_cs/slurm_out/%A_%a.out
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu 16G

export LOGLEVEL=INFO
export PATH="/FirstIntelligence/home/wenkunh/miniconda3/envs/supir/bin:$PATH"

export TORCHRUN_PORT=$((SLURM_ARRAY_TASK_ID + 38344))
cd /FirstIntelligence/home/wenkunh

read -r -d '' cmd <<EOF
python -m .supir_launcher yaml=exp_wenkunh/supir/4k_cs/config.yaml task_id=${SLURM_ARRAY_TASK_ID}
EOF

srun bash -c "${cmd}"
