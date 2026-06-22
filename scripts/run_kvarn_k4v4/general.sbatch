#!/bin/bash

#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20 
#SBATCH --time=4:10:00
#SBATCH --mem=150GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|h200'
#SBATCH --job-name=general_h100_k4v4
#SBATCH --output=general_h100_k4v4-%j.out

echo "Running general FP16"
module purge

singularity exec --nv --fakeroot \
    --overlay /scratch/yw6594/sig/overlay-50G-10M-wsvd.ext3:ro \
    /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif\
    /bin/bash -c "source /ext3/env.sh; conda activate kvarn;
    cd /scratch/yw6594/quant/KVarN;
    EVALSCOPE_MODEL=Qwen/Qwen3-4B \
    EVALSCOPE_MAX_MODEL_LEN=32768 \
    EVALSCOPE_KV_CACHE_DTYPE=kvarn_k4v4_g128 \
    EVALSCOPE_BATCH_SIZE=20 \
    EVALSCOPE_ENABLE_THINKING=1 \
    bash myscript/evalscope/run_evalscope.sh general "