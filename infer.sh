#!/bin/bash
set -x

# Basic environment setup
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')
export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export RANK=$SLURM_PROCID
export LOCAL_RANK=$(expr $SLURM_PROCID % $GPUS_PER_NODE)    
export WORLD_SIZE=$GPUS_PER_NODE

export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT="./run_wan/jano_generate.py"
# SCRIPT="./run_wan/pab_generate.py"
ARGS=""
EXTRA_ARGS=""

exec python $SCRIPT $ARGS $EXTRA_ARGSdui