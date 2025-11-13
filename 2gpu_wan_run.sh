#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NNODES=1
export GPUS_PER_NODE=2
export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

srun \
    -p 'h01' \
    -K \
    -N $NNODES \
    --job-name=jano \
    --ntasks-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --export=ALL \
    bash infer.sh