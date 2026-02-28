#!/bin/bash
set -x

SCRIPT=${1:-$SCRIPT}
shift || true

if [ -z "$SCRIPT" ]; then
    echo "Usage: bash infer.sh <script.py> [script args...]"
    exit 1
fi

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

if [ -n "$SLURM_JOB_ID" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')}
else
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi

export RANK=${RANK:-${SLURM_PROCID:-0}}
export LOCAL_RANK=${LOCAL_RANK:-${SLURM_LOCALID:-0}}
export NODE_RANK=${NODE_RANK:-$(expr ${RANK} / ${GPUS_PER_NODE})}
export WORLD_SIZE=${WORLD_SIZE:-$(expr ${NNODES} \* ${GPUS_PER_NODE})}

export CUDA_DEVICE_MAX_CONNECTIONS=1

exec python "$SCRIPT" "$@"
