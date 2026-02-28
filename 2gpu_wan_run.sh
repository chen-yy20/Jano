#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NNODES=${NNODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

SCRIPT=${1:-"./run_wan/jano_generate.py"}
shift || true
PARTITION=${PARTITION:-h01} bash pysrun.sh "$SCRIPT" "$@"
