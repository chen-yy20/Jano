#!/bin/bash
set -x

SCRIPT=$1

srun \
    -p 'debug' \
    -K \
    -N 1 \
    --job-name=Jano \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --export=ALL \
    python $SCRIPT