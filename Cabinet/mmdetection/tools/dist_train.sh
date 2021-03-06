#!/usr/bin/env bash

# PYTHON=${PYTHON:-"python"}
PYTHON='/home/zhangzjn/anaconda3/bin/python3'

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
