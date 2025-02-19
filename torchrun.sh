#!/bin/sh

python -m torch.distributed.run --standalone --nproc_per_node=gpu main.py "$@"