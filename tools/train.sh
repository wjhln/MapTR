#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$PYTHONPATH
python tools/train.py projects/configs/maptr/maptr_nano_r18_110e.py --gpus 1 --deterministic