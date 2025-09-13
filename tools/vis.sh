#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$PYTHONPATH
python tools/maptr/vis_pred.py 
PYTHONPATH=$(pwd):$PYTHONPATH python tools/maptr/vis_pred.py projects/configs/maptr/maptr_nano_r18_110e.py work_dirs/maptr_nano_r18_110e/latest.pth
python tools/maptr/generate_video.py work_dirs/maptr_nano_r18_110e/vis_pred/
