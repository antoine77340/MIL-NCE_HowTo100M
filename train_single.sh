#!/bin/bash

python3 main_distributed.py --verbose --n_display=1 \
       --multiprocessing-distributed --batch_size=256 \
       --num_thread_reader=40 --cudnn_benchmark=1 --pin_memory \
       --checkpoint_dir=milnce --num_candidates=$2 --lr=0.001 \
       --warmup_steps=10000 --epochs=1 --caption_root=csv
       --eval_video_root=data --seed=$1

# python main_distributed.py --verbose --n_display=1 \
#       --multiprocessing-distributed --batch_size=256 \
#       --num_thread_reader=40 --cudnn_benchmark=1 --pin_memory \
#       --checkpoint_dir=milnce --num_candidates=4 --resume --lr=0.001 \
#       --warmup_steps=10000 --epochs=300 --caption_root=path_to_howto_csv
#       --eval_video_path=.