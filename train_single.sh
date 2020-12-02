#!/bin/bash

python3 main_distributed.py --n_display=1 \
       --batch_size=32 \
       --num_thread_reader=40 --cudnn_benchmark=1 --pin_memory \
       --checkpoint_dir=pmilnce --num_candidates=$2 --lr=0.001 \
       --warmup_steps=10000 --epochs=4 \
       --seed=$1 --world-size=1 --rank=0\
       --caption_root=data/small_train_captions \
       --train_csv=data/small_train_videos.csv --video_path=data/training \
       --word2vec_path $PWD/data/word2vec.pth \
       # --eval_video_root=data/small_val_videos --evaluate

# python main_distributed.py --verbose --n_display=1 \
#       --multiprocessing-distributed --batch_size=256 \
#       --num_thread_reader=40 --cudnn_benchmark=1 --pin_memory \
#       --checkpoint_dir=milnce --num_candidates=4 --resume --lr=0.001 \
#       --warmup_steps=10000 --epochs=300 --caption_root=path_to_howto_csv
#       --eval_video_path=.
