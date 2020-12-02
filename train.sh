#!/bin/bash

# export SEED_LST=$"42 68 99"
export OUT_HEAD="job_outputs/out-"
export ERR_HEAD="job_outputs/err-"
export SEED_LST="42 64"
export CAND_LST="1 4 7 10"
# export NUM_NODE=1
export NUM_GRU_NODE=2

echo "TRAINING ON SMALL_HOWTO100M"
for CANDIDATE in $CAND_LST;
  do
  for SEED in $SEED_LST;
    do
    echo "______________________________"
    echo "Executing for seed=$SEED & candidate=$CANDIDATE"
    name="TRAIN-seed_$SEED"
    OUT_FILE="$OUT_HEAD$name.txt"
    ERR_FILE="$ERR_HEAD$name.txt"
    sbatch -J $name -o $OUT_FILE  -e $ERR_FILE -t 4:00:00 -p gpu --gres=gpu:$NUM_GRU_NODE --mem=16G train_single.sh $SEED $CANDIDATE
    echo "Done."
    exit
    done
  done
