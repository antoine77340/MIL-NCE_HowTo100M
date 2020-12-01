# export SEED_LST=$"42 68 99"
export OUT_HEAD="job_outputs/out-"
export ERR_HEAD="job_outputs/err-"
export SEED_LST="42 64"
export CAND_LST="1 4 7 10"

echo "TRAINING ON YOUCOOK2"
for CANDIDATE in $CAND_LST;
  do
  for SEED in $SEED_LST;
    do
    echo "______________________________"
    echo "Executing for seed=$SEED & candidate=$CANDIDATE"
    name="TRAIN-seed_$SEED"
    OUT_FILE="$OUT_HEAD$name.txt"
    ERR_FILE="$ERR_HEAD$name.txt"
    sbatch -J $name -o $OUT_FILE  -e $ERR_FILE -t 4:00:00 -p gpu --gres=gpu:2 --mem=40G train_single.sh $SEED $CANDIDATE
    echo "Done."
    exit
    done
  done
