#!/bin/bash
# §21.7 follow-up: K = BN_EARLY_DIV_BITS runs. Seed 0, ssweep config (A=9, BNF=16) with QC_K and log2 S per job.
# Sharing GPUs with another user's job, so the criterion is FREE memory: launch one job on a GPU in 0-3 once it has
# >= 12 GB free on 3 consecutive 10-min checks. setsid fully detaches the job from this shell.
cd /home/ryan; L=checkpoints/logs; Q=("6 16" "6 15" "5 15")
declare -A ok
while [ ${#Q[@]} -gt 0 ]; do
  for g in 0 1 2 3; do
    free=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$free" -ge 12000 ]; then ok[$g]=$(( ${ok[$g]:-0} + 1 )); else ok[$g]=0; fi
    if [ ${ok[$g]} -ge 3 ] && [ ${#Q[@]} -gt 0 ]; then
      set -- ${Q[0]}; K=$1; S=$2; Q=("${Q[@]:1}"); tag=S${S}_k${K}; [ -f $L/ssweep_$tag.txt ] && tag=${tag}_r2
      CUDA_VISIBLE_DEVICES=$g QC_SEED=0 QC_K=$K QC_A=9 QC_RING=32 QC_DIVRING=1 QC_PRESHIFT=2 QC_FXOPT=1 QC_FW=20 QC_FV=16 QC_BNF=16 \
        QC_SAVE=checkpoints/quant_ssweep_$tag.pt setsid nohup python _quant_converged.py 8 checkpoints/baseline_555.pt $S > $L/ssweep_$tag.txt 2>&1 < /dev/null &
      echo "$(date '+%m-%d %H:%M') LAUNCHED K=$K S=2^$S on GPU $g -> ssweep_$tag.txt"; ok[$g]=0
    fi
  done
  echo "$(date '+%m-%d %H:%M') queue: ${Q[*]:-empty}"
  [ ${#Q[@]} -gt 0 ] && sleep 600
done
echo "ALL K RUNS LAUNCHED $(date)"
