#!/bin/bash
# §21.6 follow-up: loss-scale window sweep with the BN-backward overflow fix. Seed 0, identical to bnf16_s0
# except QC_K=7 (BN_EARLY_DIV_BITS 5->7) and log2 S in {13..18}; softmax elevation held at A=9 (QC_A) so S is
# the only variable. Queue: whenever one of GPUs 0-3 is free (< 2 GB on 3 consecutive 10-min checks), launch
# the next 2 jobs on it (each ~5 GB). Order: 16 (fix-only reference), 15, 14, 17, 13, 18.
cd /home/ryan; L=checkpoints/logs; Q=(16 15 14 17 13 18)
declare -A ok
while [ ${#Q[@]} -gt 0 ]; do
  for g in 0 1 2 3; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used" -lt 2000 ]; then ok[$g]=$(( ${ok[$g]:-0} + 1 )); else ok[$g]=0; fi
    if [ ${ok[$g]} -ge 3 ] && [ ${#Q[@]} -gt 0 ]; then
      for j in 1 2; do
        [ ${#Q[@]} -eq 0 ] && break; S=${Q[0]}; Q=("${Q[@]:1}")
        CUDA_VISIBLE_DEVICES=$g QC_SEED=0 QC_K=7 QC_A=9 QC_RING=32 QC_DIVRING=1 QC_PRESHIFT=2 QC_FXOPT=1 QC_FW=20 QC_FV=16 QC_BNF=16 \
          QC_SAVE=checkpoints/quant_ssweep_S$S.pt nohup python _quant_converged.py 8 checkpoints/baseline_555.pt $S > $L/ssweep_S$S.txt 2>&1 &
        echo "$(date '+%m-%d %H:%M') LAUNCHED S=2^$S on GPU $g"
      done
      ok[$g]=-1000   # don't reuse this GPU (our own jobs now occupy it)
    fi
  done
  echo "$(date '+%m-%d %H:%M') queue: ${Q[*]:-empty} | ok: 0=${ok[0]} 1=${ok[1]} 2=${ok[2]} 3=${ok[3]}"
  [ ${#Q[@]} -gt 0 ] && sleep 600
done
echo "ALL S-SWEEP JOBS LAUNCHED $(date)"
