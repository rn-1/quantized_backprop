#!/bin/bash
# §21.6 overflow hunt: wait for ANY of GPUs 0-3 to be free (< 2 GB used on 3 consecutive checks, 10 min apart),
# then replay the bnf16 seed-0 run through e8 with per-site payload logging (threshold 2^30).
cd /home/ryan; L=checkpoints/logs; ok=0; g=-1
while [ $ok -lt 3 ]; do
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<=3 && $2<2000 {print $1; exit}')
  if [ -n "$free" ] && { [ $g -eq -1 ] || [ "$free" -eq $g ]; }; then g=$free; ok=$((ok+1)); else ok=0; g=-1; fi
  echo "$(date '+%m-%d %H:%M') free_gpu=${free:-none} ok=$ok"; [ $ok -lt 3 ] && sleep 600
done
CUDA_VISIBLE_DEVICES=$g QC_SEED=0 QC_K=5 QC_RING=32 QC_DIVRING=1 QC_PRESHIFT=2 QC_FXOPT=1 QC_FW=20 QC_FV=16 QC_BNF=16 \
  QC_SITE=30 QC_EPOCHS=8 nohup python _quant_converged.py 8 checkpoints/baseline_555.pt 16 > $L/ovfhunt_s0.txt 2>&1 &
echo "LAUNCHED ovfhunt on GPU $g $(date)"
