#!/bin/bash
# Wait until GPUs 0-2 are free (each < 2 GB used on 3 consecutive checks, 10 min apart), then launch the
# 3-seed QC_BNF=16 rerun (§21.6): identical to evalfix_s* except BN running stats accumulate at 2^-16.
cd /home/ryan; L=checkpoints/logs; ok=0
while [ $ok -lt 3 ]; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1<=2 && $2>=2000' | wc -l)
  if [ "$busy" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
  echo "$(date '+%m-%d %H:%M') busy_gpus=$busy ok=$ok"; [ $ok -lt 3 ] && sleep 600
done
for s in 0 1 2; do snap=checkpoints/baseline_555.pt; [ $s -gt 0 ] && snap=checkpoints/baseline_555_s$s.pt
  CUDA_VISIBLE_DEVICES=$s QC_SEED=$s QC_K=5 QC_RING=32 QC_DIVRING=1 QC_PRESHIFT=2 QC_FXOPT=1 QC_FW=20 QC_FV=16 QC_BNF=16 \
    QC_SAVE=checkpoints/quant_bnf16_s$s.pt nohup python _quant_converged.py 8 $snap 16 > $L/bnf16_s$s.txt 2>&1 &
done
echo "LAUNCHED bnf16 runs $(date)"
