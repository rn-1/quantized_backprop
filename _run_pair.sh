#!/bin/bash
# One paired run: float baseline, then the final fixed-point recipe at l=35 (f=8, S=2^S, A=9, K=5), same seed.
# Usage: _run_pair.sh <gpu> <seed> <tag> [log2 S]
#   env: DATASET (cifar100|cifar10|tinyimagenet), BLOCKS (e.g. 555), RESIDUAL (1|0), EPOCHS (default 80)
# Files: checkpoints/baseline_<tag>_s<seed>.pt, checkpoints/quant_<tag>_s<seed>.pt,
#        logs checkpoints/logs/<tag>_baseline_s<seed>.txt and <tag>_s<seed>.txt
# The float baseline is skipped if its snapshot already exists (so a quant rerun reuses it).
cd /home/ryan; g=$1; s=$2; tag=$3; LOGS=${4:-16}; L=checkpoints/logs
export CUDA_VISIBLE_DEVICES=$g
snap=checkpoints/baseline_${tag}_s$s.pt
[ -f $snap ] || python _baseline_snapshot.py $snap $s 2> $L/${tag}_baseline_s$s.txt || exit 1
QC_SEED=$s QC_RING=35 QC_A=9 QC_K=5 QC_DIVRING=1 QC_PRESHIFT=2 QC_FXOPT=1 QC_FW=20 QC_FV=16 QC_BNF=16 QC_BNDIAG=1 \
  QC_SAVE=checkpoints/quant_${tag}_s$s.pt python _quant_converged.py 8 $snap $LOGS > $L/${tag}_s$s.txt 2>&1
