#!/bin/bash
# Tiny-ImageNet, one seed: float baseline (40 ep) then the final fixed-point recipe at l=35 (S=2^16, A=9, K=5),
# paired by seed. Usage: _run_tiny_seed.sh <gpu> <seed>
cd /home/ryan; g=$1; s=$2; L=checkpoints/logs
export CUDA_VISIBLE_DEVICES=$g DATASET=tinyimagenet EPOCHS=40
python _baseline_snapshot.py checkpoints/baseline_tiny_s$s.pt $s 2> $L/tiny_baseline_s$s.txt || exit 1
QC_SEED=$s QC_RING=35 QC_A=9 QC_K=5 QC_DIVRING=1 QC_PRESHIFT=2 QC_FXOPT=1 QC_FW=20 QC_FV=16 QC_BNF=16 \
  QC_SAVE=checkpoints/quant_tiny_r35_s$s.pt python _quant_converged.py 8 checkpoints/baseline_tiny_s$s.pt 16 > $L/tiny_r35_s$s.txt 2>&1
