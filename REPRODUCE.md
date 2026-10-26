# Reproducing the results in `RESULTS.md`

Logs go to `checkpoints/logs/` (`L/`). Each run is one process on one GPU, about 5 GB and about 6 min per
epoch on an A100 (80 epochs ≈ 8.5 h). Several runs can share a GPU.

## 0. Common settings
Every fixed-point run uses `_quant_converged.py <BITs> <float snapshot> <log2 S>` plus environment knobs.
The **final recipe**:

```sh
RECIPE="QC_K=5 QC_DIVRING=1 QC_PRESHIFT=2 QC_FXOPT=1 QC_FW=20 QC_FV=16 QC_BNF=16"
```

| knob | meaning (simulate.py) |
|---|---|
| `QC_K` | `BN_EARLY_DIV_BITS`: public partial early division in BN backward |
| `QC_DIVRING` | `DIV_PUBLIC_RING`: ring-faithful public division (CE backward) |
| `QC_PRESHIFT` | `LIN_GRAD_PRESHIFT`: pre-truncation of the Linear backward gradient |
| `QC_FXOPT`, `QC_FW`, `QC_FV` | fixed-point SGD; master weights at 2^-FW, momentum at 2^-FV |
| `QC_BNF` | `BN_STATS_F`: BN running-stat precision |
| `QC_RING` | `RING_BITS` (ring width ℓ; wraps are simulated) |
| `QC_A` | softmax elevation A (default log2 S − 7) |
| `QC_SEED` | init and data-order seed; must match the snapshot's seed |
| `QC_SAVE` | path to save the final model |
| `QC_SITE`, `QC_EPOCHS` | per-site payload logging threshold (log2); stop early |

Newton refinement (`RECIP_NEWTON`, `ISQRT_NEWTON`) is on by default.

## 1. Float baselines (shared references, same seed, init and data order)
```sh
python _baseline_snapshot.py checkpoints/baseline_555.pt    0 2> L/baseline_snap.txt
python _baseline_snapshot.py checkpoints/baseline_555_s1.pt 1 2> L/baseline_snap_s1.txt
python _baseline_snapshot.py checkpoints/baseline_555_s2.pt 2 2> L/baseline_snap_s2.txt
```

## 2. Headline, C1 (`L/r35_s*.txt`)
```sh
for s in 0 1 2; do snap=checkpoints/baseline_555.pt; [ $s -gt 0 ] && snap=checkpoints/baseline_555_s$s.pt
  env CUDA_VISIBLE_DEVICES=$s QC_SEED=$s QC_RING=35 $RECIPE QC_SAVE=checkpoints/quant_r35_s$s.pt \
    python _quant_converged.py 8 $snap 16 > L/r35_s$s.txt 2>&1 &
done
```
`QC_RING=32` gives the ℓ=32 variant (`L/bnf16_s*.txt`).

## 3. Loss-scale controls and window, C2
```sh
# implicit lift only (S=2^7, A=0) at ℓ=32, and with A=9 at ℓ=57
env QC_RING=32 $RECIPE QC_BNF=0 python _quant_converged.py 8 checkpoints/baseline_555.pt 7 > L/final_s0_S7_ksctrl.txt 2>&1
env QC_RING=57 QC_A=9 $RECIPE QC_BNF=0 python _quant_converged.py 8 checkpoints/baseline_555.pt 7 > L/ksctrl_S7_A9_r57.txt 2>&1

# S window (A pinned at 9); K=7 in ssweep_S{13..18}, K=5/6 in the *_k5_r2 / *_k6 runs
for S in 13 14 15 16 17 18; do
  env QC_RING=32 QC_A=9 $RECIPE QC_K=7 python _quant_converged.py 8 checkpoints/baseline_555.pt $S > L/ssweep_S$S.txt 2>&1 &
done
```
The two S=2^7 controls predate `BN_STATS_F`, hence `QC_BNF=0`.

## 4. BN-statistics diagnostics, C5.4 (saved models from §2 / the ℓ=32 runs)
```sh
python _bn_eval_diag.py <seed>            # stored / batch / recalibrated / folded
python _bn_eval_float_recal.py <seed> <snapshot>
python _bn_stats_proxy.py <seed>          # in-training rule, 2^-8 vs 2^-16, replayed on the final model
```

## 5. Overflow localisation (`L/ovfhunt_s0.txt`)
```sh
env QC_SEED=0 QC_RING=32 $RECIPE QC_SITE=30 QC_EPOCHS=8 python _quant_converged.py 8 checkpoints/baseline_555.pt 16
```
This replay is seeded and draws no extra random numbers, so it reproduces `L/bnf16_s0.txt` exactly.

## 6. MP-SPDZ, C3 and C4
Setup (patch, build, data link): `mpspdz_patch/README.md`. Then, from the patched tree:

```sh
# C3 — gradient check of the patches (f=24) vs torch
./compile.py -R 128 gradcheck res precise f24 2 && ./emulate.x -R 128 gradcheck-res-precise-f24 > L/mpspdz/gradcheck_res.txt
python /path/to/repo/_mpspdz_gradcheck.py res

# C3 — training agreement, [2,2,2], 5 epochs, lr 0.05 (args: blocks f k LS epochs lr*1000 threads)
./compile.py -R 128 resnet_train 222 8 31 9 5 50 32  && ./emulate.x -R 128 resnet_train-222-8-31-9-5-50-32
./compile.py -R 128 resnet_train 222 8 31 0 5 50 32  && ./emulate.x -R 128 resnet_train-222-8-31-0-5-50-32
./compile.py -R 128 resnet_train 222 16 47 0 5 50 32 && ./emulate.x -R 128 resnet_train-222-16-47-0-5-50-32
python /path/to/repo/_mpspdz_match.py 222 <f> <LS> 5 0.05     # simulator counterpart: (8,9) (8,0) (16,0)

# C4 — one real 3-party training step (ℓ=35 build; the ℓ=64 control needs a RING_SIZE=64 build)
./compile.py -R 35 -E ring resnet_step 222 8 16 16 && Scripts/ring.sh -R 35 resnet_step-222-8-16-16
./compile.py -R 64 -E ring resnet_step 222 16 31 16 && Scripts/ring.sh -R 64 resnet_step-222-16-31-16
./compile.py -R 35 -E ring resnet_step 555 8 16 16   # static estimate only ([5,5,5])
```

Notes:
- `resnet_train` at f=16 needs k=47. With k=31 it overflows: 15 integer bits (§21.5).
- `resnet_train` at k=31 needs `-R 128` in the emulator, because division needs at least 71 bits.
- MP-SPDZ's `fit` halves the lr and re-initialises when test accuracy is below 1.2× chance. Failing runs
  therefore leave the shared recipe after their first failure.
