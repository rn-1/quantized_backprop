# Loss-scaled fixed-point MPC training at f=8 in a 35-bit ring: claims and evidence

All numbers come from the logs linked in each section; `L/` = `checkpoints/logs/`. The full narrative,
including dead ends and corrections, is in `BITS_REPORT.md` (§ references below). How to rerun everything:
`REPRODUCE.md`.

## Claim
A deep BatchNorm ResNet can be trained under MPC-faithful fixed-point arithmetic with **f=8 fractional
bits in a 35-bit ring**. The usual setting is f=16 in a 64-bit ring. Our result is **within 0.84 points
of float**, with zero ring wraparounds. What makes it work is a **static, public loss scale S=2^16**,
which costs no secure operations. The smaller ring cuts **measured communication by 1.61×** (MP-SPDZ,
semi-honest 3PC replicated ring).

## C1 — Accuracy at f=8, ℓ=35 (§21.8)
ResNet [5,5,5] (8.66M params), CIFAR-100. Recipe: 80 epochs, SGD with momentum 0.9, lr 0.1 cosine,
wd 5e-4, batch 128, standard augmentation. Float and fixed point use the same seed, data order and init.
Both are scored in eval mode with BN running statistics.

| seed | float | fixed point | gap | peak payload | wraps |
|---|---|---|---|---|---|
| 0 | 71.97 | 70.98 | −0.99 | 2^31.23 | 0 |
| 1 | 71.68 | 70.92 | −0.76 | 2^30.74 | 0 |
| 2 | 71.73 | 70.96 | −0.77 | 2^30.18 | 0 |
| **mean ± sd** | **71.79 ± 0.16** | **70.95 ± 0.03** | **−0.84 ± 0.13** | | **0** |

Logs: `L/r35_s{0,1,2}.txt`; float: `L/baseline_snap{,_s1,_s2}.txt`.

**Precision map:**

| quantity | precision |
|---|---|
| compute: activations, weights' compute copy, gradients | 2^-8 |
| master weights | 2^-20 |
| momentum | 2^-16 |
| BN running statistics | 2^-16 |

Everything lives in the same 35-bit ring. Storage precision costs no ring width and no communication
(§21.2: BN statistics at 2^-16 measured 0.09 MB *cheaper* per step).

## C2 — Loss scaling is the enabler (§19.7, §21.7)
| configuration (f=8) | e80 top-1 | log |
|---|---|---|
| implicit lift only (S = batch = 2^7, Keller & Sun's deferred 1/B) | 1.18 (collapse) | `L/final_s0_S7_ksctrl.txt` |
| same with softmax elevation A=9 (rules out the forward path) | 0.99 (collapse) | `L/ksctrl_S7_A9_r57.txt` |
| S=2^15 | 69.24 (K=5) / 68.97 (K=6) | `L/ssweep_S15_k5_r2.txt`, `L/ssweep_S15_k6.txt` |
| **S=2^16** | **70.95 ± 0.03** (3 seeds) | C1 |
| S=2^17, S=2^18 | 71.76, 71.75 (1 seed; need ℓ ≥ 33 / 34) | `L/ssweep_S17.txt`, `L/ssweep_S18.txt` |

- **Working window:** S=2^16 is the accuracy point. Lower S loses about 2 points at 2^15. At ≤ 2^14 it
  collapses with K=7 (`L/ssweep_S1{3,4}.txt`), and in an earlier pre-recipe run (`L/qcfix_b8_s14.txt`).
  2^14 with the final K=5 recipe is **untested**. Higher S trains but needs a wider ring.
- **K note:** K = `BN_EARLY_DIV_BITS`, a public partial early division in BN backward. At K=7 the lower
  edge rises: S=2^15 collapses (`L/ssweep_S15.txt`).

## C3 — An independent implementation reproduces C2 (§21.4–21.5)
MP-SPDZ 0.4.3's own fixed-point numerics (softmax, BN, probabilistic truncation, SGD), emulated.
Network [2,2,2], 5 epochs, same data, order and init as our simulator.

| configuration | MP-SPDZ e5 top-1 | simulator e5 top-1 |
|---|---|---|
| f=16, implicit 2^7 lift | 37.87 | 37.55 |
| f=8, implicit 2^7 lift | fails (best 2.7) | fails (1.00) |
| **f=8, S=2^16** | **38.79** | **38.91** |

- **Logs:** `L/mpspdz/emu_train_222_{f16_k47_ls0,f8_ls0,f8_ls9}.txt` and `L/mpspdz/sim_match_222_*.txt`.
- **Patches:** two MP-SPDZ bugs had to be fixed first (residual fan-out, strided-conv backward). Both are
  exact against torch at f=24: cosine 1.00000, relative error ≤ 1.5e-4 (`L/mpspdz/gradcheck_*.txt`).
  Patch: `mpspdz_patch/`.

## C4 — Measured communication (§21.1–21.2)
One training step, 3 parties on localhost, `replicated-ring-party.x`. Both VMs built with g++.

| network | config | ring | data sent (all parties) | static estimate | log |
|---|---|---|---|---|---|
| [2,2,2] | f=16, k=31 | 64 | 14.88 GB | 15.44 GB | `L/mpspdz/run_222_R64gcc.txt` |
| [2,2,2] | **f=8, k=16** | **35** | **9.25 GB (1.61×)** | 9.59 GB | `L/mpspdz/run_222_R35_stock.txt` |
| [5,5,5] | f=16, k=31 | 64 | — | 34.06 GB | `L/mpspdz/compile_step_555_R64.txt` |
| [5,5,5] | **f=8, k=16** | **35** | — | **21.08 GB (1.62×)** | `L/mpspdz/compile_step_555_R35.txt` |

Rounds are unchanged (−0.8%). Wall-clock time is equal on localhost (40.7 s vs 41.7 s). **We claim
communication, not speed.**

## C5 — The recipe (§19, §21.6)
Each item was found by ruling out a bug before blaming precision. Each costs ~nothing in MPC.
1. **Newton refinement** of the reciprocal (softmax normaliser) and of inverse sqrt (BN). The unrefined
   approximations were biased (Σp − 1 ≈ −0.006), which caused a late collapse.
2. **Payload control:**
   - partial early division in BN backward (K=5);
   - ring-faithful public division in CE backward;
   - a 2-bit pre-shift on the Linear gradient.
3. **Fixed-point optimizer:** master weights at 2^-20 and momentum at 2^-16. A fresh stochastic
   truncation publishes the 2^-8 compute copy each step.
4. **BN running statistics at 2^-16.** On the 2^-8 grid, 9% of channels end with variance ≤ 0. Effect:
   68.95 → 71.03 at ℓ=32 (`L/evalfix_s*.txt` vs `L/bnf16_s*.txt`).

## Ring width: why 35, not 32 (§21.6–21.7)
Worst payload over all runs: **2^32.14** (`L/ssweep_S16_k6.txt`, e4). The 1-bit-slack truncation is
correct only when the payload is below 2^(ℓ−1), so **ℓ ≥ 34**; ℓ=35 leaves 1.86 bits of headroom.

At ℓ=32, 2 of 5 runs at S=2^16 wrapped once. Training absorbed every wrap, but a wrapped step computes the
wrong value, so ℓ=32 is not claimed.

The tail is data-dependent, not a truncation fault. The overflow site was localised by a seeded replay
(`L/ovfhunt_s0.txt`): BN backward `grad_std·inv_var`, plus a near-miss tail at the conv backward.

**Consistency check:** seeds that never wrapped at ℓ=32 reproduce bit-identically at ℓ=35.

## What is *not* claimed
- **Not the first sub-64-bit training:** AriaNN and Falcon used Z_2^32 on small nets.
- **Not "8-bit training":** f=8 compute, with higher-precision storage.
- **No speed claim.**
- **No worst-case guarantee:** headroom is measured, not bounded.
- **Not a secure training run end to end:** training is MPC-faithful emulation; one step ran under a real
  protocol, semi-honest 3PC only. Malicious security (e.g. SPDZ2k) needs extra bits.
- **Not general:** one architecture and one dataset; MP-SPDZ cross-check on a smaller net for 5 epochs.

## Prior art (§20)
To our knowledge, no prior work uses a static, public loss scale in fixed-point secure **training** to
cut f or ℓ.
- Loss scaling itself is from float mixed precision (Micikevicius et al. 2018).
- **Nearest:**
  - BEACON (USENIX Sec '23): float-emulated training.
  - Truncation Untangled (ePrint 2024/1953): 32-bit, inference only.
  - Silenzio (arXiv 2504.17785): FHE, MLPs, dynamic scaling.
- **The norm:** "64-bit ring" for private training (SoK, S&P '23).
- **Keller & Sun:** their "f=8 diverges" (ICML'22) predates their 2023 max-pooling fix (ePrint 2023/1219)
  and was not revisited. Our C2/C3 controls do not depend on it.

## Corrections made during the project (disclose in methods)
1. **Evaluation protocol.** Fixed-point BN was always in training mode, so it scored with test-batch
   statistics and the eval path was dead code. The earlier headline of 70.77 is superseded.
2. **BN running statistics on the 2^-8 grid** (C5.4).
3. **Wrong headroom figures** stated during analysis (3.8, then 2.77 bits). Correct: 1.86 bits vs the
   worst observed payload.
