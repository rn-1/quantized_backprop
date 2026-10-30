# Training a ResNet under MPC with 8 fractional bits in a 35-bit ring

*Status 2026-10-29. Rerun instructions: `REPRODUCE.md`. Full chronological notes, including dead ends:
`BITS_NOTEBOOK.md`; "§N" in code comments refers to that notebook. Logs: `checkpoints/logs/` (`L/`).*

## 1. Summary

Secure multiparty computation (MPC) runs neural-network training on secret-shared fixed-point numbers.
Communication scales with the ring width ℓ, so a smaller ring is cheaper. Secure training normally uses
**16 fractional bits in a 64-bit ring**. The one reported attempt at 8 fractional bits diverged
(Keller & Sun 2022).

**We train with 8 fractional bits in a 35-bit ring and match float accuracy within a fraction of a point.**
The enabler is a **static, public loss scale S**. It multiplies the loss by a constant power of two, so
small gradients stay above the fixed-point grid. It costs no secure operations.

| setting | float | fixed point (f=8, ℓ=35) | gap |
|---|---|---|---|
| CIFAR-100, ResNet [5,5,5], **S=2^17** | 71.79 ± 0.16 | **71.57 ± 0.27** | **−0.22 ± 0.14** |
| CIFAR-100, S=2^16 | 71.79 ± 0.16 | 70.95 ± 0.03 | −0.84 ± 0.13 |
| CIFAR-10, S=2^17 | 93.45 ± 0.16 | 93.21 ± 0.07 | −0.24 ± 0.22 |
| CIFAR-10, S=2^16 | 93.45 ± 0.16 | 93.20 ± 0.06 | −0.25 ± 0.19 |
| Tiny-ImageNet (64×64), S=2^16 | 58.11 ± 0.07 | **collapses** | — (§7.1) |

Mean ± sd over 3 seeds; every fixed-point run is paired with a float run of the same seed. **Zero ring
wraparounds** in every CIFAR run at ℓ=35. On CIFAR-10 the gap is already small at 2^16, so 2^17 changes
nothing there.

- **Communication:** one real 3-party training step sends **1.61× less data** at ℓ=35 than the usual
  ℓ=64 (MP-SPDZ, semi-honest replicated ring). Rounds are unchanged.
- **An independent implementation agrees:** MP-SPDZ's own fixed-point arithmetic reproduces the
  loss-scaling effect epoch by epoch (§4).
- **Limits:** it fails at 64×64 resolution (a BatchNorm-backward precision breakdown, §7.1). It also
  depends on skip connections (§7.2).

## 2. Terms

| term | meaning |
|---|---|
| **f** | fractional bits. Values are integers times 2^-f; f=8 means a grid step of 1/256 |
| **ℓ** | ring width: every value is an ℓ-bit integer mod 2^ℓ. This is what communication pays for |
| **truncation** | after a multiply (2f fractional bits), drop f bits. In MPC this is an interactive protocol |
| **payload** | the largest value entering a truncation. The 1-bit-slack protocol we model is exact only if payload < 2^(ℓ−1) |
| **wrap** | a payload above that limit: the value silently becomes wrong, as it would in MPC |
| **S** | loss scale. Gradients are S× larger through the backward pass and removed in the update. A public power of two, so free |
| **A** | softmax elevation: probabilities are carried at 2^-(f+A) so small ones don't round to 0 (A=9) |
| **K** | bits of BatchNorm-backward's public division by n applied early, to cap the payload (K=5) |
| **F_W, F_V** | storage precision of master weights (20) and momentum (16); compute uses f=8 |

## 3. Setup

- **Model:** ResNet [5,5,5] (33 conv layers, widths 64/128/256, BatchNorm), global average pooling, and
  a small residual MLP head; 8.66M parameters.
- **Recipe, identical for float and fixed point:** SGD with momentum 0.9, lr 0.1 cosine over 80 epochs,
  weight decay 5e-4, batch 128, flip and crop augmentation.
- **Emulation, not a secure run.** `simulate.py` reproduces MPC arithmetic exactly:
  - every multiply is followed by stochastic truncation on the ℓ-bit ring, with wraps simulated;
  - BatchNorm's 1/√var, the softmax exp and reciprocal, and the optimizer are fixed-point protocols.
  - ReLU is computed in the clear, which is error-free (a sign select). Masking randomness and
    preprocessing are not modelled.
  - The real MPC run (§5) covers cost.
- **Where precision is spent:**

  | quantity | fractional bits |
  |---|---|
  | activations, weight compute copy, gradients | 8 |
  | momentum, BN running statistics | 16 |
  | master weights | 20 |

  All of it lives in one 35-bit ring. Storage precision costs no ring width and no communication.

## 4. Results

### 4.1 The loss-scale window (CIFAR-100, 3 seeds each; `L/stab_S*_s*.txt`, `L/r35_s*.txt`)

| S | top-1 | gap to float | peak payload (worst run) | headroom at ℓ=35 |
|---|---|---|---|---|
| 2^7 (implicit batch lift, as in Keller & Sun) | 1.2 | collapses | — | — |
| 2^14 | 63.04 ± 1.86 | −8.76 ± 1.73 | 2^33.19 | 0.8 bits |
| 2^15 | 69.90 ± 0.58 | −1.89 ± 0.74 | 2^30.32 | 3.7 bits |
| 2^16 | 70.95 ± 0.03 | −0.84 ± 0.13 | 2^31.23 | 2.8 bits |
| **2^17** | **71.57 ± 0.27** | **−0.22 ± 0.14** | 2^31.95 | **2.0 bits** |
| 2^18 | 71.75 ± 0.48 | −0.04 ± 0.58 | 2^32.53 | 1.5 bits |

- **Too little S starves the gradients:** S=2^7 diverges (`L/final_s0_S7_ksctrl.txt`), and accuracy
  degrades smoothly below 2^16.
- **More S raises the payload** by about 0.6 bits per doubling.
- **2^17 is the balanced operating point:** within 0.22 points of float, with 2 bits of headroom.
- 2^18 is statistically indistinguishable from float, with less headroom.

### 4.2 MP-SPDZ reproduces the effect (`L/mpspdz/`)

MP-SPDZ 0.4.3, using its own fixed-point softmax, BatchNorm, truncation and SGD in the emulator.
Network [2,2,2], 5 epochs, the same data, order and initialisation as our simulator:

| configuration | MP-SPDZ top-1 | our simulator |
|---|---|---|
| f=16, implicit 2^7 lift | 37.87 | 37.55 |
| f=8, implicit 2^7 lift | fails (≤ 2.7) | fails (1.0) |
| **f=8, S=2^16** | **38.79** | **38.91** |

MP-SPDZ needed two bug fixes to train a ResNet at all (residual fan-out, strided-conv backward). Both
match torch exactly at f=24 (gradient cosine 1.00000). Patch: `mpspdz_patch/`.

## 5. Communication (`L/mpspdz/run_222_*`, `compile_step_*`)

One training step, 3 parties, MP-SPDZ `replicated-ring-party.x`, semi-honest.

| network | config | data sent | |
|---|---|---|---|
| [2,2,2] | f=16, ℓ=64 | 14.88 GB | measured |
| [2,2,2] | **f=8, ℓ=35** | **9.25 GB (1.61×)** | measured |
| [5,5,5] | f=16, ℓ=64 | 34.06 GB | compiler estimate |
| [5,5,5] | **f=8, ℓ=35** | **21.08 GB (1.62×)** | compiler estimate |

- Rounds are unchanged (−0.8%).
- Wall-clock time is equal on localhost, where bytes are not the bottleneck. **We claim communication,
  not speed.**
- Storing BN statistics at 2^-16 changes the cost by −0.09 MB per step.

**Why ℓ=35:** MP-SPDZ itself requires ℓ ≥ 35 for f=8 with its k=16 integer range. Our worst observed
payload in any CIFAR run is 2^33.19, which needs ℓ ≥ 34. At ℓ=32, 2 of 5 S=2^16 runs wrapped once. Training
absorbed every wrap, but a wrapped step computes a wrong value, so we do not claim ℓ=32.

## 6. What it took

Each fix came from ruling out a bug before blaming precision. Each is free or nearly free in MPC.

| problem | fix | effect |
|---|---|---|
| backward gradients underflow the grid | **loss scale S** | the core enabler; f=22 → 8 |
| small softmax probabilities round to 0 | softmax elevation A=9 | keeps wrong-class gradients |
| a 50k-input fc layer put weights at ~1 grid step | global average pooling (new architecture) | removed the weight-grid limit |
| reciprocal (softmax) and 1/√var (BN) approximations biased −0.7% / −0.4% | one Newton step each | fixed a late collapse that had looked like a precision limit |
| payload hot spots in BN backward, CE backward and Linear backward | early partial division (K=5), ring-faithful public division, 2-bit pre-shift | backward payload 2^32.5 → ~2^29–31 |
| weight updates smaller than the 2^-8 grid | master weights at 2^-20, momentum at 2^-16 | +1.8 points over storing weights at 2^-8 |
| BN running statistics on the 2^-8 grid (9% of channels end with variance ≤ 0) | accumulate them at 2^-16 | 68.95 → 71.03 at ℓ=32 |

## 7. Limits and mechanisms

### 7.1 Higher resolution breaks BatchNorm backward (Tiny-ImageNet; `L/tiny*`, `L/tinydiag_*`)

At 64×64 every seed collapses around epoch 11 (at S=2^15, by epoch 3).
- **It is not an overflow.** A run at ℓ=57, where nothing can wrap, collapses identically.
- **What happens:** in stage 0 (the 32×32 stage), activation variance runs away (273 → 10^4) for
  ~10 epochs before the collapse, and the largest weight grows from 1.1 to 5.4.
- **Mechanism, measured.** A conv feeding BatchNorm is scale-invariant, so its exact gradient is
  orthogonal to its weights: cos(w, ∇w) = 0. Measured on the epoch-5 checkpoint (`_radial_probe.py`):

  | | stage 0 | stage 1 | stage 2 |
  |---|---|---|---|
  | float64 | 0.000 | 0.000 | 0.000 |
  | fixed point, Tiny-ImageNet (n = 2^17 per channel in stage 0) | **0.05** | 0.005 | 0.001 |
  | fixed point, CIFAR-100 (n = 2^15) | 0.006 | 0.001 | 0.001 |

  Fixed point breaks BatchNorm's scale invariance where the per-channel count n = batch × H × W is
  largest. The gradient gains a radial component, and weight and activation scale drift.
- **The same signature appears at too-small S on CIFAR:** at S=2^7, stage-0 pre-BN activations grow
  1.4× per step. We read the lower edge of the S window as this breakdown.
- **Raising S helps only slowly** (stage 0: ≈0.05 at 2^16, ≈0.04 at 2^18, 0.01–0.02 at 2^20).
- **The obvious BN knobs don't help** (`L/tiny_knob_probe.txt`; the noise level is about ±0.01):

  | knob | stage-0 radial error | payload |
  |---|---|---|
  | K = 0, 3 or 5 | unchanged (0.05–0.06) | K=0 raises it to 2^31.7 |
  | 1/σ precision of 10, 12 or 16 bits | unchanged | 16 bits → 2^40 |

  The error therefore comes from elsewhere. The next suspects are the 2^-8 grid of x_norm and of the
  gradient inside BN backward.
- **Status:** the claim holds for CIFAR-resolution BN statistics. Higher resolution needs a fix.

### 7.2 Skip connections are load-bearing (`_ablation_probe.py`, `L/ablation_probe.txt`, `L/plain555_s0.txt`)

- **The probe:** gradient sizes in grid units at S=2^16, on weights after 3 float epochs.
  - Skips make first-block gradients **2× larger at [2,2,2] and 7× at [5,5,5]**.
  - Without skips, 11% of first-block gradients at [5,5,5] fall below the grid, against 1.5% with skips.
    At [9,9,9] it is 82%, and fixed point no longer tracks float (cosine 0.09).
  - Skips cost ~1 bit of payload. The net effect is a wider S window.
- **Training without skips** ([5,5,5], 1 seed): float 48.6, **fixed point 21.7 (−27)**. Fixed point
  stalls during lr decay, as underflow predicts. The residual network loses 0.2–0.8.
- **Depth costs S even with skips:** first-block gradients roughly halve from [2,2,2] to [5,5,5] to
  [9,9,9]. Deeper networks will need a larger S.

## 8. Not claimed

- **Not the first sub-64-bit secure training:** AriaNN and Falcon used 32-bit rings, on small networks.
- **Not "8-bit training":** compute is f=8; storage is 16–20 bits.
- **No speed claim:** communication only.
- **No worst-case guarantee:** headroom is measured, not bounded.
- **Not secure training end to end:** training is emulated; one step ran under a real protocol,
  semi-honest 3-party only. Malicious security adds bits.
- **Not general yet:** one architecture family, CIFAR resolution, ResNet with skips.

## 9. Prior art

- Loss scaling comes from float mixed-precision training (Micikevicius et al. 2018).
- To our knowledge, no prior work uses a static, public loss scale to cut f or ℓ in fixed-point secure
  **training**.
- **Nearest work:**
  - BEACON (USENIX Sec '23): floating-point secure training.
  - Truncation Untangled (ePrint 2024/1953): 32-bit shares, **inference only**.
  - Silenzio (arXiv 2504.17785): FHE, MLPs, dynamic scaling.
- **The norm:** secure training "usually uses the 64-bit ring" (SoK, S&P '23).
- **Keller & Sun (ICML '22) report that f=8 diverges.** That predates their 2023 max-pooling bug fix
  (ePrint 2023/1219) and was not revisited. Our own controls (§4) show the divergence independently.

## 10. Corrections made during the project

These should be disclosed in a methods section.
1. **Evaluation protocol:** fixed-point BN always ran in training mode, so it was scored with test-batch
   statistics. An early headline of 70.77 is void.
2. **BN running statistics on the 2^-8 grid** (fixed; see §6).
3. **The late-training collapse attributed to precision** was the biased reciprocal (fixed; see §6).
4. **Headroom figures stated during analysis** (3.8 and 2.77 bits) were wrong; the table in §4.1 is
   correct.

## 11. Open items

1. A BatchNorm-backward fix for high resolution. K and 1/σ precision are ruled out; next, test higher precision for x_norm and the gradient inside BN backward (a code change), then a short training test.
2. Optional: a second architecture family; a malicious-security cost measurement.
