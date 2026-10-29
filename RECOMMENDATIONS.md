# Recommendations & TODOs — Fixed-Point Quantization (fp57 / BITs)

> **Superseded (2026-10-26).** Written before the final results; several recommendations and TODOs are
> obsolete. See `RESULTS.md` for current claims and `REPRODUCE.md` for how to rerun them.

*Actionable companion to `BITS_NOTEBOOK.md` (the lab notebook). Section references (§) point into
that report.*

---

## Recommendations

1. **Adopt `BITs = 22`** as the fixed-point format for this network. It is the **minimum** value
   that trains (§3); below it BatchNorm's `1/√var` underflows to 0, above it costs ring bits for
   nothing. 23–24 add no accuracy but eat into the 2^53 accumulation ceiling (36.7 % at BITs=24).

2. **Treat the usable window as narrow (22–24).** The precision floor and the accumulation ceiling
   nearly meet for this workload (§1, §3). Any change that raises activation variance or weight
   magnitude pushes toward the ceiling; any drop in precision hits the floor. Re-check the budget
   whenever the architecture or data scale changes.

3. **For a real SMPC deployment, budget for a 64–128-bit ring — not 57.** The 57-bit ring is
   enough for the *numerics* but not once statistical hiding (κ ≈ 40 bits on top of the ~2^44
   product magnitude → ~2^84) is required (§7.1). Treat the ~40-bit security margin as a
   first-class part of the width calculation. 128-bit is ≈2× the per-share communication of 64-bit.

4. **Keep the `p_truncate` ringfix assertions in place.** At BITs=22 the BN backward holds grad
   terms at ~2^44 before truncation — the path most exposed to the (fixed) negative-input bug in
   `p_truncate`. The reduce-into-ring step and the dtype assertion guard it; don't remove them.

5. **Use early stopping (~epoch 9 here).** It bounds the one unbounded quantity — the param RMSE
   drift (§6.2) — to the ≤3 % regime, and it is where test accuracy peaks anyway. The drift is
   functionally silent regardless, but early stopping keeps training out of the untested
   very-low-loss regime entirely.

6. **Use fixed public loss scaling to run below BITs=22 — confirmed −5 to −11 bits of ring (§9).**
   The §8 diagnosis (floor = small backward gradients underflowing) motivated fixed public loss
   scaling; §9 confirms it over a full epoch: **BITs=13 + S=2^18** and **BITs=10 + S=2^18** both
   match BITs=22 accuracy (top-1 17.0/17.1 vs 16.9) while truncating values 2^43 / 2^37 instead of
   2^48 — a **5–11-bit smaller ring**, which directly cuts per-share MPC communication. A power-of-
   two S is MPC-benign (free public-constant multiply, data-oblivious; unscale is a public shift).
   **Recommended sweet spot: BITs=13 + S=2^12** (refined from the S-exponent sweep, §9.2) — matches
   baseline (top-1 17.07 vs 17.08) at ptrunc 2^37 → **~−11-bit ring** vs BITs=22's 2^48, with a
   2-bit margin above the floor knee at log2 S=10. Keep S a power of two (free public shift). Do
   **not** stack S with high BITs, and do **not** over-scale: the usable window is log2 S ∈ [10, 22];
   below 10 gradients underflow, above ~26 ptrunc crosses the 2^53/2^56 ceiling (and the sim
   *understates* that failure — no ring-wraparound modeled). BITs=10 or log2 S=10 squeeze −13 bits
   but sit on a knee with no margin (BITs=10 also fragile on the forward `inv_var` floor `var>2^20`).
   **Validation status: locked in.** Multi-epoch fidelity confirmed at the refined S=2^12 (§9.1) —
   12 epochs tracking the baseline within noise (top-1 gap mean −0.01, worst −0.31 pts, no trend),
   ptrunc flat ~2^38 (≤0.01 % of 2^53), param RMSE 3.18 % (< BITs=22's 4.15 %). Degradation vs
   float32 at early-stop: top-1 −0.20 pts / top-5 −0.35 pts (0.5 % relative) — negligible. The
   6-bit-lower floor margin vs S=2^18 cost nothing (forward-grid quantization, not gradient
   precision, dominates weight drift at BITs=13).

7. **UPDATE — the floor is actually BITs=10 + S=2^14, not 13 (§10).** The "BITs=10 fragile on the
   forward `inv_var` floor" caveat in rec #6 was tested directly (the `EXTRA` sweep, §10.1) and does
   **not** bind on this net — variance stays O(1–100), far below even BITs=8's floor of `2^16`. With
   that caveat gone, **BITs=10 + S=2^14 is multi-epoch-validated** (§10.3): 12 epochs tracking the
   baseline within noise (top-1 gap mean **+0.08**, worst −0.20; e9 early-stop 40.69/70.46 vs
   40.88/70.53), 0 nonfinite, ring a *flat* ~2^34 (**−4 bits vs BITs=13**), param RMSE 4.76 % (the
   coarser 2^-10 forward grid drifts a bit faster than BITs=13's 3.18 %, still linear/bounded). This
   is the **new recommended aggressive floor**; keep **BITs=13 + S=2^12** as the conservative choice
   (lower drift, wider margin). **BITs=8 + S=2^16 is a soft floor** — trainable, 0 nonfinite, but
   ~0.5–1.1 pt short early and 2× the drift (RMSE 9.06 %); its gap is the forward activation/weight
   grid, not `inv_var`, so no inverse-variance fix recovers it. Method A (dedicated `inv_var` scale)
   was implemented but is **off by default** — it removes a floor this net never hits and only costs
   ring; Method B (floating `inv_var`) is shelved for the same reason (§10.1/§10.4).

---

## Changes already applied (`simulate.py`) ✅

- **`ring_truncate(tensor, m)` extracted from `p_truncate`** — the ring-truncation core is now a
  standalone public-shift-by-`m` primitive; `p_truncate` delegates to it and is behaviorally
  identical for every existing caller (verified: BITs=13 + S=2^12 12-epoch run reproduces §9.1 to the
  decimal). Enables truncating by an arbitrary public bit count (needed for Method A / asymmetric
  scales).
- **`B_IV` scale for `inv_var` ("Method A", default `B_IV = BITs` → OFF).** Raising `B_IV` above BITs
  decouples the forward `inv_var` floor from BITs (`var>2^(2·B_IV)`); left at `BITs` because §10 shows
  the floor never binds on this net and the extra bits only grow the ring. `inverse_sqrt` returns
  `inv_var` at `2^B_IV`; the four consumers truncate by `B_IV` via `ring_truncate`.

- **`BITs = 7 → 22`** (line 16).
- **Loss: `BCEWithLogitsLoss` → `CrossEntropyLoss`** in `train_model` — targets arrive one-hot
  `(N,100)` and are converted to class indices via `argmax(1)`. fc3 already emits raw logits, which
  is exactly what CE's internal log-softmax expects. (Fixes the base-rate collapse diagnosed in
  §6.1 — accuracy went from chance to 17 % top-1 in one epoch.)
- **MPC op-count instrumentation** — `OP_COUNTS` dict + `reset_op_counts` / `snapshot_op_counts` /
  `attach_relu_counter`; `p_truncate` and `inverse_sqrt` increment counters (§7.2).
- **Silenced per-forward debug prints** in `BatchNorm2dQ.forward` (`expected inv_var`, `x_norm`,
  `expected x_norm`, per-batch `report_error`) that flooded ~65 MB and slowed every run.
- **Stale `BITs=7`-specific comments made `BITs`-parametric:** eps-clamp ("rounds to 0 whenever
  eps < 2^-BITs"), inv_var underflow ("once num_element > 2^BITs"), and the BN-backward scale
  annotations (`2**(2*BITs)` / `2**BITs`, were hardcoded `2**14` / `2**7`).
- **Left intentionally:** the `/128` weight-init literals (overwritten by `sync_models`; init
  magnitude shouldn't track the fixed-point scale) and `fast_rsqrt` (dead code — see TODO below).

---

## Open TODOs

### Code hygiene
- [ ] **`fast_rsqrt` is dead code** (the Newton-Raphson / Quake-magic inverse sqrt at the top of
      `simulate.py`). The live inverse-variance path is `inverse_sqrt` → `pseudosep` (a polynomial
      approximation, *not* Newton-Raphson). Either delete `fast_rsqrt` or mark it `# UNUSED` so
      nobody assumes it is the active inv-sqrt. It also still contains a live `print`.
- [ ] Consider giving `BatchNorm2dQ` a **running-stats eval path** (`training=False`). Today it
      hardcodes `training=True`, so inference uses batch statistics; all accuracy numbers were
      measured with both models in train-mode BN to stay apples-to-apples (§6). A real eval path
      would let inference use running stats.

### Validation / follow-up experiments
- [x] **Gradient/loss scaling experiment — done, positive (§9).** Fixed public `S=2^18` lets
      BITs=13/10 match BITs=22 accuracy at 2^43 / 2^37 ptrunc (−5 / −11 bits of ring). See rec #6.
- [x] **Multi-epoch fidelity for the loss-scaled low-BITs config — done, holds (§9.1).** BITs=13 +
      S=2^18 over 12 epochs: accuracy within noise, ptrunc flat ~2^44, param RMSE 3.28 % (< BITs=22's
      4.15 %), 0 nonfinite. Ring saving confirmed over full training.
- [x] **Tune S — done (§9.2).** Full-epoch log2 S sweep at BITs=13: floor knee at log2 S=10, ceiling
      knee ~26–30, flat plateau [10,22]. Refined sweet spot BITs=13 + S=2^12 (−11-bit ring vs the
      −5 of the old 2^18 pick). 2^18 was over-spending ring by ~6 bits.
- [x] **12-epoch fidelity at the refined S=2^12 — done, holds (§9.1).** Accuracy within noise (mean
      top-1 gap −0.01 pts), ptrunc flat ~2^38, param RMSE 3.18 %. BITs=13 + S=2^12 locked in.
- [x] **Method A (decouple the forward `inv_var` floor) — done, shelved (§10).** Implemented
      (`ring_truncate` + `B_IV`); the forward floor is never binding on this net (`EXTRA` sweep: floor
      present ≡ removed, at BITs 8 and 10), so it buys no accuracy and only costs ring. Off by default.
- [x] **Push BITs below 13 with loss scaling — done (§10.2/§10.3).** BITs=10 + S=2^14 multi-epoch
      validated (matches baseline, ring ~2^34, 0 nonfinite). BITs=8 is a soft floor (~0.5–1.1 pt,
      2× drift). New recommended aggressive floor = BITs=10; see rec #7.
- [x] **Method B (floating `inv_var`) — assessed, not worth building here (§10.4).** Same target as
      Method A; ruled out by the `EXTRA` sweep (floor not binding) *and* §8 (inv-sqrt already the
      cleanest op). Keep as the robust upgrade to Method A's insurance for a higher-variance net.
- [ ] **Forward-grid precision is the next lever for < BITs=10 (§10.2).** BITs=8's gap is the 2^-BITs
      activation/weight grid, not `inv_var` — so try **per-tensor forward scaling** and/or
      **stochastic rounding with error feedback** on the weight grid (also targets the param-RMSE
      drift, which §9.1 pins on the forward grid). This is where a B-flavored idea (per-tensor / block
      float on gradients, generalizing `S`) would actually pay.
- [ ] **Re-check the BITs=10 floor once secure softmax/CE is modeled (§11.1).** The clear-float loss
      currently seeds an *exact* backward gradient; a real secure-exp/reciprocal would inject noise
      into exactly the backward gradients the floor depends on, and could move it.
- [ ] **Longer run at lower lr (or lr decay)** to reach genuinely low loss without the epoch-9
      overfitting wall, and watch whether the param-RMSE drift (§6.2, ~+0.33 %/epoch) stays linear
      or bends. Current runs reached loss 0.22, not ~0.
- [ ] **Correlate the RMSE drift with per-layer ptrunc rates** to see which layers contribute it
      (likely `fc1`, fan-in 50176). Would localise where truncation noise accumulates.

### SMPC realism — what this simulation does NOT model (for a faithful cost/security story)
- [ ] **Secure softmax / cross-entropy (PRIORITY — see BITS_NOTEBOOK §11.1).** `CrossEntropyLoss` runs
      on **clear float64**: the per-sample max, `exp`, sum, and reciprocal are exact, and so is the
      seed gradient `(softmax−onehot)/N` handed to `fc3` backward. A real run needs secure **max**,
      **exp**, and **reciprocal**. This is the *only* unmodeled op that injects error into the
      **backward gradients** — the exact quantity §8/§9 identify as the binding precision floor — so
      modeling it could shift the loss-scaling sweet spot and the BITs=10/13 floor. Highest-value
      primitive to add. **Skeleton in place:** `CrossEntropyLossQ` (autograd.Function) +
      `cross_entropy_q` wrapper in `simulate.py` — fixed-point forward/backward wired and
      grad-reachable (verified: matches torch CE to 1e-7 at BITs=22, scales with S, trains
      end-to-end). Three nonlinear primitives are **stubbed in the clear** and need real
      implementations: `secure_max` (DReLU/comparison tree), `fixed_exp` and `fixed_reciprocal`
      (range-reduce + polynomial, mirroring `inverse_sqrt`). Not yet swapped into `train_model`.
- [ ] **Fixed-point optimizer (momentum + update) (BITS_NOTEBOOK §11.2).** `optimizer.step()` is stock
      float64 SGD. `μ=0.9` and `lr=1e-3` are non-power-of-two public constants → each is a fixed-point
      public multiply **+ truncation** on the ring, and the momentum buffer must be ring-resident.
      Adds ~2 truncations/param-tensor/step (absent from §7.2) and a truncation-free momentum buffer
      that likely contributes to the param-RMSE drift (§6.2/§9.1). Mitigation: power-of-two `lr` makes
      `lr·v` a free shift. (BN running-stats momentum is *already* fixed-point, ~L716–724 — the
      pattern exists; the optimizer just isn't held to it.)
- [ ] **Modular ring arithmetic in the reductions.** Conv/Linear sums accumulate in **float64**,
      not on the ring — true wraparound is exercised only inside `p_truncate`, not in the large
      matmul accumulations. A faithful protocol would need the accumulation itself checked against
      the ring.
- [ ] **Masking security.** The mask `R = r1·2^m + r2` reproduces the *rounding* only; it provides
      no statistical hiding (this is why §7.1's κ margin is unmodelled). No MAC / malicious-security
      overhead, no share-reconstruction error.
- [ ] **Protocol-specific truncation error.** Current error is idealised probabilistic rounding,
      not any specific protocol's error profile.
- [ ] **Round-count model.** §7.2 counts SIMD op-invocations (`calls`) as a round-*structure*
      proxy; a real estimate needs the data-dependency graph to collapse independent truncations
      into shared rounds.

---

## Status of prior recommendations (closed)

- ✅ **Loss function fixed** — `CrossEntropyLoss` adopted; accuracy comparison now meaningful (§6).
- ✅ **Multi-epoch fidelity confirmed** — 12 epochs to loss 0.22 / ~40 % top-1: accuracy gap within
  noise (±0.66 pts), 0 nonfinite, ptrunc ≤5.2 % of the 2^53 ceiling. The representation does **not**
  fail at low loss over the tested range (§6.2).
- ✅ **"Would a better inverse-sqrt lower the required precision?" — answered: no** (§8, via
  `_noise_probe.py`). inverse-sqrt is the best-conditioned op in the graph (154 dB SNR, 0 %
  underflow, ~0 % of injected noise); the floor is set by small backward gradients. The dead-regime
  underflow onset is `var ≈ 2^(2·BITs)` — a pure representation threshold, independent of the
  approximation. Superseded by recommendation #6 (target backward-gradient scaling instead).
