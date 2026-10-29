# MP-SPDZ 0.4.3 patch (BITS_NOTEBOOK §21)

`mp-spdz-0.4.3-bits.patch` turns a stock MP-SPDZ 0.4.3 release into the tree used for the §21 cross-check.

## What it changes

| file | change |
|---|---|
| `Compiler/ml.py` | 4 patches, each marked `LOCAL PATCH` (see below) |
| `Programs/Source/resnet_train.mpc` | new: trains the `simulate.py` ResNet with MP-SPDZ numerics (§21.5) |
| `Programs/Source/resnet_step.mpc` | new: one training step, for static cost and the real 3PC run (§21.1–21.2) |
| `Programs/Source/gradcheck.mpc` | new: gradient check of patches 1–2 against torch (§21.4) |
| `CONFIG.mine` | new: g++ build against a user-space micromamba env; `RING_SIZE` defaults to 35 |

`Compiler/ml.py` patches:
1. **Residual gradient routing.** The stock optimizer assumes each output has one consumer. With fan-out,
   later consumers overwrote the gradient, and `Add` had no working backward. Now gradients are summed per
   consumer, and `Add` routes its gradient to both inputs. Bug fix.
2. **Stride-2 conv backward.** The stock code was correct only for stride 1. Now the output gradient is
   zero-dilated and the stride-1 code is reused, which is exact. Bug fix.
3. **Public loss scale, `LOSS_SCALE_BITS`.** The seed gradient is multiplied by 2^LS and the SGD update
   truncates by LS extra bits. Off by default.
4. **High-precision BN running stats, `BN_STATS_F`.** Running mean and var are held as sint at 2^-F with
   one truncation per update. Off by default. The 2^-f eval copy is not refreshed, so this is for cost
   checks only.

Patches 3 and 4 are off by default, so stock programs behave as before. Patches 1 and 2 change results
only for graphs that hit the bugs (fan-out, stride ≠ 1).

## Use

```sh
tar -xJf mp-spdz-0.4.3.tar.xz && cd mp-spdz-0.4.3      # github.com/data61/MP-SPDZ/releases/tag/v0.4.3
patch -p1 < /path/to/mp-spdz-0.4.3-bits.patch
ln -s /path/to/cifar/data data                          # simulate.load_data reads ./data
export SIMULATE_DIR=/path/to/repo                       # directory containing simulate.py

# build (no sudo): libraries from conda-forge
micromamba create -n mpspdz -c conda-forge libsodium gmp libboost-devel openssl yasm
make -j replicated-ring-party.x                         # ℓ=35; use RING_SIZE=64 (fresh tree) for the control
Scripts/setup-ssl.sh 3

# one secure training step, ℓ=35, f=8 (§21.2)
./compile.py -R 35 -E ring resnet_step 222 8 16 16      # expect: 9586.06 MB expected communication
Scripts/ring.sh -R 35 resnet_step-222-8-16-16           # measured: 9252.33 MB global
```

Program arguments are documented in each `.mpc` header. `resnet_train` runs in the emulator: compile
without `-E ring` and run `./emulate.x -R 128 ...`.

## Verified (2026-10-26)
- Applies cleanly to a fresh v0.4.3 tarball, and reproduces the working-tree files byte-for-byte.
- `resnet_step 222 8 16 16` compiled in the clean patched tree gives the same expected communication
  as the runs in §21.2 (9586.06 MB).
