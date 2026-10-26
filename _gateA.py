import torch, sys, math
import simulate
from simulate import CNNModelQ, load_data, reset_op_counts, snapshot_op_counts, attach_relu_counter, cross_entropy_q

# GATE A (BITS_REPORT §19): analytic communication/round cost of ONE training step, per configuration.
# Simulator wall-clock is not a cost proxy; protocol cost is (op counts) x (per-op cost formulas in the
# ring width l). Op counts are measured here on the live model; payloads come from _gateB.py.
#
# Setting: 3-party replicated secret sharing, semi-honest (ABY3 / Falcon / Piranha family). All costs are
# TOTAL bits sent by all parties. The formulas are first-order and taken from the literature survey;
# the same formulas price every config, so the RATIOS are more reliable than the absolute numbers:
#   multiply (incl. conv/matmul output):  3l bits, 1 round (reshare; one per output element)
#   truncation, local TS{L} (SecureML/ABY3 trunc1-style):  +l online, 0 offline, 0 extra rounds;
#        correctness needs headroom: P(fail) per element ~ 2^(payload+1-l)
#   truncation, 1-bit slack TS{1} (Harth-Kitzerow et al., PoPETs 2025):  3l offline + 2l online
#        (0 online if fused into the multiply); exact while payload < 2^(l-1)
#   ReLU/DReLU (A2B MSB extraction + bit injection): 3*l*log2(l) + 6l bits, log2(l)+2 rounds
#        (low-end variant, carry-only MSB: 3*2l + 6l bits)
#   backward ReLU mask (grad x shared bit): 3l bits per element
#   inverse sqrt / exp / reciprocal: negligible bandwidth (per-channel / per-sample), ~log2(l)+12 rounds
# Rounds: SIMD calls are counted as sequential (an upper bound on the dependency depth).
B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
dev = torch.device('cuda')
simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True
simulate.BITs = B; simulate.B_IV = B; simulate.SOFTMAX_EXTRA_BITS = 9
torch.manual_seed(0)
tl, _ = load_data(128); x, y = next(iter(tl)); x = x.to(dev).double(); t = y.to(dev).argmax(1)
m = CNNModelQ().to(dev).train(); attach_relu_counter(m)
reset_op_counts(); (2.0**16 * cross_entropy_q(m(x), t)).backward(); c = snapshot_op_counts()
TR, RL, NL = c['p_truncate_elems'], c['relu_elems'], c['inv_sqrt_elems']
TRc, RLc, NLc = c['p_truncate_calls'], c['relu_calls'], c['inv_sqrt_calls']
print('op counts / step (batch 128): truncated elems %.1fM (%d calls), ReLU %.1fM (%d calls), nonlinear %d elems (%d calls)'
      % (TR/1e6, TRc, RL/1e6, RLc, NL, NLc))

def cost(l, trunc, relu_variant='high', fused=False):
    lg = math.log2(l)
    mult = 3*l*TR
    if trunc == 'L': tr_on, tr_off = l*TR, 0
    else:            tr_on, tr_off = (0 if fused else 2*l*TR), 3*l*TR
    relu_bits = (3*l*lg + 6*l) if relu_variant == 'high' else (3*2*l + 6*l)
    relu = relu_bits*RL + 3*l*RL                      # fwd DReLU + select, bwd mask
    online = mult + tr_on + relu; offline = tr_off
    rounds = TRc + RLc*(lg+2) + NLc*(lg+12)
    return online, offline, rounds

LAN = (10e9, 0.2e-3); WAN = (1e9, 70e-3)            # (bits/s, RTT s)
def t(bits, rounds, net): return bits/net[0] + rounds*net[1]

configs = [  # label, ring, trunc, fused, payload log2 (from gate B) for the TS{L} failure estimate
    ('field: l=64, f=16, no S, local trunc',   64, 'L', False, 41.0),
    ('ours@64: l=64, f=8, S=2^16, local trunc', 64, 'L', False, 32.5),
    ('ours: l=34, f=8, S=2^16, TS{1}',          34, '1', False, 32.5),
    ('ours: l=34, TS{1} fused',                 34, '1', True,  32.5),
    ('ours: l=32 (if -2 bits), TS{1} fused',    32, '1', True,  30.5),
]
ref = None
print('\n%-42s | %8s %8s | %6s | %7s | %8s %8s | %s' % ('config', 'on GB', 'off GB', 'rounds', 'vs ref', 'LAN s', 'WAN s', 'TS{L} fails/step'))
for lab, l, tr, fu, pay in configs:
    for rv in ['high', 'low']:
        on, off, rd = cost(l, tr, rv, fu)
        tot = on + off
        if ref is None: ref = {}
        if rv not in ref: ref[rv] = tot
        fails = TR * 2.0**(pay + 1 - l) if tr == 'L' else 0.0
        print('%-42s | %8.2f %8.2f | %6d | %6.2fx | %8.2f %8.1f | %s'
              % (lab + (' [relu-lo]' if rv == 'low' else ''), on/8e9, off/8e9, rd, ref[rv]/tot,
                 t(tot, rd, LAN), t(tot, rd, WAN), ('%.3g' % fails) if tr == 'L' else 'exact'))
print('\nper epoch (391 steps) at ref config: %.1f TB' % (cost(64, 'L')[0]*391/8e12))
