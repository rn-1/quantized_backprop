import torch, sys, math
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data,
                      cross_entropy_q)

# CERTAINTY CHECK for fixed-point validity + integer-bit budget.
# The simulation stores fixed-point values as int64 but repeatedly casts them to float64 (conv runs
# F.conv2d on int-valued float64; to_float / div_public / to_fixed_no_shift all cross int<->float).
# float64 represents integers EXACTLY only up to 2^53; past that, low bits vanish silently and the sim
# would be wrong while looking fine. We instrument the max |value| at every int<->float boundary to get
# the TRUE payload peak P (bits), confirm P << 2^53 (sim faithful), and read off the integer-bit budget
# that a real fixed-point dtype would have to hold. The prior sweep only spied ring_truncate; this also
# catches the loss-scaled backward transients (grad*S in div_public, BN backward) that bypass it.
dev = torch.device('cuda')
LR = 0.001
EPOCHS = 5                              # peak in the 12-epoch sweep landed ~e5; 5 captures it closely
N_BATCH = 128
LOG2N = 7
CONFIGS = [(13, 2**12), (10, 2**14)]   # hottest payload + the viable floor

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

# --- global peak trackers, one per boundary -------------------------------------------------
PEAK = {}
def rec(name, t):
    if t.numel() == 0: return
    v = t.abs().max().item()
    if v > PEAK.get(name, 0.0): PEAK[name] = v

_o_ring   = simulate.ring_truncate
_o_div    = simulate.div_public
_o_tofl   = simulate.to_float
_o_tofns  = simulate.to_float_no_shift
_o_tofxns = simulate.to_fixed_no_shift

def ring_truncate_t(tensor, m):
    rec('ring_truncate.in', tensor)          # pre-truncation conv/linear/BN accumulator @ ~2^(2*BITs)
    return _o_ring(tensor, m)
def div_public_t(tensor, divisor):
    rec('div_public.in', tensor)             # grad*S transient in CE / BN backward
    return _o_div(tensor, divisor)
def to_float_t(*tensors):
    for t in tensors:
        if t.dtype == torch.int64: rec('to_float.in', t)
    return _o_tofl(*tensors)
def to_float_no_shift_t(*tensors):
    for t in tensors:
        if t.dtype == torch.int64: rec('to_float_no_shift.in', t)
    return _o_tofns(*tensors)
def to_fixed_no_shift_t(*tensors):
    for t in tensors:                        # input here is the float64 conv accumulator itself
        rec('conv_accum(float64).in', t)
    return _o_tofxns(*tensors)

simulate.ring_truncate     = ring_truncate_t
simulate.div_public        = div_public_t
simulate.to_float          = to_float_t
simulate.to_float_no_shift = to_float_no_shift_t
simulate.to_fixed_no_shift = to_fixed_no_shift_t

torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
log('PK cached %d train batches, epochs=%d  (float64 exact-int limit = 2^53)' % (len(train), EPOCHS))

for BITS, S in CONFIGS:
    S = float(S)
    A = max(0, int(round(math.log2(S))) - LOG2N)
    simulate.BITs = BITS; simulate.B_IV = BITS; simulate.SOFTMAX_EXTRA_BITS = A
    torch.manual_seed(0)
    mb = CNNModelBaseline().to(dev); mq = CNNModelQ().to(dev); sync_models(mb, mq)
    oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=0.9)
    PEAK.clear()
    mq.train()
    for ep in range(EPOCHS):
        for x, tgt in train:
            oq.zero_grad(); out = mq(x); loss = cross_entropy_q(out, tgt)
            (S*loss).backward()
            with torch.no_grad():
                for p in mq.parameters():
                    if p.grad is not None: p.grad /= S
            oq.step()
    log('PK === BITs=%d S=2^%d A=%d ===' % (BITS, int(math.log2(S)), A))
    gmax = max(PEAK.values())
    for name in sorted(PEAK, key=lambda k: -PEAK[k]):
        v = PEAK[name]
        log('PK   %-26s peak = 2^%5.2f  (%.3e)' % (name, math.log2(v) if v > 0 else 0, v))
    log('PK   -> GLOBAL PAYLOAD PEAK P = 2^%.2f   headroom to 2^53 = %.1f bits' %
        (math.log2(gmax), 53 - math.log2(gmax)))
    # minimum ring width l for a real fixed-point deployment, at matched truncation-failure rate.
    # ring_truncate garbage rate ~ |x|/2^l (code comment: ~1e-6 at |x|=2^37 on l=57). To hold rate eps
    # need l >= log2(P) + log2(1/eps). Report a few operating points + 1 sign bit.
    Pb = math.log2(gmax)
    log('PK   integer-bit budget: P=%.1f bits payload + 1 sign' % Pb)
    for eps_exp in (10, 20, 30):
        log('PK     min ring l for trunc-fail<=2^-%d :  %.0f bits' % (eps_exp, math.ceil(Pb + eps_exp + 1)))

simulate.BITs = 22; simulate.B_IV = 22; simulate.SOFTMAX_EXTRA_BITS = 0
log('PK DONE')
