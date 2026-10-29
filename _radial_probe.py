import torch, sys, math, os
import torch.nn.functional as F
import simulate
from simulate import CNNModelBaseline, CNNModelQ, load_data, cross_entropy_q

# §22 mechanism test. A conv feeding BatchNorm is scale-invariant, so in exact arithmetic its weight gradient
# is orthogonal to the weight: cos(w, dL/dw) = 0 (up to weight decay, which is applied separately). If BN's
# backward correction terms (the grad_mean / grad_std terms, each divided by n = batch*H*W) underflow in fixed
# point, the gradient gains a radial component and the weight norm drifts. Measure cos(w, g) per conv, float64
# vs fixed point (f=8, final recipe) at several S, on a saved checkpoint. Usage: _radial_probe.py <ckpt> [dataset]
CK = sys.argv[1]; DS = sys.argv[2] if len(sys.argv) > 2 else 'tinyimagenet'
dev = torch.device('cuda'); simulate.set_dataset(DS); simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True
simulate.BITs = simulate.B_IV = 8; simulate.SOFTMAX_EXTRA_BITS = 9; simulate.BN_EARLY_DIV_BITS = 5
simulate.DIV_PUBLIC_RING = 1; simulate.LIN_GRAD_PRESHIFT = 2; simulate.BN_STATS_F = 16; simulate.RING_BITS = 57
# knob overrides for the §22 fix search: RP_BIV (inv_var fractional bits), RP_K (BN early-division bits),
# RP_LOGS (comma list of log2 S). Payload peak (max |input| to ring_truncate, fixed-point passes) is reported.
simulate.B_IV = int(os.environ.get('RP_BIV', simulate.B_IV)); simulate.BN_EARLY_DIV_BITS = int(os.environ.get('RP_K', 5))
LOGS = [int(v) for v in os.environ.get('RP_LOGS', '14,16,18,20').split(',')]
PEAK = [0.0]; _rt = simulate.ring_truncate
def _spy(t, m):
    PEAK[0] = max(PEAK[0], t.abs().max().item()); return _rt(t, m)
torch.manual_seed(0); tl, _ = load_data(128); it = iter(tl); batches = [next(it) for _ in range(4)]
sd = torch.load(CK)['state']
mq = CNNModelQ().to(dev); mq.load_state_dict(sd); mq.train()
mb = CNNModelBaseline().to(dev).double(); mb.load_state_dict({k: v.double() for k, v in sd.items()}); mb.train()
convs = [n for n, p in mq.named_parameters() if p.dim() == 4]
def stage(n): return n.split('.')[1] if n.startswith('stages') else 'stem'
def radial(model, grads):
    P = dict(model.named_parameters())
    return {n: (P[n] * grads[n]).sum().item() / (P[n].norm().item() * grads[n].norm().item() + 1e-300) for n in convs}
def run(model, S, q):
    acc = {n: 0 for n in convs}
    for x, y in batches:
        x = x.to(dev).double(); t = y.to(dev).argmax(1); model.zero_grad()
        if q: (S * cross_entropy_q(model(x), t)).backward()
        else: F.cross_entropy(model(x), t).backward()
        for n, p in model.named_parameters():
            if n in acc: acc[n] = acc[n] + p.grad.detach().clone()
    return radial(model, acc)
rf = run(mb, 1, False)
print('ckpt %s | mean |cos(w,g)| per stage (and signed mean)' % os.path.basename(CK))
simulate.ring_truncate = _spy
rows = [('float64', rf)]
for L in LOGS:
    PEAK[0] = 0.0; r = run(mq, 2.0 ** L, True)
    rows.append(('fixed S=2^%d B_IV=%d K=%d [peak 2^%.1f]' % (L, simulate.B_IV, simulate.BN_EARLY_DIV_BITS, math.log2(PEAK[0] or 1)), r))
for name, r in rows:
    by = {}
    for n in convs: by.setdefault(stage(n), []).append(r[n])
    print('%-40s ' % name + ' | '.join('%s %.3f (%+.3f)' % (k, sum(abs(v) for v in vs) / len(vs), sum(vs) / len(vs)) for k, vs in by.items()))
