import torch, sys, math, os
import torch.nn.functional as F
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, augment_batch, cross_entropy_q

# §22 Step 0: is the Tiny-ImageNet collapse a gradient-noise / weight-norm feedback loop in stage 0?
#  A (resolution): float-pretrain CIFAR-100 and Tiny-ImageNet for PRE epochs (seed 0, same recipe), then measure per
#    stage the gradient entering each BatchNorm's backward (= dL/d BN-output), scaled by S, in 2^-8 grid units:
#    median size and fraction below one grid step. Prediction: Tiny stage 0 (32x32) ~4x smaller than CIFAR stage 0
#    (16x16); Tiny stage 1 (16x16) ~ CIFAR stage 0.
#  B (loop): on the per-epoch Tiny checkpoints, per epoch and stage: mean conv-weight norm, the same BN-gradient
#    size, the conv weight-gradient cosine fixed point vs float64, and |cos(w, grad)| (radial error). Prediction:
#    in stage 0 the weight norm grows while the BN gradient shrinks and the cosine drops, together.
# Final recipe at f=8, S=2^16; RING_BITS=57 so late-epoch payloads do not wrap inside the probe.
# Usage: _loop_probe.py A | B <ckpt_prefix> <first_ep> <last_ep> [dataset]
MODE = sys.argv[1]; S = 2.0 ** int(os.environ.get('LP_LOGS', 16)); NB = int(os.environ.get('NB', 4))
dev = torch.device('cuda'); G = 2.0 ** -8
def setup(ds):
    simulate.set_dataset(ds); simulate.STAGE_BLOCKS = [5, 5, 5]; simulate.GLOBAL_AVG_POOL = True; simulate.RESIDUAL = True
    simulate.BITs = simulate.B_IV = 8; simulate.SOFTMAX_EXTRA_BITS = 9; simulate.BN_EARLY_DIV_BITS = 5
    simulate.DIV_PUBLIC_RING = 1; simulate.LIN_GRAD_PRESHIFT = 2; simulate.BN_STATS_F = 16; simulate.RING_BITS = 57
def stage(n):
    return 'stem' if n.startswith('stem') else ('stage%s' % n.split('.')[1] if n.startswith('stages') else 'head')
ORDER = ['stem', 'stage0', 'stage1', 'stage2']

def bn_grads(mb, batches):
    # per stage, over NONZERO g only: median |S*g| in grid steps and fraction < 1 step; g = dL/d(BN output), float64 ref
    store = []
    def mk(name):
        def fh(mod, i, o): o.register_hook(lambda g: store.append((name, g.detach())))
        return fh
    hs = [m.register_forward_hook(mk(n)) for n, m in mb.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    acc = {}
    for x, t in batches:
        store.clear(); mb.zero_grad(); F.cross_entropy(mb(x), t).backward()
        for n, g in store:
            a = (S * g).abs().flatten() / G; a = a[a > 0]      # exclude exact zeros (inactive ReLU after BN)
            acc.setdefault(stage(n), []).append((a.median().item(), (a < 1).double().mean().item()))
    for h in hs: h.remove()
    return {k: (sum(m for m, _ in v) / len(v), sum(f for _, f in v) / len(v)) for k, v in acc.items()}

PEAK = [0.0]
_rt = simulate.ring_truncate
def _spy(t, m):
    PEAK[0] = max(PEAK[0], t.abs().max().item()); return _rt(t, m)
simulate.ring_truncate = _spy   # payload peak over the fixed-point passes (reported in mode B)

def conv_stats(mb, mq, batches):
    # per stage: mean conv weight norm; fixed-vs-float64 grad cosine; mean |cos(w, g_fixed)|
    gf = {}; gq = {}; PEAK[0] = 0.0
    for x, t in batches:
        mb.zero_grad(); F.cross_entropy(mb(x), t).backward()
        mq.zero_grad(); (S * cross_entropy_q(mq(x), t)).backward()
        for (n, p), (_, q) in zip(mb.named_parameters(), mq.named_parameters()):
            if p.dim() == 4:
                gf[n] = gf.get(n, 0) + p.grad.detach(); gq[n] = gq.get(n, 0) + q.grad.detach() / S
    P = dict(mq.named_parameters()); out = {}
    for n in gf:
        w = P[n].detach(); a, b = gq[n], gf[n]
        cos = (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-300)
        rad = abs((w * a).sum().item()) / (w.norm().item() * a.norm().item() + 1e-300)
        out.setdefault(stage(n), []).append((w.norm().item(), cos, rad))
    return {k: tuple(sum(z[i] for z in v) / len(v) for i in range(3)) for k, v in out.items()}

def probe_batches(ds):
    torch.manual_seed(0); tl, _ = load_data(128); it = iter(tl)
    return [(x.to(dev).double(), y.to(dev).argmax(1)) for x, y in (next(it) for _ in range(NB))], tl

if MODE == 'A':
    PRE = int(os.environ.get('PRE', 3))
    for ds in ('cifar100', 'tinyimagenet'):
        setup(ds); batches, tl = probe_batches(ds)
        torch.manual_seed(0); m = CNNModelBaseline().to(dev)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4); m.train()
        for _ in range(PRE):
            for x, y in tl:
                opt.zero_grad(); F.cross_entropy(m(augment_batch(x.to(dev))), y.to(dev).argmax(1)).backward(); opt.step()
        mb = m.double(); mb.train(); r = bn_grads(mb, batches)
        print('A %-12s after %d float epochs | BN-input grad, median grid steps (fraction < 1 step): %s' % (
            ds, PRE, ' | '.join('%s %.1f (%.3f)' % (k, *r[k]) for k in ORDER if k in r)), flush=True)
else:
    prefix, e0, e1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); ds = sys.argv[5] if len(sys.argv) > 5 else 'tinyimagenet'
    setup(ds); batches, _ = probe_batches(ds)
    print('B ep | stage: weight-norm, BN-grad median steps (frac<1), grad cos vs float64, |cos(w,g)|', flush=True)
    for ep in range(e0, e1 + 1):
        sd = torch.load('%s.e%d' % (prefix, ep))['state']
        mq = CNNModelQ().to(dev); mq.load_state_dict(sd); mq.train()
        mb = CNNModelBaseline().to(dev).double(); mb.load_state_dict({k: v.double() for k, v in sd.items()}); mb.train()
        bg = bn_grads(mb, batches); cs = conv_stats(mb, mq, batches)
        print('B e%-2d S=2^%d peak 2^%.2f | ' % (ep, math.log2(S), math.log2(PEAK[0] or 1)) + ' | '.join('%s: |w| %.2f, %.1f (%.3f), cos %.3f, rad %.3f' % (k, cs[k][0], *bg[k], cs[k][1], cs[k][2])
                                         for k in ORDER if k in cs and k in bg), flush=True)
        del mq, mb; torch.cuda.empty_cache()
