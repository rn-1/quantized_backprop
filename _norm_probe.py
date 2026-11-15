import torch, sys, math, json, os, time
import torch.nn.functional as F
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, augment_batch, cross_entropy_q

# §24: does normalisation keep the scales stable enough for ONE global S?
# For [5,5,5] with BatchNorm and without it (NORM=0: no BN, normaliser-free init = zero last conv per residual
# branch), float-train with the standard recipe (lr 0.1 cosine over 80 ep, aug) and snapshot at POINTS epochs.
# At each snapshot, on NB identical un-augmented batches, record per conv layer (input -> output order):
#   fwd  : median |activation| at the conv output, in 2^-8 grid steps (forward values are not scaled by S)
#   bwd  : median |S * dL/d(conv output)| in grid steps and the fraction below one step (over nonzero entries),
#          from the float64 reference -- computed once at S=1 and scaled, since S is an exact multiplier there
#   payload: the fixed-point model's peak truncation input at each S in LOGS_LIST (RING_BITS=57, nothing wraps)
# Window at ring l: each active layer needs log2 S = log2(T / median grad) to put its median gradient at T grid
# steps (T = 8, 16, 32); S_max = largest S with payload < 2^(l-1); window = log2 S_max - (worst layer's need). Usage: _norm_probe.py <out.json>
# env: POINTS (0,1,3,10,20), LOGS_LIST (10..24 step 2), NB (4), SUB (0.05), RING (35), LR (0.1), SMOKE (n batches)
OUT = sys.argv[1]
POINTS = [int(v) for v in os.environ.get('POINTS', '0,1,3,10,20').split(',')]
LOGS_LIST = [int(v) for v in os.environ.get('LOGS_LIST', '10,12,14,16,18,20,22,24').split(',')]
NB = int(os.environ.get('NB', 4)); SUB = float(os.environ.get('SUB', 0.05)); RING = int(os.environ.get('RING', 35))
LR = float(os.environ.get('LR', 0.1)); SMOKE = int(os.environ.get('SMOKE', 0))
dev = torch.device(os.environ.get('DEV', 'cuda')); G = 2.0 ** -8
def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()

simulate.set_dataset('cifar100'); simulate.STAGE_BLOCKS = [5, 5, 5]; simulate.GLOBAL_AVG_POOL = True; simulate.RESIDUAL = True
simulate.BITs = simulate.B_IV = 8; simulate.SOFTMAX_EXTRA_BITS = 9; simulate.BN_EARLY_DIV_BITS = 5
simulate.DIV_PUBLIC_RING = 1; simulate.LIN_GRAD_PRESHIFT = 2; simulate.BN_STATS_F = 16; simulate.RING_BITS = 57
torch.manual_seed(0); tl, _ = load_data(128)
train = [(x.to(dev), y.to(dev).argmax(1)) for x, y in tl]
if SMOKE: train = train[:SMOKE]
probe = [(x.double(), t) for x, t in train[:NB]]

PEAK = [0.0]; _rt = simulate.ring_truncate
def _spy(t, m):
    PEAK[0] = max(PEAK[0], t.abs().max().item()); return _rt(t, m)

def per_layer(mb):
    # float64 reference: per conv layer, forward |out| and backward |dL/d out| at S=1 (medians over nonzero, in steps)
    convs = [(n, m) for n, m in mb.named_modules() if isinstance(m, torch.nn.Conv2d)]
    fwd = {n: [] for n, _ in convs}; bwd = {n: [] for n, _ in convs}; hs = []
    for n, m in convs:
        def fh(mod, i, o, n=n):
            a = o.detach().abs().flatten() / G; a = a[a > 0]
            if a.numel(): fwd[n].append(a.median().item())
            o.register_hook(lambda g, n=n: bwd[n].append(g.detach().abs().flatten()))
        hs.append(m.register_forward_hook(fh))
    for x, t in probe:
        mb.zero_grad(); F.cross_entropy(mb(x), t).backward()
    for h in hs: h.remove()
    out = []
    for n, _ in convs:
        g_all = torch.cat(bwd[n]) / G; g = g_all[g_all > 0]
        out.append(dict(layer=n, fwd_med=sum(fwd[n]) / max(len(fwd[n]), 1) if fwd[n] else 0.0, g=g,
                        g_nz=g.numel() / max(g_all.numel(), 1), fwd_nz=1.0 if fwd[n] else 0.0))
    return out

def payload(sd, logS):
    mq = CNNModelQ().to(dev); mq.load_state_dict(sd, strict=False); mq.train()
    PEAK[0] = 0.0; simulate.ring_truncate = _spy
    for x, t in probe:
        mq.zero_grad(); (2.0 ** logS * cross_entropy_q(mq(x), t)).backward()
    simulate.ring_truncate = _rt
    return math.log2(PEAK[0] or 1)

results = []
for norm in (True, False):
    simulate.NORM = norm; name = 'BN' if norm else 'noBN'
    torch.manual_seed(0); m = CNNModelBaseline().to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    ep = 0; diverged = False
    for P in POINTS:
        while ep < P and not diverged:
            m.train()
            for x, t in train:
                opt.zero_grad(); loss = F.cross_entropy(m(augment_batch(x)), t)
                if not torch.isfinite(loss): diverged = True; break
                loss.backward(); opt.step()
            sch.step(); ep += 1
        if diverged:
            log('NP %s diverged in float before epoch %d' % (name, P)); results.append(dict(net=name, point=P, diverged=True)); break
        sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        mb = CNNModelBaseline().to(dev); mb.load_state_dict(sd); mb = mb.double(); mb.train()
        layers = per_layer(mb)
        # active layers: >= 1% nonzero backward gradient and a nonzero forward output (zero-init branches excluded)
        act = [l for l in layers if l['g_nz'] >= 0.01 and l['fwd_nz'] > 0 and l['g'].numel() > 0]
        gmed = [l['g'].median().item() for l in act]; fmed = [l['fwd_med'] for l in act]
        # per-layer log2 S needed to lift the median gradient to T grid steps; window = payload limit - worst layer
        need = {T: [math.log2(T / g) for g in gmed] for T in (8, 16, 32)}
        rows = [dict(logS=L, payload=payload(sd, L),
                     sub_med=sorted(((2.0 ** L * l['g']) < 1).double().mean().item() for l in act)[len(act) // 2]) for L in LOGS_LIST]
        ok_hi = [r['logS'] for r in rows if r['payload'] < RING - 1]; smax = max(ok_hi) if ok_hi else None
        win = {T: (smax - max(v)) if smax is not None else None for T, v in need.items()}
        spread_g = math.log2(max(gmed) / min(gmed)); spread_f = math.log2(max(fmed) / min(fmed))
        worst = act[max(range(len(act)), key=lambda i: need[16][i])]['layer']
        r = dict(net=name, point=P, n_layers=len(layers), n_active=len(act), S_max=smax,
                 need_log2S={T: dict(max=max(v), median=sorted(v)[len(v) // 2], min=min(v)) for T, v in need.items()},
                 window_bits=win, grad_spread_bits=spread_g, fwd_spread_bits=spread_f, fwd_med_min=min(fmed), fwd_med_max=max(fmed),
                 worst_layer=worst, per_layer=[dict(layer=l['layer'], g_med_S1=gm, fwd_med=fm) for l, gm, fm in zip(act, gmed, fmed)], rows=rows)
        results.append(r)
        log('NP %-4s e%-2d | %d/%d layers active | grad spread %.1f bits, fwd spread %.1f bits (medians %.1f..%.0f steps) | '
            'log2 S needed (T=16): worst %.1f (%s), median %.1f | S_max(l=%d)=2^%s | window T=8/16/32: %s'
            % (name, P, len(act), len(layers), spread_g, spread_f, min(fmed), max(fmed), max(need[16]),
               worst, sorted(need[16])[len(act) // 2], RING, smax, '/'.join('%s' % (('%.1f' % win[T]) if win[T] is not None else '-') for T in (8, 16, 32))))
        log('NP        payload by S: ' + ' '.join('2^%d:2^%.1f' % (row['logS'], row['payload']) for row in rows))
json.dump(results, open(OUT, 'w'), indent=1)
log('NP DONE -> %s' % OUT)
