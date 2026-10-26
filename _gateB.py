import torch, sys, math, time
import torch.nn.functional as F
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, cross_entropy_q

# GATE B (BITS_REPORT §19). Two checks on the [5,5,5] net, S=2^16, A=9:
#  (1) gradient fidelity -- load IDENTICAL weights into the float64 reference and the fixed-point model,
#      feed IDENTICAL batches (no augmentation), compare per-parameter gradients: cosine similarity and
#      norm ratio |g_q|/|g_f|. A correct simulator gives ratio ~1 and cos -> 1 as BITs grows (residual
#      only stochastic-truncation noise). A systematic ratio != 1 or a low cos at high BITs is a bug.
#      Done at two weight points: seed-0 init, and the converged float baseline (where the late
#      collapse happens).
#  (2) ring payload by phase -- max |value| entering ring_truncate, split forward / backward and by call
#      site, at BITs=8. A 1-bit-slack truncation in Z_2^l needs payload < 2^(l-1), so l=32 needs < 2^31.
# Usage: _gateB.py <snapshot.pt> [NBATCH]
import os
SNAP = sys.argv[1]; NB = int(sys.argv[2]) if len(sys.argv) > 2 else 6
S = 2.0**int(os.environ.get('GB_LOGS', 16)); A = 9
BITS_LIST = [int(b) for b in os.environ.get('GB_BITS', '8,10,13,16').split(',')]
POINTS = os.environ.get('GB_POINTS', 'init,final').split(',')
simulate.BN_EARLY_DIV_BITS = int(os.environ.get('GB_K', 0))   # §19.4 payload knob
simulate.DIV_PUBLIC_RING = int(os.environ.get('GB_DIVRING', 0))
simulate.LIN_GRAD_PRESHIFT = int(os.environ.get('GB_PRESHIFT', 0))
dev = torch.device('cuda')
simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True
simulate.SOFTMAX_EXTRA_BITS = A

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()
snap = torch.load(SNAP)
torch.manual_seed(0)
tl, _ = load_data(128)
batches = []
for i,(x,y) in enumerate(tl):
    if i >= NB: break
    batches.append((x.to(dev).to(torch.float64), y.to(dev).argmax(1)))

# ---- payload spy: max |input| into ring_truncate, keyed by (phase, call site) ----
PHASE = ['fwd']; PEAK = {}
orig_rt = simulate.ring_truncate
def spy(t, m):
    f = sys._getframe(1); site = getattr(f.f_code, 'co_qualname', f.f_code.co_name)
    if site == 'p_truncate': f = f.f_back; site = getattr(f.f_code, 'co_qualname', f.f_code.co_name) + '>p_trunc'
    site += ':%d' % f.f_lineno                          # simulate.py line of the call site
    k = (PHASE[0], site); v = t.abs().max().item()
    if v > PEAK.get(k, 0): PEAK[k] = v
    return orig_rt(t, m)

def make_models(point):
    torch.manual_seed(0)
    mb = CNNModelBaseline().to(dev)                     # seed-0 init (same as every quant run)
    if point == 'final':
        missing = mb.load_state_dict(snap['final_params'], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    mb = mb.double()
    mq = CNNModelQ().to(dev); sync_models(mb, mq)
    mb.train(); mq.train()
    return mb, mq

def grads_float(mb, x, t):
    mb.zero_grad(); F.cross_entropy(mb(x), t).backward()
    return {n: p.grad.detach().clone() for n,p in mb.named_parameters()}

def grads_q(mq, x, t):
    mq.zero_grad()
    PHASE[0] = 'fwd'; out = mq(x); loss = cross_entropy_q(out, t)
    PHASE[0] = 'bwd'; (S*loss).backward()
    return {n: (p.grad.detach()/S).clone() for n,p in mq.named_parameters() if p.grad is not None}

simulate.ring_truncate = spy
log('GB [5,5,5] S=2^%d A=9,' % int(math.log2(S)) + ' %d batches/point, float64 reference, no augmentation, BN_EARLY_DIV_BITS=%d DIV_PUBLIC_RING=%d' % (NB, simulate.BN_EARLY_DIV_BITS, simulate.DIV_PUBLIC_RING))
for point in POINTS:
    for BITS in BITS_LIST:
        simulate.BITs = BITS; simulate.B_IV = BITS; PEAK.clear()
        mb, mq = make_models(point)
        stats = {}                                       # name -> list of (cos, ratio)
        gdot = gqq = gff = 0.0
        for x,t in batches:
            gf = grads_float(mb, x, t); gq = grads_q(mq, x, t)
            for n in gf:
                a, b = gq.get(n), gf[n]
                if a is None: stats.setdefault(n, []).append((float('nan'), 0.0)); continue
                d = (a*b).sum().item(); na = a.norm().item(); nb_ = b.norm().item()
                gdot += d; gqq += na*na; gff += nb_*nb_
                stats.setdefault(n, []).append((d/(na*nb_+1e-300), na/(nb_+1e-300)))
        rows = [(n, sum(c for c,_ in v)/len(v), sum(r for _,r in v)/len(v)) for n,v in stats.items()]
        cos_all = gdot/math.sqrt(gqq*gff); ratio_all = math.sqrt(gqq/gff)
        cs = sorted(c for _,c,_ in rows); rs = sorted(r for _,_,r in rows)
        log('GB %-5s BITs=%-2d | global cos %.4f ratio %.4f | per-param cos min %.3f med %.3f | ratio min %.3f med %.3f max %.3f'
            % (point, BITS, cos_all, ratio_all, cs[0], cs[len(cs)//2], rs[0], rs[len(rs)//2], rs[-1]))
        for n,c,r in sorted(rows, key=lambda z: z[1])[:4]:
            log('GB      worst-cos  %-34s cos %.3f ratio %.3f' % (n, c, r))
        for n,c,r in sorted(rows, key=lambda z: -abs(math.log(max(z[2],1e-12))))[:3]:
            log('GB      worst-ratio %-33s cos %.3f ratio %.3f' % (n, c, r))
        # payload by phase
        for ph in ['fwd', 'bwd']:
            items = [(k[1], v) for k,v in PEAK.items() if k[0] == ph]
            if not items: continue
            top = max(items, key=lambda z: z[1])
            log('GB      payload %s max 2^%.2f  (%s)   ring needed (1-bit slack) >= %d'
                % (ph, math.log2(top[1]), top[0], math.ceil(math.log2(top[1]))+1))
        if BITS == 8:
            for (ph, site), v in sorted(PEAK.items(), key=lambda z: -z[1]):
                log('GB        b8 site %-3s %-40s 2^%.2f' % (ph, site, math.log2(v)))
        del mb, mq; torch.cuda.empty_cache()
simulate.ring_truncate = orig_rt
log('GB DONE')
