import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, cross_entropy_q

# BITs=8 RESIDUAL DIAGNOSTIC. Forward softmax scaling (§13) improved BITs=8 but did not clear it, so the
# residual limiter is elsewhere in the forward grid. This localizes it: after training into the regime
# where BITs=8 hurts (late epochs), measure per LAYER how coarse the 2^-BITs grid is for (a) activations
# and (b) weights -- in units of the grid step. A tensor whose typical |value| is only a few grid steps
# carries <3 bits and is a candidate for per-tensor scaling; if it's activations -> per-layer activation
# scaling helps; if it's weights -> the weight grid binds (scaling weights needs compensation, harder).
dev = torch.device('cuda')
LR = 0.001
PROBE_EPOCHS = 6
N_BATCH = 128
CONFIGS = [(8, 2**16, 9)]
LAYERS = ['conv1','bn1','conv2','bn2','conv3','bn3','fc1','fc2','fc3']
WEIGHTS = {'conv1':'k1','conv2':'k2','conv3':'k3','fc1':'w1','fc2':'w2','fc3':'w3',
           'bn1':'bnw1','bn2':'bnw2','bn3':'bnw3'}

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
probe_x = train[0][0]
ce = torch.nn.CrossEntropyLoss()

def granularity(t, grid):
    a = t.detach().abs().flatten()
    nz = a[a > 0]
    if nz.numel() == 0: return (0.0, 0.0, 0.0, 0.0)
    steps = nz / grid                      # |value| in units of the grid step
    q = steps.float()
    if q.numel() > 8_000_000:              # torch.quantile caps ~16M; subsample for the percentiles
        q = q[torch.randint(0, q.numel(), (4_000_000,), device=q.device)]
    p10 = torch.quantile(q, 0.10).item()
    med = torch.quantile(q, 0.50).item()
    frac_lt4 = (steps < 4).float().mean().item()      # <4 steps => <2 bits of precision
    zero_frac = 1.0 - nz.numel() / a.numel()
    return (p10, med, frac_lt4, zero_frac)

for BITS, S, A in CONFIGS:
    S = float(S)
    simulate.BITs = BITS; simulate.B_IV = BITS; simulate.SOFTMAX_EXTRA_BITS = A
    grid = 2.0 ** -BITS
    torch.manual_seed(0)
    mq = CNNModelQ().to(dev); mb = CNNModelBaseline().to(dev); sync_models(mb, mq)
    oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=0.9)
    mq.train()
    for ep in range(PROBE_EPOCHS):
        for x, tgt in train:
            oq.zero_grad(); out = mq(x); loss = cross_entropy_q(out, tgt)
            (S * loss).backward()
            with torch.no_grad():
                for p in mq.parameters():
                    if p.grad is not None: p.grad /= S
            oq.step()

    # capture per-layer activations on a probe batch
    acts = {}
    orig = {}
    for nm in LAYERS:
        orig[nm] = getattr(mq, nm)
        def mk(nm, o):
            def f(*a, **k):
                out = o(*a, **k); acts[nm] = out
                return out
            return f
        setattr(mq, nm, mk(nm, orig[nm]))
    with torch.no_grad():
        _ = mq(probe_x)
    for nm in LAYERS: setattr(mq, nm, orig[nm])

    log('GD === BITs=%d S=2^%d A=%d  grid step = 2^-%d = %.2e  (after %d epochs) ===' %
        (BITS, int(math.log2(S)), A, BITS, grid, PROBE_EPOCHS))
    log('GD layer | ACT p10/med steps | ACT frac<4steps | ACT zero%% | WGT p10/med steps | WGT frac<4steps')
    for nm in LAYERS:
        ap10, amed, alt4, az = granularity(acts[nm], grid)
        w = getattr(mq, WEIGHTS[nm])
        wp10, wmed, wlt4, _ = granularity(w, grid)
        log('GD %-6s | %6.1f / %6.1f     | %6.1f%%         | %5.1f%%   | %6.1f / %6.1f     | %6.1f%%' %
            (nm, ap10, amed, 100*alt4, 100*az, wp10, wmed, 100*wlt4))

simulate.BITs = 22; simulate.B_IV = 22; simulate.SOFTMAX_EXTRA_BITS = 0
log('GD DONE')
