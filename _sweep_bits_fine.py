import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

dev = torch.device('cuda')
NB = 150          # batches per run
CKPTS = [0, 20, 40, 60, 90, 120, 149]
GRID = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# cache a fixed set of batches so every BITs sees identical data
torch.manual_seed(0)
tl, _ = load_data(128)
batches = []
it = iter(tl)
for _ in range(NB):
    x, y = next(it)
    batches.append((x.to(dev).to(torch.float64), y.to(dev).to(torch.float64)))
sys.stderr.write('SW cached %d batches\n' % len(batches)); sys.stderr.flush()

bce = torch.nn.BCEWithLogitsLoss()

def train_run(model):
    # wrap p_truncate to monitor the magnitude actually fed to the ring
    orig = simulate.p_truncate
    mon = {'max': 0.0}
    def spy(*ts):
        for t in ts:
            m = t.abs().max().item()
            if m > mon['max']: mon['max'] = m
        return orig(*ts)
    simulate.p_truncate = spy
    try:
        opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        losses = []
        for i in range(NB):
            x, y = batches[i]
            opt.zero_grad(); out = model(x); loss = bce(out, y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        wmax = max(p.abs().max().item() for p in model.parameters())
        return losses, mon['max'], wmax
    finally:
        simulate.p_truncate = orig

# --- baseline reference (float32, no quantization) ---
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev)
optb = torch.optim.SGD(mb.parameters(), lr=0.001, momentum=0.9)
bl = []
for i in range(NB):
    x, y = batches[i]; xb = x.to(torch.float32); yb = y.to(torch.float32)
    optb.zero_grad(); out = mb(xb); l = bce(out, yb); l.backward(); optb.step()
    bl.append(l.item())
sys.stderr.write('SW %-9s ' % 'BASELINE' + ' '.join('%7.4f' % bl[c] for c in CKPTS) + '  min %.4f\n' % min(bl))
sys.stderr.flush()

def verdict(losses, bl_min):
    finite = [x for x in losses if math.isfinite(x)]
    if not finite: return 'BLOWN'
    mn = min(finite)
    if mn > 0.68: return 'DEAD'
    if mn <= bl_min * 1.5: return 'WORKS'
    return 'PARTIAL'

bl_min = min(bl)
for bits in GRID:
    simulate.BITs = bits
    torch.manual_seed(0)
    mb2 = CNNModelBaseline().to(dev)   # fresh baseline init to copy from (same seed -> same weights)
    mq = CNNModelQ().to(dev)
    sync_models(mb2, mq)
    losses, pmax, wmax = train_run(mq)
    finite = [x for x in losses if math.isfinite(x)]
    row = ' '.join('%7.4g' % losses[c] for c in CKPTS)
    sys.stderr.write('SW BITs=%-2d %-7s %s  min %.4f  |w|max %.3g  ptrunc_max %.3g (2^%.1f, %.2f%% of 2^53, %.2f%% of 2^56)  nonfin %d\n'
        % (bits, verdict(losses, bl_min), row, (min(finite) if finite else float('nan')), wmax, pmax,
           (math.log2(pmax) if pmax > 0 else 0),
           100*pmax/2**53, 100*pmax/2**56,
           sum(1 for x in losses if not math.isfinite(x))))
    sys.stderr.flush()
simulate.BITs = 22
