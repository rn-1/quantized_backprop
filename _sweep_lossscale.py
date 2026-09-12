import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

dev = torch.device('cuda')
NB = 150
CKPTS = [0, 20, 40, 60, 90, 120, 149]
GRID = [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
S_BITS = 18                       # public loss-scale S = 2^18 (power of two -> free public shift in MPC)
S = float(2 ** S_BITS)

torch.manual_seed(0)
tl, _ = load_data(128)
batches = []
it = iter(tl)
for _ in range(NB):
    x, y = next(it)
    batches.append((x.to(dev).to(torch.float64), y.to(dev).argmax(1)))
sys.stderr.write('LS cached %d batches; loss-scale S = 2^%d\n' % (len(batches), S_BITS)); sys.stderr.flush()

ce = torch.nn.CrossEntropyLoss()

def train_run(model, scale):
    # scale=1.0 -> no loss scaling (reference); scale=S -> scaled loss, unscaled grads before step
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
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y)
            (scale * loss).backward()                       # every grad is scale x larger DURING backprop
            if scale != 1.0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= scale                 # unscale before the step -> update is identical
            opt.step()
            losses.append(loss.item())                      # log the true (unscaled) loss
        wmax = max(p.abs().max().item() for p in model.parameters())
        return losses, mon['max'], wmax
    finally:
        simulate.p_truncate = orig

# baseline float32 reference (no scaling needed)
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev)
optb = torch.optim.SGD(mb.parameters(), lr=0.001, momentum=0.9)
bl = []
for i in range(NB):
    x, y = batches[i]; xb = x.to(torch.float32)
    optb.zero_grad(); l = ce(mb(xb), y); l.backward(); optb.step()
    bl.append(l.item())
bl_min = min(bl)
sys.stderr.write('LS %-22s ' % 'BASELINE(f32)' + ' '.join('%7.3f' % bl[c] for c in CKPTS) + '  min %.3f\n' % bl_min)
sys.stderr.flush()

def verdict(losses):
    finite = [x for x in losses if math.isfinite(x)]
    if not finite: return 'BLOWN'
    mn = min(finite)
    if mn > 4.0: return 'DEAD'          # CE start ~ ln100 = 4.605
    if mn <= bl_min * 1.5: return 'WORKS'
    return 'PARTIAL'

for scale, tag in [(1.0, 'noscale'), (S, 'S=2^%d' % S_BITS)]:
    sys.stderr.write('LS --- %s ---\n' % tag); sys.stderr.flush()
    for bits in GRID:
        simulate.BITs = bits
        torch.manual_seed(0)
        mb2 = CNNModelBaseline().to(dev)
        mq = CNNModelQ().to(dev)
        sync_models(mb2, mq)
        losses, pmax, wmax = train_run(mq, scale)
        finite = [x for x in losses if math.isfinite(x)]
        row = ' '.join('%7.4g' % losses[c] for c in CKPTS)
        sys.stderr.write('LS BITs=%-2d %-7s %s %s  min %.4f  |w|max %.3g  ptrunc 2^%.1f (%.2f%% of 2^53)  nonfin %d\n'
            % (bits, verdict(losses), tag, row, (min(finite) if finite else float('nan')), wmax,
               (math.log2(pmax) if pmax > 0 else 0), 100*pmax/2**53,
               sum(1 for x in losses if not math.isfinite(x))))
        sys.stderr.flush()
simulate.BITs = 22
sys.stderr.write('LS DONE\n'); sys.stderr.flush()
