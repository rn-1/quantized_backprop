import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

# Isolate Method A: sweep BITs low with the dedicated inv_var scale (B_IV) that decouples the
# FORWARD inverse-variance floor from BITs. No loss scaling here -- this measures what Method A
# alone buys, so the remaining floor should be the BACKWARD-gradient underflow (BITS_REPORT sec 8),
# NOT the forward inv_var death that BITS_REPORT sec 3 attributed the low-BITs DEAD regime to.
dev = torch.device('cuda')
NB = 150
CKPTS = [0, 40, 90, 149]
GRID = [6, 7, 8, 9, 10, 11, 12, 13, 16, 19, 22]

torch.manual_seed(0)
tl, _ = load_data(128)
batches = []
it = iter(tl)
for _ in range(NB):
    x, y = next(it)
    batches.append((x.to(dev).to(torch.float64), y.to(dev).argmax(1)))
sys.stderr.write('SW cached %d batches  (B_IV=%d)\n' % (len(batches), simulate.B_IV)); sys.stderr.flush()

ce = torch.nn.CrossEntropyLoss()

def train_run(model):
    orig = simulate.ring_truncate
    mon = {'max': 0.0}
    def spy(t, m):
        v = t.abs().max().item()
        if v > mon['max']: mon['max'] = v
        return orig(t, m)
    simulate.ring_truncate = spy
    try:
        opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        losses = []
        for i in range(NB):
            x, y = batches[i]
            opt.zero_grad(); out = model(x); loss = ce(out, y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        wmax = max(p.abs().max().item() for p in model.parameters())
        return losses, mon['max'], wmax
    finally:
        simulate.ring_truncate = orig

# baseline float32
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev)
optb = torch.optim.SGD(mb.parameters(), lr=0.001, momentum=0.9)
bl = []
for i in range(NB):
    x, y = batches[i]; xb = x.to(torch.float32)
    optb.zero_grad(); out = mb(xb); l = ce(out, y); l.backward(); optb.step()
    bl.append(l.item())
bl_min = min(bl)
sys.stderr.write('SW %-9s ' % 'BASELINE' + ' '.join('%7.4f' % bl[c] for c in CKPTS) + '  min %.4f\n' % bl_min)
sys.stderr.flush()

def verdict(losses, bl_min):
    finite = [x for x in losses if math.isfinite(x)]
    if not finite: return 'BLOWN'
    mn = min(finite)
    if mn > 4.55: return 'DEAD'            # never left ln(100)=4.605
    if mn <= bl_min * 1.05: return 'WORKS' # tracks baseline within 5%
    return 'PARTIAL'

METHOD_A = (len(sys.argv) < 2 or sys.argv[1] != 'off')   # 'off' -> B_IV=BITs (original behavior)
sys.stderr.write('SW Method A = %s\n' % ('ON (B_IV=%d)' % simulate.B_IV if METHOD_A else 'OFF (B_IV=BITs)')); sys.stderr.flush()
for bits in GRID:
    simulate.BITs = bits
    if not METHOD_A:
        simulate.B_IV = bits     # collapse inv_var scale back onto the global grid = pre-Method-A
    torch.manual_seed(0)
    mb2 = CNNModelBaseline().to(dev)
    mq = CNNModelQ().to(dev)
    sync_models(mb2, mq)
    losses, pmax, wmax = train_run(mq)
    finite = [x for x in losses if math.isfinite(x)]
    row = ' '.join('%7.4g' % losses[c] for c in CKPTS)
    sys.stderr.write('SW BITs=%-2d %-7s %s  min %.4f (base %.4f)  |w|max %.3g  ring_max 2^%.1f  nonfin %d\n'
        % (bits, verdict(losses, bl_min), row, (min(finite) if finite else float('nan')), bl_min, wmax,
           (math.log2(pmax) if pmax > 0 else 0),
           sum(1 for x in losses if not math.isfinite(x))))
    sys.stderr.flush()
simulate.BITs = 22
