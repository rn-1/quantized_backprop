import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

# How many extra bits does inv_var actually need? Sweep EXTRA = B_IV - BITs at the winning config
# (BITs=10, S=2^14). EXTRA=0 collapses inv_var back onto the grid (== Method A OFF) and should show
# the forward-floor fragility; larger EXTRA buys forward-floor margin but adds to the ring transient
# (amplified by S in the BN backward). Find the smallest EXTRA that still matches baseline.
dev = torch.device('cuda')
BITS = 8
S = float(2**16)

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(128)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
ce = torch.nn.CrossEntropyLoss()

def evaluate(model, dtype=torch.float64):
    model.train()
    c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            logits = model(x.to(dtype))
            c1 += (logits.argmax(1) == tgt).sum().item()
            c5 += (logits.topk(5, 1).indices == tgt.unsqueeze(1)).any(1).sum().item()
            n += x.shape[0]
    return 100*c1/n, 100*c5/n

def run(extra):
    simulate.BITs = BITS
    simulate.B_IV = BITS + extra
    torch.manual_seed(0)
    ref = CNNModelBaseline().to(dev)
    model = CNNModelQ().to(dev); sync_models(ref, model); model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    pmax = {'m': 0.0}
    orig = simulate.ring_truncate
    def spy(t, m):
        v = t.abs().max().item()
        if v > pmax['m']: pmax['m'] = v
        return orig(t, m)
    simulate.ring_truncate = spy
    try:
        losses = []; nonfin = 0
        for x, tgt in train:
            opt.zero_grad(); out = model(x); loss = ce(out, tgt)
            (S*loss).backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None: p.grad /= S
            opt.step(); losses.append(loss.item())
            nonfin += int((~torch.isfinite(out)).sum().item())
    finally:
        simulate.ring_truncate = orig
    t1, t5 = evaluate(model)
    pm = pmax['m']
    log('EX EXTRA=%-2d (B_IV=%2d, fwd floor var>2^%d) | top1 %.2f top5 %.2f | min %.3f ring 2^%.1f nonfin %d'
        % (extra, BITS+extra, 2*(BITS+extra), t1, t5, min(losses), math.log2(pm) if pm>0 else 0, nonfin))

# baseline
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev)
optb = torch.optim.SGD(mb.parameters(), lr=0.001, momentum=0.9)
for x, tgt in train:
    optb.zero_grad(); l = ce(mb(x.to(torch.float32)), tgt); l.backward(); optb.step()
b1, b5 = evaluate(mb, torch.float32)
log('EX baseline f32 top1 %.2f top5 %.2f' % (b1, b5))
log('EX === EXTRA sweep at BITs=%d S=2^%d (EXTRA=0 == Method A OFF) ===' % (BITS, int(math.log2(S))))
for extra in [0, 4, 8]:
    run(extra)
simulate.BITs = 22; simulate.B_IV = 30
log('EX DONE')
