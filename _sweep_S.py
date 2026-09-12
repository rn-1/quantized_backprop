import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

dev = torch.device('cuda')
BITS = 13
LOG2_S = [0, 4, 8, 10, 12, 14, 16, 18, 22, 26, 30]   # 0 = noscale (lowest failure end)

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

log('SS device %s  BITs=%d  full-epoch S-exponent sweep' % (dev, BITS))

torch.manual_seed(0)
tl, te = load_data(128)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
log('SS cached %d train / %d test batches' % (len(train), len(test)))
ce = torch.nn.CrossEntropyLoss()

def evaluate(model, dtype):
    model.train()   # quantized BN has no eval path
    c1 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            c1 += (model(x.to(dtype)).argmax(1) == tgt).sum().item(); n += x.shape[0]
    return 100 * c1 / n

# baseline reference (float32)
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev)
ob = torch.optim.SGD(mb.parameters(), lr=0.001, momentum=0.9)
bl = []
for x, tgt in train:
    ob.zero_grad(); l = ce(mb(x.to(torch.float32)), tgt); l.backward(); ob.step(); bl.append(l.item())
b1 = evaluate(mb, torch.float32)
log('SS baseline(f32)   last %.3f  min %.3f  top1 %.2f' % (bl[-1], min(bl), b1))

def run(log2s):
    simulate.BITs = BITS
    S = float(2 ** log2s)
    torch.manual_seed(0)
    ref = CNNModelBaseline().to(dev)
    mq = CNNModelQ().to(dev); sync_models(ref, mq); mq.train()
    opt = torch.optim.SGD(mq.parameters(), lr=0.001, momentum=0.9)
    pmax = {'m': 0.0}
    orig = simulate.p_truncate
    def spy(*ts):
        for t in ts:
            v = t.abs().max().item()
            if v > pmax['m']: pmax['m'] = v
        return orig(*ts)
    simulate.p_truncate = spy
    try:
        losses = []
        for x, tgt in train:
            opt.zero_grad(); loss = ce(mq(x), tgt)
            (S * loss).backward()
            if log2s > 0:
                with torch.no_grad():
                    for p in mq.parameters():
                        if p.grad is not None: p.grad /= S
            opt.step(); losses.append(loss.item())
    finally:
        simulate.p_truncate = orig
    t1 = evaluate(mq, torch.float64)
    pm = pmax['m']
    nonfin = sum(1 for v in losses if not math.isfinite(v))
    finite = [v for v in losses if math.isfinite(v)]
    p2 = math.log2(pm) if pm > 0 else 0
    tag = 'noscale' if log2s == 0 else 'S=2^%d' % log2s
    log('SS %-9s eff.grad 2^-%-2d | last %7.4g min %7.4g | top1 %5.2f (d%+.2f) | ptrunc 2^%.1f (%.1f%% 2^53, %.1f%% 2^56) | nonfin %d'
        % (tag, BITS + log2s if log2s > 0 else BITS,
           (losses[-1] if losses else float('nan')), (min(finite) if finite else float('nan')),
           t1, t1 - b1, p2, 100 * pm / 2**53, 100 * pm / 2**56, nonfin))

for e in LOG2_S:
    run(e)
simulate.BITs = 22
log('SS DONE')
