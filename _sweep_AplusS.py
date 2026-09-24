import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

# Combined experiment: Method A (dedicated inv_var scale, B_IV = BITs + 8) + fixed public loss
# scaling S. Method A removes the FORWARD inv_var floor; S lifts the BACKWARD gradients off the
# grid (BITS_REPORT sec 8/9). Question: with the forward floor gone, how low can BITs go while
# still matching float32 top-1 -- and does BITs=8 (forward-floor-dead without Method A at var>2^16)
# become viable? Full epoch, real CIFAR-100 top-1/top-5.
dev = torch.device('cuda')
IV_EXTRA = 8

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(128)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
log('AS cached %d train / %d test batches' % (len(train), len(test)))
ce = torch.nn.CrossEntropyLoss()

def evaluate(model, dtype):
    model.train()  # quantized BN has no eval path; keep both in train-mode BN
    c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            logits = model(x.to(dtype))
            c1 += (logits.argmax(1) == tgt).sum().item()
            c5 += (logits.topk(5, 1).indices == tgt.unsqueeze(1)).any(1).sum().item()
            n += x.shape[0]
    return 100*c1/n, 100*c5/n

def run(tag, bits, scale, dtype, quantized):
    if quantized:
        simulate.BITs = bits
        simulate.B_IV = bits + IV_EXTRA          # Method A: inv_var scale (>= grid, floor >= 2^(2*(bits+8)))
    torch.manual_seed(0)
    ref = CNNModelBaseline().to(dev)
    if quantized:
        model = CNNModelQ().to(dev); sync_models(ref, model)
    else:
        model = ref
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    pmax = {'m': 0.0}
    orig = simulate.ring_truncate                # spy the core -> captures the inv_var transient too
    def spy(t, m):
        v = t.abs().max().item()
        if v > pmax['m']: pmax['m'] = v
        return orig(t, m)
    if quantized: simulate.ring_truncate = spy
    try:
        losses = []; nonfin = 0
        for x, tgt in train:
            opt.zero_grad()
            out = model(x.to(dtype))
            loss = ce(out, tgt)
            (scale*loss).backward()
            if scale != 1.0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None: p.grad /= scale
            opt.step()
            losses.append(loss.item())
            nonfin += int((~torch.isfinite(out)).sum().item())
    finally:
        simulate.ring_truncate = orig
    t1, t5 = evaluate(model, dtype)
    pm = pmax['m']
    ring = ('2^%.1f' % math.log2(pm)) if pm > 0 else 'n/a'
    eff = ('2^-%d' % (bits + int(round(math.log2(scale))))) if (quantized and scale > 1) else ('2^-%d' % bits if quantized else '-')
    log('AS %-20s last %.3f min %.3f | top1 %.2f top5 %.2f | eff_grad %s ring %s nonfin %d'
        % (tag, losses[-1], min(losses), t1, t5, eff, ring, nonfin))
    return (t1, t5)

log('AS === Method A + loss scaling: how low can BITs go (full epoch, top-1) ===')
b = run('baseline f32',       None, 1.0,          torch.float32, False)
run('BITs=22 noscale',        22,   1.0,          torch.float64, True)
run('BITs=13 S=2^12',         13,   float(2**12), torch.float64, True)   # report sweet spot anchor
run('BITs=11 S=2^14',         11,   float(2**14), torch.float64, True)
run('BITs=10 S=2^14',         10,   float(2**14), torch.float64, True)
run('BITs=10 S=2^16',         10,   float(2**16), torch.float64, True)
run('BITs=8  S=2^16',         8,    float(2**16), torch.float64, True)
run('BITs=8  S=2^18',         8,    float(2**18), torch.float64, True)
simulate.BITs = 22; simulate.B_IV = 30
log('AS baseline top1/top5 = %.2f / %.2f' % b)
log('AS DONE')
