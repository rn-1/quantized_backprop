import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

dev = torch.device('cuda')

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

# cache a full epoch + test set, identical across all configs
torch.manual_seed(0)
tl, te = load_data(128)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
log('CF cached %d train / %d test batches' % (len(train), len(test)))
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
    torch.manual_seed(0)
    ref = CNNModelBaseline().to(dev)
    if quantized:
        model = CNNModelQ().to(dev); sync_models(ref, model)
    else:
        model = ref
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    pmax = {'m': 0.0}
    orig = simulate.p_truncate
    def spy(*ts):
        for t in ts:
            v = t.abs().max().item()
            if v > pmax['m']: pmax['m'] = v
        return orig(*ts)
    if quantized: simulate.p_truncate = spy
    try:
        losses = []
        for x, tgt in train:
            opt.zero_grad()
            loss = ce(model(x.to(dtype)), tgt)
            (scale*loss).backward()
            if scale != 1.0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None: p.grad /= scale
            opt.step()
            losses.append(loss.item())
    finally:
        simulate.p_truncate = orig
    t1, t5 = evaluate(model, dtype)
    pm = pmax['m']
    pmstr = ('2^%.1f (%.2f%% of 2^53)' % (math.log2(pm), 100*pm/2**53)) if pm > 0 else 'n/a'
    log('CF %-22s first %.3f last %.3f min %.3f | top1 %.2f top5 %.2f | ptrunc %s'
        % (tag, losses[0], losses[-1], min(losses), t1, t5, pmstr))
    return losses

log('CF === full-epoch confirmation: does loss-scaling + low BITs match BITs=22? ===')
run('baseline f32',          None, 1.0,           torch.float32, False)
run('BITs=22 noscale',       22,   1.0,           torch.float64, True)
run('BITs=13 S=2^18',        13,   float(2**18),  torch.float64, True)
run('BITs=10 S=2^18',        10,   float(2**18),  torch.float64, True)
simulate.BITs = 22
log('CF DONE')
