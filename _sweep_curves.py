import torch, sys, math, json
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data

dev = torch.device('cuda')
EPOCHS = 12
BITS = 13
LOG2_S = [8, 10, 12, 26]     # below-knee, knee, recommended, ceiling-pressure

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(128)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
log('CV cached %d train / %d test batches' % (len(train), len(test)))
ce = torch.nn.CrossEntropyLoss()

def evaluate(model, dtype):
    model.train()
    c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            lg = model(x.to(dtype))
            c1 += (lg.argmax(1) == tgt).sum().item()
            c5 += (lg.topk(5, 1).indices == tgt.unsqueeze(1)).any(1).sum().item()
            n += x.shape[0]
    return 100*c1/n, 100*c5/n

data = {'epochs': list(range(1, EPOCHS+1))}

# baseline once (deterministic; reused as the reference for every config)
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev)
ob = torch.optim.SGD(mb.parameters(), lr=0.001, momentum=0.9)
b_loss, b_t1, b_t5 = [], [], []
for ep in range(EPOCHS):
    mb.train()
    for x, tgt in train:
        ob.zero_grad(); l = ce(mb(x.to(torch.float32)), tgt); l.backward(); ob.step()
    t1, t5 = evaluate(mb, torch.float32)
    b_loss.append(l.item()); b_t1.append(t1); b_t5.append(t5)
    log('CV base e%-2d loss %.3f t1 %.2f t5 %.2f' % (ep+1, l.item(), t1, t5))
data['baseline'] = {'loss': b_loss, 't1': b_t1, 't5': b_t5}

for log2s in LOG2_S:
    simulate.BITs = BITS
    S = float(2**log2s)
    torch.manual_seed(0)
    ref = CNNModelBaseline().to(dev)
    mq = CNNModelQ().to(dev); sync_models(ref, mq)
    opt = torch.optim.SGD(mq.parameters(), lr=0.001, momentum=0.9)
    q_loss, q_t1, q_t5, q_pt = [], [], [], []
    pmax = {'m': 0.0}
    orig = simulate.p_truncate
    def spy(*ts):
        for t in ts:
            v = t.abs().max().item()
            if v > pmax['m']: pmax['m'] = v
        return orig(*ts)
    for ep in range(EPOCHS):
        mq.train(); pmax['m'] = 0.0; simulate.p_truncate = spy
        try:
            for x, tgt in train:
                opt.zero_grad(); loss = ce(mq(x), tgt)
                (S*loss).backward()
                with torch.no_grad():
                    for p in mq.parameters():
                        if p.grad is not None: p.grad /= S
                opt.step()
        finally:
            simulate.p_truncate = orig
        t1, t5 = evaluate(mq, torch.float64)
        q_loss.append(loss.item()); q_t1.append(t1); q_t5.append(t5)
        q_pt.append(math.log2(pmax['m']) if pmax['m'] > 0 else 0)
        log('CV S=2^%d e%-2d loss %.3f t1 %.2f (d%+.2f) t5 %.2f ptrunc 2^%.1f'
            % (log2s, ep+1, loss.item(), t1, t1-b_t1[ep], t5, q_pt[-1]))
    data['S%d' % log2s] = {'loss': q_loss, 't1': q_t1, 't5': q_t5, 'ptrunc': q_pt}

with open('/home/ryan/report_figs/sweep_curves_data.json', 'w') as f:
    json.dump(data, f, indent=1)
simulate.BITs = 22
log('CV DONE wrote sweep_curves_data.json')
