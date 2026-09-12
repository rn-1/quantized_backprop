import torch, sys, math
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data,
                      calculateFinalParamError)

dev = torch.device('cuda')
LR = 0.001
EPOCHS = 12
BITS = 13
S = float(2 ** 12)

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

log('ML device %s  BITs=%d  S=2^%d  epochs=%d  lr=%g' % (dev, BITS, int(math.log2(S)), EPOCHS, LR))

torch.manual_seed(0)
tl, te = load_data(128)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
log('ML cached %d train / %d test batches' % (len(train), len(test)))

simulate.BITs = BITS
torch.manual_seed(0)
model_base = CNNModelBaseline().to(dev)
model_Q = CNNModelQ().to(dev)
sync_models(model_base, model_Q)

ce = torch.nn.CrossEntropyLoss()
opt_base = torch.optim.SGD(model_base.parameters(), lr=LR, momentum=0.9)
opt_Q    = torch.optim.SGD(model_Q.parameters(),    lr=LR, momentum=0.9)

orig_ptrunc = simulate.p_truncate
pmon = {'max': 0.0}
def spy(*ts):
    for t in ts:
        m = t.abs().max().item()
        if m > pmon['max']: pmon['max'] = m
    return orig_ptrunc(*ts)

def train_epoch_base():
    model_base.train()
    losses = []
    for x, tgt in train:
        opt_base.zero_grad(); loss = ce(model_base(x.to(torch.float32)), tgt)
        loss.backward(); opt_base.step(); losses.append(loss.item())
    return losses

def train_epoch_Q():
    model_Q.train()
    losses = []
    for x, tgt in train:
        opt_Q.zero_grad()
        loss = ce(model_Q(x), tgt)
        (S * loss).backward()                       # scale gradients through the truncations
        with torch.no_grad():
            for p in model_Q.parameters():
                if p.grad is not None: p.grad /= S  # exact unscale before the step
        opt_Q.step(); losses.append(loss.item())
    return losses

def evaluate(model, dtype):
    model.train()  # quantized BN has no eval path -> both in train-mode BN
    c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            logits = model(x.to(dtype))
            c1 += (logits.argmax(1) == tgt).sum().item()
            c5 += (logits.topk(5, 1).indices == tgt.unsqueeze(1)).any(1).sum().item()
            n += x.shape[0]
    return 100*c1/n, 100*c5/n

def wmax(m): return max(p.abs().max().item() for p in m.parameters())

log('ML epoch | base loss | quant loss | base t1/t5 | quant t1/t5 | t1 gap | paramRMSE | wmax b/q | ptrunc(2^) %2^53 | nonfin')
for ep in range(1, EPOCHS + 1):
    bl = train_epoch_base()
    pmon['max'] = 0.0
    simulate.p_truncate = spy
    try:
        ql = train_epoch_Q()
    finally:
        simulate.p_truncate = orig_ptrunc
    nonfin = sum(1 for v in ql if not math.isfinite(v))
    b1, b5 = evaluate(model_base, torch.float32)
    q1, q5 = evaluate(model_Q, torch.float64)
    rmse = calculateFinalParamError(model_base, model_Q)
    pm = pmon['max']
    log('ML e%-2d | %.3f | %.3f | %5.2f/%5.2f | %5.2f/%5.2f | %+5.2f | %6.3f%% | %.2g/%.2g | 2^%.1f %.2f%% | %d'
        % (ep, bl[-1], ql[-1], b1, b5, q1, q5, (q1 - b1), 100 * rmse,
           wmax(model_base), wmax(model_Q),
           (math.log2(pm) if pm > 0 else 0), 100 * pm / 2**53, nonfin))
simulate.BITs = 22
log('ML DONE')
