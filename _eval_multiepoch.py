import torch, sys, math
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data,
                      paired_tensors, calculateFinalParamError)

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR = 0.001
EPOCHS = 12

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

log('ME device %s  BITs=%d  epochs=%d  lr=%g' % (dev, simulate.BITs, EPOCHS, LR))

# cache one epoch of batches, identical order every epoch and identical for both models
torch.manual_seed(0)
train_loader, test_loader = load_data(128)
train_batches = [(x.clone(), y.clone()) for x, y in train_loader]
test_batches  = [(x.clone(), y.clone()) for x, y in test_loader]
log('ME cached %d train / %d test batches' % (len(train_batches), len(test_batches)))

# build + sync so both start from identical weights
torch.manual_seed(0)
model_base = CNNModelBaseline().to(dev)
model_Q = CNNModelQ().to(dev)
sync_models(model_base, model_Q)

ce = torch.nn.CrossEntropyLoss()
# one optimizer per model, created ONCE so SGD momentum persists across epochs
opt_base = torch.optim.SGD(model_base.parameters(), lr=LR, momentum=0.9)
opt_Q    = torch.optim.SGD(model_Q.parameters(),    lr=LR, momentum=0.9)

# monitor the magnitude actually fed to the ring during the quantized epoch
orig_ptrunc = simulate.p_truncate
pmon = {'max': 0.0}
def spy(*ts):
    for t in ts:
        m = t.abs().max().item()
        if m > pmon['max']: pmon['max'] = m
    return orig_ptrunc(*ts)

def train_epoch(model, opt, dtype):
    model.train()
    losses = []
    for x, y in train_batches:
        x = x.to(dev).to(dtype); target = y.to(dev).argmax(1)
        opt.zero_grad(); out = model(x); loss = ce(out, target)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

def top1_top5(model, dtype):
    # train-mode BN (quantized has no eval path); no grad
    model.train()
    c1 = c5 = n = 0
    with torch.no_grad():
        for x, y in test_batches:
            x = x.to(dev).to(dtype)
            logits = model(x)
            target = y.argmax(1).to(dev)
            c1 += (logits.argmax(1) == target).sum().item()
            c5 += (logits.topk(5, 1).indices == target.unsqueeze(1)).any(1).sum().item()
            n += x.shape[0]
    return 100 * c1 / n, 100 * c5 / n, n

def wmax(model):
    return max(p.abs().max().item() for p in model.parameters())

log('ME epoch | base loss last/min | quant loss last/min | base t1/t5 | quant t1/t5 | t1_gap | paramRMSE | wmax b/q | ptrunc(%2^53) | nonfin')
for ep in range(1, EPOCHS + 1):
    bl = train_epoch(model_base, opt_base, torch.float32)
    pmon['max'] = 0.0
    simulate.p_truncate = spy
    try:
        ql = train_epoch(model_Q, opt_Q, torch.float64)
    finally:
        simulate.p_truncate = orig_ptrunc

    nonfin = sum(1 for v in ql if not math.isfinite(v))
    b1, b5, n = top1_top5(model_base, torch.float32)
    q1, q5, _ = top1_top5(model_Q, torch.float64)
    rmse = calculateFinalParamError(model_base, model_Q)
    pm = pmon['max']
    log('ME e%-2d | %.3f/%.3f | %.3f/%.3f | %5.2f/%5.2f | %5.2f/%5.2f | %+5.2f | %6.3f%% | %.2g/%.2g | %.2f%% | %d'
        % (ep, bl[-1], min(bl), ql[-1], min(ql),
           b1, b5, q1, q5, (q1 - b1), 100 * rmse,
           wmax(model_base), wmax(model_Q), 100 * pm / 2**53, nonfin))
log('ME DONE')
