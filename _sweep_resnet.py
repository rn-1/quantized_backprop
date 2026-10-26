import torch, sys, math, time
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data, cross_entropy_q)

# Converged-model re-sweep of the BITs floor on the NEW residual architecture (GAP head), with a proper
# training recipe: SGD lr=0.05 + momentum 0.9 + weight decay 5e-4, cosine-annealed over 25 epochs.
# Answers two things: (1) the new net's real accuracy ceiling (does it beat the old ~40%?), and
# (2) whether BITs=13/10/8 stay faithful when the model is actually trained hard (the sharper-model
# regime that stresses the weight grid). Baseline is trained ONCE (deterministic) and reused.
dev = torch.device('cuda')
LR = 0.05
MOM = 0.9
WD = 5e-4
EPOCHS = 25
N_BATCH = 128
simulate.GLOBAL_AVG_POOL = True
CONFIGS = [(13, 2**12, 5), (10, 2**14, 7), (8, 2**16, 9)]

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
ce = torch.nn.CrossEntropyLoss()
log('RS cached %d train / %d test  recipe: lr=%g mom=%g wd=%g cosine  epochs=%d GAP=%s'
    % (len(train), len(test), LR, MOM, WD, EPOCHS, simulate.GLOBAL_AVG_POOL))

def evaluate(model, dtype):
    model.train(); c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            lo = model(x.to(dtype))
            c1 += (lo.argmax(1) == tgt).sum().item()
            c5 += (lo.topk(5,1).indices == tgt.unsqueeze(1)).any(1).sum().item(); n += x.shape[0]
    return 100*c1/n, 100*c5/n

# ---- Phase 1: train the float baseline once; snapshot per-epoch accuracy + params (CPU) ----
simulate.BITs = 22; simulate.B_IV = 22            # baseline is float; BITs irrelevant to it
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev)
ob = torch.optim.SGD(mb.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
sb = torch.optim.lr_scheduler.CosineAnnealingLR(ob, T_max=EPOCHS)
base_acc = []; base_snap = []
log('RS --- Phase 1: baseline (%.2fM params) ---' % (sum(p.numel() for p in mb.parameters())/1e6))
log('RS ep | base t1/t5 | sec')
for ep in range(1, EPOCHS+1):
    t0 = time.time(); mb.train()
    for x, tgt in train:
        ob.zero_grad(); l = ce(mb(x.to(torch.float32)), tgt); l.backward(); ob.step()
    sb.step()
    b1, b5 = evaluate(mb, torch.float32)
    base_acc.append((b1, b5))
    base_snap.append({n: p.detach().float().cpu().clone() for n, p in mb.named_parameters()})
    log('RS e%-2d | %5.2f/%5.2f | %.0f' % (ep, b1, b5, time.time()-t0))
log('RS baseline BEST t1 = %.2f (e%d)' % (max(a[0] for a in base_acc),
                                          1+max(range(EPOCHS), key=lambda i: base_acc[i][0])))

def param_rmse(snap, model):
    tot = totref = 0.0
    for n, p in model.named_parameters():
        b = snap[n].double(); q = p.detach().float().cpu().double()
        tot += torch.sum((b-q)**2).item(); totref += torch.sum(b**2).item()
    return math.sqrt(tot/totref) if totref > 0 else float('nan')

# ---- Phase 2: per config, train quant from the same init; compare to the stored baseline curve ----
for BITS, S, A in CONFIGS:
    Sf = float(S)
    simulate.BITs = BITS; simulate.B_IV = BITS; simulate.SOFTMAX_EXTRA_BITS = A
    torch.manual_seed(0)
    mb_init = CNNModelBaseline().to(dev)          # reproduce identical init to sync from
    mq = CNNModelQ().to(dev); sync_models(mb_init, mq); del mb_init
    oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    sq = torch.optim.lr_scheduler.CosineAnnealingLR(oq, T_max=EPOCHS)
    log('RS === BITs=%d S=2^%d A=%d ===' % (BITS, int(math.log2(S)), A))
    log('RS ep | base t1/t5 | quant t1/t5 | t1 gap | paramRMSE | nonfin | sec')
    gaps = []
    for ep in range(1, EPOCHS+1):
        t0 = time.time(); mq.train(); nonfin = 0
        for x, tgt in train:
            oq.zero_grad(); out = mq(x); loss = cross_entropy_q(out, tgt); (Sf*loss).backward()
            with torch.no_grad():
                for p in mq.parameters():
                    if p.grad is not None: p.grad /= Sf
            oq.step(); nonfin += int((~torch.isfinite(out)).sum().item())
        sq.step()
        q1, q5 = evaluate(mq, torch.float64)
        b1, b5 = base_acc[ep-1]; rmse = param_rmse(base_snap[ep-1], mq); gaps.append(q1-b1)
        log('RS e%-2d | %5.2f/%5.2f | %5.2f/%5.2f | %+5.2f | %6.3f%% | %d | %.0f'
            % (ep, b1, b5, q1, q5, q1-b1, 100*rmse, nonfin, time.time()-t0))
    log('RS summary BITs=%d: t1 gap mean %+.2f worst %+.2f  final quant t1 %.2f' %
        (BITS, sum(gaps)/len(gaps), min(gaps), q1))

simulate.BITs=22; simulate.B_IV=22; simulate.SOFTMAX_EXTRA_BITS=0
log('RS DONE')
