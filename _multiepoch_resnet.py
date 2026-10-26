import torch, sys, math, time
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data,
                      calculateFinalParamError, cross_entropy_q)

# Validation of the new residual architecture: baseline vs fixed-point fidelity at the BITs=10 floor,
# GAP head on. Confirms the deeper net (a) trains, (b) the quantized model tracks the float baseline,
# (c) reaches better accuracy than the old ~40% ceiling. Times per-epoch cost (the net is deeper now).
dev = torch.device('cuda')
LR = 0.001
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N_BATCH = 128
BITS, S, A = 10, 2**14, 7
simulate.GLOBAL_AVG_POOL = True

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
ce = torch.nn.CrossEntropyLoss()
log('RN cached %d train / %d test  BITs=%d S=2^%d A=%d GAP=%s epochs=%d' %
    (len(train), len(test), BITS, int(math.log2(S)), A, simulate.GLOBAL_AVG_POOL, EPOCHS))

def evaluate(model, dtype):
    model.train(); c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            lo = model(x.to(dtype))
            c1 += (lo.argmax(1) == tgt).sum().item()
            c5 += (lo.topk(5,1).indices == tgt.unsqueeze(1)).any(1).sum().item(); n += x.shape[0]
    return 100*c1/n, 100*c5/n

simulate.BITs = BITS; simulate.B_IV = BITS; simulate.SOFTMAX_EXTRA_BITS = A
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev); mq = CNNModelQ().to(dev); sync_models(mb, mq)
nq = sum(p.numel() for p in mq.parameters())
log('RN model params = %.2fM  fc_in=%d' % (nq/1e6, mq.fc_in_dim))
ob = torch.optim.SGD(mb.parameters(), lr=LR, momentum=0.9)
oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=0.9)
Sf = float(S)
log('RN ep | base t1/t5 | quant t1/t5 | t1 gap | paramRMSE | nonfin | sec')
for ep in range(1, EPOCHS+1):
    t0 = time.time()
    mb.train()
    for x, tgt in train:
        ob.zero_grad(); l = ce(mb(x.to(torch.float32)), tgt); l.backward(); ob.step()
    mq.train(); nonfin = 0
    for x, tgt in train:
        oq.zero_grad(); out = mq(x); loss = cross_entropy_q(out, tgt); (Sf*loss).backward()
        with torch.no_grad():
            for p in mq.parameters():
                if p.grad is not None: p.grad /= Sf
        oq.step(); nonfin += int((~torch.isfinite(out)).sum().item())
    b1,b5 = evaluate(mb, torch.float32); q1,q5 = evaluate(mq, torch.float64)
    rmse = calculateFinalParamError(mb, mq)
    log('RN e%-2d | %5.2f/%5.2f | %5.2f/%5.2f | %+5.2f | %6.3f%% | %d | %.0f'
        % (ep, b1,b5, q1,q5, q1-b1, 100*rmse, nonfin, time.time()-t0))
simulate.BITs=22; simulate.B_IV=22; simulate.SOFTMAX_EXTRA_BITS=0
log('RN DONE')
