import torch, sys, math
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data,
                      calculateFinalParamError, cross_entropy_q)

# Multi-epoch fidelity WITH forward softmax scaling (SOFTMAX_EXTRA_BITS = A) added on top of the
# backward loss scaling S. A carries the softmax probs at 2^(BITs+A) so small off-target probs survive
# the forward instead of underflowing at 2^-BITs -- the genuine forward floor that capped BITs=8.
# A = log2(S) - log2(N), N=128=2^7, lines the forward floor up with the backward floor so they descend
# together. Re-checks the old configs (should tighten, esp. BITs=8) and pushes the frontier to 7 and 6.
dev = torch.device('cuda')
LR = 0.001
EPOCHS = 12
N_BATCH = 128
LOG2N = 7                                    # log2(128)
CONFIGS = [(13, 2**12), (10, 2**14), (8, 2**16), (7, 2**17), (6, 2**18)]

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
log('MC cached %d train / %d test batches  epochs=%d lr=%g (FORWARD SCALING ON)' % (len(train), len(test), EPOCHS, LR))
ce = torch.nn.CrossEntropyLoss()

def evaluate(model, dtype):
    model.train()
    c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            logits = model(x.to(dtype))
            c1 += (logits.argmax(1) == tgt).sum().item()
            c5 += (logits.topk(5, 1).indices == tgt.unsqueeze(1)).any(1).sum().item()
            n += x.shape[0]
    return 100*c1/n, 100*c5/n

def wmax(m): return max(p.abs().max().item() for p in m.parameters())

for BITS, S in CONFIGS:
    S = float(S)
    A = max(0, int(round(math.log2(S))) - LOG2N)   # forward scale = log2(S) - log2(N)
    simulate.BITs = BITS
    simulate.B_IV = BITS                     # Method A OFF
    simulate.SOFTMAX_EXTRA_BITS = A
    torch.manual_seed(0)
    mb = CNNModelBaseline().to(dev)
    mq = CNNModelQ().to(dev)
    sync_models(mb, mq)
    ob = torch.optim.SGD(mb.parameters(), lr=LR, momentum=0.9)
    oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=0.9)

    pmon = {'max': 0.0}
    orig = simulate.ring_truncate
    def spy(t, m):
        v = t.abs().max().item()
        if v > pmon['max']: pmon['max'] = v
        return orig(t, m)

    log('MC === BITs=%d S=2^%d A=%d (B_IV=%d, Method A OFF) ===' % (BITS, int(math.log2(S)), A, simulate.B_IV))
    log('MC epoch | base loss | quant loss | base t1/t5 | quant t1/t5 | t1 gap | paramRMSE | wmax b/q | ring 2^ | nonfin')
    gaps = []
    for ep in range(1, EPOCHS + 1):
        mb.train()
        for x, tgt in train:
            ob.zero_grad(); l = ce(mb(x.to(torch.float32)), tgt); l.backward(); ob.step()
        mq.train(); pmon['max'] = 0.0; simulate.ring_truncate = spy
        nonfin = 0; qlast = 0.0
        try:
            for x, tgt in train:
                oq.zero_grad(); out = mq(x); loss = cross_entropy_q(out, tgt)  # fixed-point softmax/CE
                (S*loss).backward()
                with torch.no_grad():
                    for p in mq.parameters():
                        if p.grad is not None: p.grad /= S
                oq.step(); qlast = loss.item()
                nonfin += int((~torch.isfinite(out)).sum().item())
        finally:
            simulate.ring_truncate = orig
        b1, b5 = evaluate(mb, torch.float32)
        q1, q5 = evaluate(mq, torch.float64)
        rmse = calculateFinalParamError(mb, mq)
        pm = pmon['max']; gaps.append(q1 - b1)
        log('MC e%-2d | %.3f | %.3f | %5.2f/%5.2f | %5.2f/%5.2f | %+5.2f | %6.3f%% | %.2g/%.2g | 2^%.1f | %d'
            % (ep, l.item(), qlast, b1, b5, q1, q5, (q1 - b1), 100*rmse,
               wmax(mb), wmax(mq), (math.log2(pm) if pm > 0 else 0), nonfin))
    log('MC summary BITs=%d S=2^%d A=%d: t1 gap mean %+.2f worst %+.2f  (all epochs)' %
        (BITS, int(math.log2(S)), A, sum(gaps)/len(gaps), min(gaps)))

simulate.BITs = 22; simulate.B_IV = 22; simulate.SOFTMAX_EXTRA_BITS = 0
log('MC DONE')
