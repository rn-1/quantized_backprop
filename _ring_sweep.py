import torch, sys, math
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data,
                      calculateFinalParamError, cross_entropy_q)

# SMALLEST FIXED-POINT REPRESENTATION sweep. Hold the viable-floor config (BITs=10, S=2^14, A=7) and
# sweep the ring bit-width RING_BITS down. ring_truncate's failure rate ~ payload/2^RING_BITS, so as l
# shrinks the stochastic truncation injects more per-element noise until training breaks. The smallest l
# that still matches the baseline is the minimum backing-integer width this training tolerates. We also
# track the payload peak P each run: if P >= 2^(l-1) the failure is OVERFLOW (a different, harder wall),
# not truncation noise -- flagged so we don't misread the cliff.
dev = torch.device('cuda')
LR = 0.001
EPOCHS = 6
N_BATCH = 128
BITS, S, A = 10, 2**14, 7
RING_LS = [39, 38, 37, 36, 35, 34]     # push below P=2^35.6 to hit the overflow cliff

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x, y in te]
ce = torch.nn.CrossEntropyLoss()
log('RS cached %d train / %d test  BITs=%d S=2^%d A=%d  epochs=%d' %
    (len(train), len(test), BITS, int(math.log2(S)), A, EPOCHS))

def evaluate(model, dtype):
    model.train(); c1 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            logits = model(x.to(dtype))
            c1 += (logits.argmax(1) == tgt).sum().item(); n += x.shape[0]
    return 100 * c1 / n

Sf = float(S)
for L in RING_LS:
    simulate.BITs = BITS; simulate.B_IV = BITS; simulate.SOFTMAX_EXTRA_BITS = A
    simulate.RING_BITS = L
    torch.manual_seed(0)
    mb = CNNModelBaseline().to(dev); mq = CNNModelQ().to(dev); sync_models(mb, mq)
    ob = torch.optim.SGD(mb.parameters(), lr=LR, momentum=0.9)
    oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=0.9)

    peak = {'p': 0.0}
    orig = simulate.ring_truncate
    def spy(t, m, _o=orig, _p=peak):
        v = t.abs().max().item()
        if v > _p['p']: _p['p'] = v
        return _o(t, m)

    log('RS === RING_BITS=%d  (headroom over P: ~%.1f bits) ===' % (L, L - 35.6))
    log('RS ep | base t1 | quant t1 | gap | paramRMSE | payloadP 2^ | overflow? | nonfin')
    broke = False
    for ep in range(1, EPOCHS + 1):
        mb.train()
        for x, tgt in train:
            ob.zero_grad(); l_ = ce(mb(x.to(torch.float32)), tgt); l_.backward(); ob.step()
        mq.train(); peak['p'] = 0.0; simulate.ring_truncate = spy
        nonfin = 0
        try:
            for x, tgt in train:
                oq.zero_grad(); out = mq(x); loss = cross_entropy_q(out, tgt)
                (Sf * loss).backward()
                with torch.no_grad():
                    for p in mq.parameters():
                        if p.grad is not None: p.grad /= Sf
                oq.step()
                nonfin += int((~torch.isfinite(out)).sum().item())
        finally:
            simulate.ring_truncate = orig
        b1 = evaluate(mb, torch.float32); q1 = evaluate(mq, torch.float64)
        rmse = calculateFinalParamError(mb, mq)
        P = peak['p']; Pb = math.log2(P) if P > 0 else 0
        ovf = 'OVERFLOW' if Pb >= (L - 1) else '-'
        log('RS e%-2d | %5.2f | %5.2f | %+6.2f | %6.3f%% | 2^%5.2f | %-8s | %d'
            % (ep, b1, q1, q1 - b1, 100 * rmse, Pb, ovf, nonfin))
        if not math.isfinite(q1) or q1 < 1.5:      # collapsed to ~random (100 classes -> 1%)
            log('RS   -> COLLAPSED at RING_BITS=%d epoch %d' % (L, ep)); broke = True; break
    log('RS summary RING_BITS=%d : %s' % (L, 'BROKE' if broke else 'survived'))

simulate.BITs = 22; simulate.B_IV = 22; simulate.SOFTMAX_EXTRA_BITS = 0; simulate.RING_BITS = 57
log('RS DONE')
