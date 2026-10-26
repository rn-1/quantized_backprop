import torch, sys, math, time
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, augment_batch, cross_entropy_q

# Test the hypothesis: on the deep [5,5,5] residual net, skip connections keep gradients from vanishing,
# so the S-lifted gradient (grad*S=2^16) through the residual accumulation grows large -- and at higher
# BITs (base scale 2^BITs) the ring payload approaches the 2^(RING_BITS-1) ceiling, causing silent
# wraparound late in training (worse at higher bits). Cheap instrumentation (GPU running-max, synced once
# per epoch; grad stats only on the last batch) tracks per epoch:
#   payload 2^ (max |value| into ring_truncate, fwd+bwd) | ovf flag | S-lifted grad max | true grad max
#   | stem vs head grad-norm (vanishing check) | accuracy gap.
BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 13
SNAP = sys.argv[2]
S = 2.0**16; A = 9
LR, MOM, WD, EPOCHS, N_BATCH = 0.1, 0.9, 5e-4, 65, 128
dev = torch.device('cuda')
simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True
simulate.BITs = BITS; simulate.B_IV = BITS; simulate.SOFTMAX_EXTRA_BITS = A

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()
snap = torch.load(SNAP); base_acc = snap['acc']
torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x,y in tl]
test  = [(x.to(dev), y.argmax(1).to(dev)) for x,y in te]

def evaluate(m):
    m.eval(); c1=n=0
    with torch.no_grad():
        for x,tgt in test:
            lo=m(x.to(torch.float64)); c1+=(lo.argmax(1)==tgt).sum().item(); n+=x.shape[0]
    return 100*c1/n

torch.manual_seed(0)
mb_init = CNNModelBaseline().to(dev)
mq = CNNModelQ().to(dev); sync_models(mb_init, mq); del mb_init
oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
sq = torch.optim.lr_scheduler.CosineAnnealingLR(oq, T_max=80)

# cheap payload spy: keep a running max on-GPU, read once per epoch (no per-call sync)
peakt = torch.zeros((), device=dev)
orig = simulate.ring_truncate
def spy(t, m):
    global peakt
    peakt = torch.maximum(peakt, t.abs().max().detach())
    return orig(t, m)

log('IG BITs=%d S=2^16 A=9 [5,5,5] RING_BITS=%d (ceiling 2^%d)' % (BITS, simulate.RING_BITS, simulate.RING_BITS-1))
log('IG ep | gap | payload 2^ | ovf | Sgrad_max 2^ | truegrad_max | stem_gn | head_gn | sec')
Sf = float(S); nb = len(train)
for ep in range(1, EPOCHS+1):
    t0=time.time(); mq.train(); peakt.zero_(); simulate.ring_truncate = spy
    sgrad_max = grad_max = stem_gn = head_gn = 0.0
    try:
        for bi,(x,tgt) in enumerate(train):
            oq.zero_grad(); out=mq(augment_batch(x)); loss=cross_entropy_q(out,tgt); (Sf*loss).backward()
            if bi == nb-1:   # sample grad stats once per epoch (last batch)
                with torch.no_grad():
                    for n_,p in mq.named_parameters():
                        if p.grad is None: continue
                        sgrad_max = max(sgrad_max, p.grad.abs().max().item())
                        if n_=='stem_conv.weight': stem_gn = p.grad.norm().item()/Sf
                        if n_=='head_fc_out.weight': head_gn = p.grad.norm().item()/Sf
            with torch.no_grad():
                for p in mq.parameters():
                    if p.grad is not None: p.grad /= Sf
            if bi == nb-1:
                with torch.no_grad():
                    grad_max = max(p.grad.abs().max().item() for p in mq.parameters() if p.grad is not None)
            oq.step()
    finally:
        simulate.ring_truncate = orig
    sq.step()
    q1 = evaluate(mq); b1 = base_acc[ep-1][0]
    P = peakt.item(); Pb = math.log2(P) if P>0 else 0
    ovf = 'OVF' if Pb >= simulate.RING_BITS-1 else '-'
    log('IG e%-2d | %+5.2f | 2^%5.2f | %-3s | 2^%5.2f | %.2e | %.2e | %.2e | %.0f'
        % (ep, q1-b1, Pb, ovf, math.log2(sgrad_max) if sgrad_max>0 else 0,
           grad_max, stem_gn, head_gn, time.time()-t0))
log('IG DONE')
