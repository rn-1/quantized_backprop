import torch, sys, math, time
import simulate
from simulate import CNNModelBaseline, load_data, augment_batch

# Strengthen the float baseline: proper recipe (SGD lr=0.1 + mom 0.9 + wd 5e-4, cosine over 80 epochs)
# WITH on-GPU augmentation (per-sample flip + reflect-pad random crop), swept over conv depth. Goal:
# find how good a converged CIFAR-100 model this architecture reaches, and how depth affects it -- the
# reference we'll later quantize (no point pursuing quantization if the float model can't predict well).
dev = torch.device('cuda')
LR = 0.1
MOM = 0.9
WD = 5e-4
EPOCHS = 80
N_BATCH = 128
simulate.GLOBAL_AVG_POOL = True
DEPTHS = [[2, 2, 2], [3, 3, 3], [5, 5, 5]]

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()

torch.manual_seed(0)
tl, te = load_data(N_BATCH)
train = [(x.to(dev).to(torch.float32), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(dev).to(torch.float32), y.argmax(1).to(dev)) for x, y in te]
ce = torch.nn.CrossEntropyLoss()
log('SB cached %d train / %d test  recipe: lr=%g mom=%g wd=%g cosine aug=on epochs=%d GAP=%s'
    % (len(train), len(test), LR, MOM, WD, EPOCHS, simulate.GLOBAL_AVG_POOL))

def evaluate(model):
    model.eval(); c1 = c5 = n = 0
    with torch.no_grad():
        for x, tgt in test:
            lo = model(x)
            c1 += (lo.argmax(1) == tgt).sum().item()
            c5 += (lo.topk(5,1).indices == tgt.unsqueeze(1)).any(1).sum().item(); n += x.shape[0]
    return 100*c1/n, 100*c5/n

for blocks in DEPTHS:
    simulate.STAGE_BLOCKS = blocks
    torch.manual_seed(0)
    m = CNNModelBaseline().to(dev)
    npar = sum(p.numel() for p in m.parameters())/1e6
    opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    log('SB === depth=%s  (%.2fM params) ===' % (blocks, npar))
    best1 = best5 = 0.0; bep = 0
    for ep in range(1, EPOCHS+1):
        t0 = time.time(); m.train()
        for x, tgt in train:
            opt.zero_grad(); l = ce(m(augment_batch(x)), tgt); l.backward(); opt.step()
        sch.step()
        if ep % 5 == 0 or ep == 1 or ep >= EPOCHS-3:
            t1, t5 = evaluate(m)
            if t1 > best1: best1, best5, bep = t1, t5, ep
            log('SB e%-3d | t1 %5.2f / t5 %5.2f | %.0fs' % (ep, t1, t5, time.time()-t0))
    log('SB depth=%s BEST: t1 %.2f / t5 %.2f (e%d)' % (blocks, best1, best5, bep))

log('SB DONE')
