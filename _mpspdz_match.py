import torch, sys, math, time
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, cross_entropy_q

# BITS_REPORT §21.5 (check 3b): the simulator under the SAME recipe as Programs/Source/resnet_train.mpc,
# so the two independent fixed-point implementations can be compared batch-for-batch:
#   same data + order (load_data seed 0, fixed batches, no augmentation), same seed-0 init, constant lr,
#   momentum 0.9, no weight decay, weights stored at f with no master copy (FixedSGD F_W = F_V = f),
#   no softmax elevation (A=0). Loss-scale mapping: MP-SPDZ seed (p-y)*2^LS  ==  ours S = 2^(7+LS).
# Logs mean training loss per 50 batches and per epoch (MP-SPDZ prints per-batch loss), and test top-1.
# Usage: _mpspdz_match.py <blocks e.g. 222> <f> <LS bits> <epochs> <lr>
blocks = [int(c) for c in sys.argv[1]]; F = int(sys.argv[2]); LS = int(sys.argv[3])
EPOCHS = int(sys.argv[4]); LR = float(sys.argv[5])
S = 2.0 ** (7 + LS)
dev = torch.device('cuda')
simulate.STAGE_BLOCKS = blocks; simulate.GLOBAL_AVG_POOL = True
simulate.BITs = F; simulate.B_IV = F; simulate.SOFTMAX_EXTRA_BITS = 0
def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()
torch.manual_seed(0)
tl, te = load_data(128)
train = [(x.to(dev).to(torch.float64), y.to(dev).argmax(1)) for x, y in tl]
test = [(x.to(dev).to(torch.float64), y.argmax(1).to(dev)) for x, y in te]
torch.manual_seed(0)
mb = CNNModelBaseline().to(dev); mq = CNNModelQ().to(dev); sync_models(mb, mq); del mb
opt = simulate.FixedSGD(mq.parameters(), lr=LR, momentum=0.9, weight_decay=0, S=S, F_W=F, F_V=F)
def evaluate():
    c = n = 0
    with torch.no_grad():
        for x, t in test:
            c += (mq(x).argmax(1) == t).sum().item(); n += x.shape[0]
    return 100 * c / n
log('MM blocks=%s f=%d LS=%d (S=2^%d) A=0 lr=%g const, mom 0.9, wd 0, no aug, F_W=F_V=f' % (blocks, F, LS, 7 + LS, LR))
for ep in range(1, EPOCHS + 1):
    t0 = time.time(); mq.train(); run = []; tot = 0.0
    for bi, (x, t) in enumerate(train):
        opt.zero_grad(); loss = cross_entropy_q(mq(x), t); (S * loss).backward(); opt.step()
        lv = loss.item(); run.append(lv); tot += lv
        if (bi + 1) % 50 == 0:
            log('MM e%d b%d loss50 %.4f' % (ep, bi + 1, sum(run) / len(run))); run = []
    log('MM e%d | mean train loss %.4f | test t1 %.2f | %.0fs' % (ep, tot / len(train), evaluate(), time.time() - t0))
log('MM DONE')
