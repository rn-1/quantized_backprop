import torch, sys, math, time
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, augment_batch, cross_entropy_q

# DEADBAND TEST (BITS_REPORT §17). Weights are full-precision floats but are rounded to the 2^-BITs grid on
# every forward (to_fixed inside Conv2dQ / LinearQ / BatchNorm2dQ), so an update only changes the network
# when it moves a weight across a rounding boundary. Hypothesis: late in the cosine schedule, lr*grad shrinks
# below a coarse 2^-8 step, so the rounded weights freeze at the converged solution, while the fine 2^-13
# grid keeps letting small, noise-dominated updates through, so the model drifts.
#
# Per epoch (same config as §16.2: [5,5,5], aug, lr 0.1 cosine T_max=80, S=2^16, A=9) we log:
#   step_chg  -- fraction of rounded weights changed by a single optimizer step (avg over sampled steps)
#   upd_med   -- median |Δw| per step, in grid steps (< 0.5 => typical update cannot cross a boundary)
#   upd_p90   -- 90th percentile |Δw| per step, in grid steps
#   epoch_chg -- fraction of rounded weights different from the previous epoch's end
#   gap       -- top-1 vs the shared float baseline
# Prediction: in the decay phase step_chg -> ~0 at BITs=8, stays high at BITs=13.
# Usage: _deadband.py <BITS> <snapshot.pt> [EPOCHS]
BITS = int(sys.argv[1]); SNAP = sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 80
S = 2.0**16; A = 9
LR, MOM, WD, N_BATCH, T_MAX = 0.1, 0.9, 5e-4, 128, 80
SAMPLE_EVERY = 50                                   # sample the per-step stats every 50 batches (~8/epoch)
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
params = [p for p in mq.parameters()]
oq = torch.optim.SGD(params, lr=LR, momentum=MOM, weight_decay=WD)
sq = torch.optim.lr_scheduler.CosineAnnealingLR(oq, T_max=T_MAX)
scale = 2.0 ** BITS

def flat():   return torch.cat([p.detach().reshape(-1) for p in params])
def rounded(v): return torch.round(v * scale).to(torch.int64)    # what to_fixed feeds the network

log('DB BITs=%d S=2^16 A=9 [5,5,5] aug lr=%g cosine T_max=%d, %d ep, %.2fM params, grid step 2^-%d'
    % (BITS, LR, T_MAX, EPOCHS, sum(p.numel() for p in params)/1e6, BITS))
log('DB ep | lr | gap | step_chg | upd_med | upd_p90 | epoch_chg | sec')
Sf = float(S)
prev_epoch_q = rounded(flat())
for ep in range(1, EPOCHS+1):
    t0 = time.time(); mq.train()
    lr_now = oq.param_groups[0]['lr']
    chg_sum = med_sum = p90_sum = 0.0; ns = 0
    for bi,(x,tgt) in enumerate(train):
        sample = (bi % SAMPLE_EVERY == SAMPLE_EVERY-1)
        if sample: w_before = flat()
        oq.zero_grad(); out=mq(augment_batch(x)); loss=cross_entropy_q(out,tgt); (Sf*loss).backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None: p.grad /= Sf
        oq.step()
        if sample:
            with torch.no_grad():
                w_after = flat()
                chg_sum += (rounded(w_after) != rounded(w_before)).double().mean().item()
                du = (w_after - w_before).abs() * scale                 # update size in grid steps
                du = du[torch.randint(0, du.numel(), (2_000_000,), device=du.device)].float()
                med_sum += torch.quantile(du, 0.5).item(); p90_sum += torch.quantile(du, 0.9).item()
                ns += 1
            del w_before, w_after, du
    sq.step()
    with torch.no_grad():
        cur_q = rounded(flat())
        epoch_chg = (cur_q != prev_epoch_q).double().mean().item(); prev_epoch_q = cur_q
    q1 = evaluate(mq); b1 = base_acc[ep-1][0]
    log('DB e%-3d | %.4f | %+6.2f | %7.4f%% | %7.3f | %7.3f | %6.2f%% | %.0f'
        % (ep, lr_now, q1-b1, 100*chg_sum/ns, med_sum/ns, p90_sum/ns, 100*epoch_chg, time.time()-t0))
log('DB DONE')
