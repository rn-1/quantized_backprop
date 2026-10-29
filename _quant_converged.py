import torch, sys, math, time
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, augment_batch, cross_entropy_q

# One quantized config on the converged [5,5,5] model: aug + lr=0.1 cosine 80 ep, matched strong scaling
# S=2^16 / A=9 (so BITs is the only variable across the sweep). Compares to the shared baseline snapshot.
# Usage: _quant_converged.py <BITS> <snapshot.pt> [log2 S (default 16)]
# A = log2 S - 7 (the §13 pairing; =9 at the default S=2^16).
BITS = int(sys.argv[1]); SNAP = sys.argv[2]
LOGS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
S = 2.0**LOGS; A = LOGS - 7
import os as _os
if 'QC_A' in _os.environ: A = int(_os.environ['QC_A'])   # decouple softmax elevation A from S (§19.7 control)
import os
# optional §19.4 knobs: QC_K = BN_EARLY_DIV_BITS, QC_RING = RING_BITS (real wraparound if payload >= 2^(l-1))
simulate.BN_EARLY_DIV_BITS = int(os.environ.get('QC_K', 0))
simulate.RING_BITS = int(os.environ.get('QC_RING', simulate.RING_BITS))
simulate.DIV_PUBLIC_RING = int(os.environ.get('QC_DIVRING', 0))
simulate.LIN_GRAD_PRESHIFT = int(os.environ.get('QC_PRESHIFT', 0))
simulate.BN_STATS_F = int(os.environ.get('QC_BNF', 0))   # §21.6 running-stat precision
SEED = int(os.environ.get('QC_SEED', 0))   # init + data order; must match the snapshot's float seed
# cheap payload monitor: running max |value| into ring_truncate, on GPU, read once per epoch
_peak = [None]; _orig_rt = simulate.ring_truncate
def _spy(t, m):
    v = t.abs().max()
    _peak[0] = v if _peak[0] is None else torch.maximum(_peak[0], v)
    return _orig_rt(t, m)
simulate.ring_truncate = _spy
# QC_SITE=<log2 thr> (§21.6 overflow hunt): per-site payload max, synced once per batch; logs every site whose
# payload in that batch exceeds 2^thr, with the simulate.py call chain (no RNG consumed, so a seeded replay
# follows the original run).
SITE_THR = os.environ.get('QC_SITE')
if SITE_THR is not None:
    SITE_THR = float(SITE_THR); _site = {}; _ep_site = {}
    def _spy(t, m):
        v = t.abs().max()
        _peak[0] = v if _peak[0] is None else torch.maximum(_peak[0], v)
        f = sys._getframe(1); chain = []
        while f is not None and len(chain) < 4:
            if f.f_code.co_filename.endswith('simulate.py') and f.f_code.co_name not in ('p_truncate',):
                chain.append('%s:%d' % (f.f_code.co_name, f.f_lineno))
            f = f.f_back
        k = (' < '.join(chain), m); _site[k] = v if k not in _site else torch.maximum(_site[k], v)
        return _orig_rt(t, m)
    simulate.ring_truncate = _spy
    def _flush(ep, bi):
        if not _site: return
        ks = list(_site); vs = torch.stack([_site[k].double() for k in ks]).tolist(); _site.clear()
        for k, v in zip(ks, vs):
            lv = math.log2(v) if v > 0 else 0
            if lv > _ep_site.get(k, 0): _ep_site[k] = lv
            if lv >= SITE_THR: log('SITE e%d b%d 2^%.3f m=%d %s%s' % (ep, bi, lv, k[1], k[0], ' OVF' if v >= 2**(simulate.RING_BITS-1) else ''))
    def _ep_report(ep):
        top = sorted(_ep_site.items(), key=lambda kv: -kv[1])[:8]
        for k, lv in top: log('SITETOP e%d 2^%.3f m=%d %s' % (ep, lv, k[1], k[0]))
        _ep_site.clear()
LR, MOM, WD, EPOCHS, N_BATCH = 0.1, 0.9, 5e-4, 80, 128
import os as _os2
EPOCHS = int(_os2.environ.get('EPOCHS', EPOCHS))   # cosine schedule length (default 80; Tiny-ImageNet runs use 40)
dev = torch.device('cuda')
simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True
# DATASET=tinyimagenet: 200 classes, 64x64. Its cache stays on the CPU (float64 on-GPU would be ~9.8 GB per
# run) and each batch is moved over in the loop; for CIFAR the .to(dev) calls below are no-ops.
DATASET = os.environ.get('DATASET', 'cifar100'); simulate.set_dataset(DATASET)
cache = torch.device('cpu') if DATASET == 'tinyimagenet' else dev
# §23 ablation knobs: BLOCKS (e.g. 555, 999) and RESIDUAL (1 = skip connections, 0 = plain network)
BLOCKS = [int(c) for c in os.environ.get('BLOCKS', '555')]; simulate.STAGE_BLOCKS = BLOCKS
simulate.RESIDUAL = bool(int(os.environ.get('RESIDUAL', 1)))
simulate.BITs = BITS; simulate.B_IV = BITS; simulate.SOFTMAX_EXTRA_BITS = A

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()
snap = torch.load(SNAP); base_acc = snap['acc']; base_final = snap['final_params']
assert snap.get('seed', 0) == SEED, 'snapshot seed %s != QC_SEED %d' % (snap.get('seed', 0), SEED)
for k, mine, default in (('dataset', DATASET, 'cifar100'), ('blocks', BLOCKS, [5,5,5]), ('residual', simulate.RESIDUAL, True)):
    assert snap.get(k, default) == mine, 'snapshot %s %s != this run %s' % (k, snap.get(k, default), mine)

torch.manual_seed(SEED)
tl, te = load_data(N_BATCH)
train = [(x.to(cache).to(torch.float64 if cache == dev else torch.float32), y.to(dev).argmax(1)) for x,y in tl]
test  = [(x.to(cache), y.argmax(1).to(dev)) for x,y in te]

def evaluate(m, batch_stats=False):
    # §21.6: eval mode = BN normalises with the running stats (same protocol as the float baseline).
    # batch_stats=True reproduces the old (buggy) protocol -- each test batch normalised by its own
    # statistics -- for comparison only; running stats are restored so it cannot leak into the model.
    bufs = [b.clone() for b in m.buffers()]
    m.train() if batch_stats else m.eval(); c1=c5=n=0
    with torch.no_grad():
        for x,tgt in test:
            lo=m(x.to(dev).to(torch.float64)); c1+=(lo.argmax(1)==tgt).sum().item()
            c5+=(lo.topk(5,1).indices==tgt.unsqueeze(1)).any(1).sum().item(); n+=x.shape[0]
    for b, s in zip(m.buffers(), bufs): b.copy_(s)
    return 100*c1/n,100*c5/n

torch.manual_seed(SEED)
mb_init = CNNModelBaseline().to(dev)               # reproduce seed-0 init to sync quant from
mq = CNNModelQ().to(dev); sync_models(mb_init, mq); del mb_init
# QC_FXOPT=1: fixed-point SGD (§20) -- master weights @2^QC_FW, momentum @2^QC_FV, all ring-resident;
# the gradient stays S-lifted (no float unscale). Otherwise stock float64 torch SGD on a float master.
FXOPT = int(os.environ.get('QC_FXOPT', 0))
if FXOPT:
    FW, FV = int(os.environ.get('QC_FW', 20)), int(os.environ.get('QC_FV', 20))
    simulate.BITs = BITS                              # FixedSGD publishes the compute copy at 2^-BITs
    oq = simulate.FixedSGD(mq.parameters(), lr=LR, momentum=MOM, weight_decay=WD, S=S, F_W=FW, F_V=FV)
    lr_at = lambda ep: 0.5 * LR * (1 + math.cos(math.pi * (ep - 1) / EPOCHS))   # = CosineAnnealingLR
else:
    oq = torch.optim.SGD(mq.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
    sq = torch.optim.lr_scheduler.CosineAnnealingLR(oq, T_max=EPOCHS)
log('QC SEED=%d RECIP_NEWTON=%d ISQRT_NEWTON=%d BN_EARLY_DIV_BITS=%d RING_BITS=%d DIV_PUBLIC_RING=%d LIN_GRAD_PRESHIFT=%d' % (SEED, simulate.RECIP_NEWTON, simulate.ISQRT_NEWTON, simulate.BN_EARLY_DIV_BITS, simulate.RING_BITS, simulate.DIV_PUBLIC_RING, simulate.LIN_GRAD_PRESHIFT))
log('QC FXOPT=%d%s BN_STATS_F=%d' % (FXOPT, (' F_W=%d F_V=%d' % (FW, FV)) if FXOPT else '', simulate.BN_STATS_F))
log('QC BITs=%d S=2^%d A=%d %s%s %s aug lr=%g cosine %d ep  (%.2fM params)'
    % (BITS, int(math.log2(S)), A, BLOCKS, '' if simulate.RESIDUAL else ' PLAIN(no skips)', DATASET, LR, EPOCHS, sum(p.numel() for p in mq.parameters())/1e6))
log('QC ep | base t1/t5 | quant t1/t5 | t1 gap | nonfin | sec | payload | batch-stat t1 (old protocol)')
Sf = float(S)
for ep in range(1, EPOCHS+1):
    t0=time.time(); mq.train(); nonfin=0
    if FXOPT: oq.lr = lr_at(ep)
    for bi,(x,tgt) in enumerate(train):
        oq.zero_grad(); out=mq(augment_batch(x.to(dev).to(torch.float64))); loss=cross_entropy_q(out,tgt); (Sf*loss).backward()
        if not FXOPT:                                  # FixedSGD consumes the S-lifted grad directly
            with torch.no_grad():
                for p in mq.parameters():
                    if p.grad is not None: p.grad/=Sf
        oq.step(); nonfin += int((~torch.isfinite(out)).sum().item())
        if SITE_THR is not None: _flush(ep, bi)
    if not FXOPT: sq.step()
    q1,q5 = evaluate(mq); qb1,_ = evaluate(mq, batch_stats=True); b1,b5 = base_acc[ep-1]
    P = _peak[0].item(); _peak[0] = None
    log('QC e%-3d | %5.2f/%5.2f | %5.2f/%5.2f | %+5.2f | %d | %.0f | 2^%.2f%s | %5.2f'
        % (ep, b1,b5, q1,q5, q1-b1, nonfin, time.time()-t0, math.log2(P) if P > 0 else 0,
           ' OVF' if P >= 2**(simulate.RING_BITS-1) else '', qb1))
    if SITE_THR is not None: _ep_report(ep)
    if os.environ.get('QC_BNDIAG'):   # §22: BN running-stat ranges + weight magnitudes (no RNG consumed)
        with torch.no_grad():
            bns = [(n, b) for n, b in mq.named_modules() if isinstance(b, simulate.QBatchNorm2d)]
            rv = torch.cat([b.running_var for _, b in bns]); rm = torch.cat([b.running_mean for _, b in bns])
            worst = max(bns, key=lambda nb: nb[1].running_var.max().item())
            wmax = max(p.detach().abs().max().item() for p in mq.parameters())
            gmax = max(b.weight.detach().abs().max().item() for _, b in bns)
            log('BNDIAG e%d | running_var min %.3g max %.3g (%s) | var<=0: %d/%d | |running_mean| max %.3g | |param| max %.3g | |BN gamma| max %.3g'
                % (ep, rv.min().item(), rv.max().item(), worst[0], (rv <= 0).sum().item(), rv.numel(), rm.abs().max().item(), wmax, gmax))
    if os.environ.get('QC_SAVE_EVERY') and os.environ.get('QC_SAVE'):
        torch.save({'state': mq.state_dict(), 'seed': SEED, 'epoch': ep}, os.environ['QC_SAVE'] + '.e%d' % ep)
    if ep >= int(os.environ.get('QC_EPOCHS', EPOCHS)): log('QC STOP after e%d (QC_EPOCHS)' % ep); break
# final param RMSE vs baseline
tot=totref=0.0
for n,p in mq.named_parameters():
    b=base_final[n].double(); q=p.detach().float().cpu().double()
    tot+=torch.sum((b-q)**2).item(); totref+=torch.sum(b**2).item()
rmse = math.sqrt(tot/totref) if totref>0 else float('nan')
bb = max(a[0] for a in base_acc)
log('QC summary BITs=%d: final quant t1 %.2f (base best %.2f) paramRMSE %.1f%%' % (BITS, q1, bb, 100*rmse))
if os.environ.get('QC_SAVE'): torch.save({'state': mq.state_dict(), 'seed': SEED}, os.environ['QC_SAVE'])
log('QC DONE')
