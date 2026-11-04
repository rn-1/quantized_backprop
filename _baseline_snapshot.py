import torch, sys, math, time
import simulate
from simulate import CNNModelBaseline, load_data, augment_batch

# Train the [5,5,5] float baseline once with the strong recipe (aug, lr=0.1 cosine, 80 ep) and save its
# per-epoch accuracy + final params, so the parallel quant jobs share ONE identical reference instead of
# retraining it three times. Init is seed-0 so the quant jobs (also seed-0) sync from the same start.
dev = torch.device('cuda')
LR, MOM, WD, EPOCHS, N_BATCH = 0.1, 0.9, 5e-4, 80, 128
import os as _os2
EPOCHS = int(_os2.environ.get('EPOCHS', EPOCHS))   # cosine schedule length (default 80; Tiny-ImageNet runs use 40)
simulate.STAGE_BLOCKS = [5, 5, 5]; simulate.GLOBAL_AVG_POOL = True
import os
DATASET = os.environ.get('DATASET', 'cifar100'); simulate.set_dataset(DATASET)   # tinyimagenet: cache on CPU
cache = torch.device('cpu') if DATASET == 'tinyimagenet' else dev
BLOCKS = [int(c) for c in os.environ.get('BLOCKS', '555')]; simulate.STAGE_BLOCKS = BLOCKS   # §23 ablation
simulate.RESIDUAL = bool(int(os.environ.get('RESIDUAL', 1)))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'baseline_555.pt'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # init + data-order seed; quant runs pair by seed

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()
torch.manual_seed(SEED)
tl, te = load_data(N_BATCH)
train = [(x.to(cache).to(torch.float32), y.to(dev).argmax(1)) for x, y in tl]
test  = [(x.to(cache).to(torch.float32), y.argmax(1).to(dev)) for x, y in te]
ce = torch.nn.CrossEntropyLoss()

def evaluate(m):
    m.eval(); c1=c5=n=0
    with torch.no_grad():
        for x,tgt in test:
            lo=m(x.to(dev)); c1+=(lo.argmax(1)==tgt).sum().item()
            c5+=(lo.topk(5,1).indices==tgt.unsqueeze(1)).any(1).sum().item(); n+=x.shape[0]
    return 100*c1/n,100*c5/n

torch.manual_seed(SEED)
m = CNNModelBaseline().to(dev)
opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=MOM, weight_decay=WD)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
log('BS baseline %s%s %s %.2fM  aug lr=%g cosine %d ep' % (BLOCKS, '' if simulate.RESIDUAL else ' PLAIN(no skips)', DATASET, sum(p.numel() for p in m.parameters())/1e6, LR, EPOCHS))
acc = []
for ep in range(1, EPOCHS+1):
    t0=time.time(); m.train()
    for x,tgt in train:
        opt.zero_grad(); l=ce(m(augment_batch(x.to(dev))),tgt); l.backward(); opt.step()
    sch.step()
    t1,t5 = evaluate(m); acc.append((t1,t5))
    if os.environ.get('BS_DIAG'):   # same quantities as QC_BNDIAG + per-stage mean conv-weight norm (float reference)
        with torch.no_grad():
            bns = [(n, b) for n, b in m.named_modules() if isinstance(b, torch.nn.BatchNorm2d)]
            rv = torch.cat([b.running_var for _, b in bns]); rm = torch.cat([b.running_mean for _, b in bns])
            worst = max(bns, key=lambda nb: nb[1].running_var.max().item())
            st = {}
            for n, p in m.named_parameters():
                if p.dim() == 4:
                    k = 'stem' if n.startswith('stem') else 'stage' + n.split('.')[1]; st.setdefault(k, []).append(p.norm().item())
            log('BSDIAG e%d | t1 %.2f | running_var min %.3g max %.3g (%s) | |running_mean| max %.3g | |param| max %.3g | |BN gamma| max %.3g | conv |w| %s'
                % (ep, t1, rv.min().item(), rv.max().item(), worst[0], rm.abs().max().item(), max(p.abs().max().item() for p in m.parameters()),
                   max(b.weight.abs().max().item() for _, b in bns), ' '.join('%s %.2f' % (k, sum(v)/len(v)) for k, v in st.items())))
    if ep%5==0 or ep==1 or ep>=EPOCHS-2:
        log('BS e%-3d | t1 %5.2f / t5 %5.2f | %.0fs' % (ep,t1,t5,time.time()-t0))
final = {n: p.detach().float().cpu().clone() for n,p in m.named_parameters()}
torch.save({'acc': acc, 'final_params': final, 'blocks': BLOCKS, 'seed': SEED, 'dataset': DATASET, 'residual': simulate.RESIDUAL}, OUT)
log('BS best t1 %.2f (e%d) -> saved %s' % (max(a[0] for a in acc), 1+max(range(EPOCHS),key=lambda i:acc[i][0]), OUT))
log('BS DONE')
