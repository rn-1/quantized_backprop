import torch, sys, simulate
from simulate import CNNModelQ, load_data, augment_batch
# §21.6 option-3 proxy for the BN_STATS_F rerun. The running stats are an EMA (momentum 0.1, ~20-40 batch
# memory) and the cosine lr at e80 is ~4e-5, so the stats a rerun would END with ~= the stats accumulated
# by running the saved final weights forward over one more augmented training epoch with the in-training
# fixed-point update rule. F=0 (old 2^-8 rule) should reproduce the measured 68.95 -- that validates the proxy;
# F=16 predicts the rerun. Usage: _bn_stats_proxy.py <seed>
seed = int(sys.argv[1]); dev = torch.device('cuda')
simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True; simulate.BITs = 8; simulate.B_IV = 8
simulate.BN_EARLY_DIV_BITS = 5; simulate.RING_BITS = 32; simulate.DIV_PUBLIC_RING = 1; simulate.LIN_GRAD_PRESHIFT = 2
torch.manual_seed(seed); tl, te = load_data(128)
test = [(x.to(dev).double(), y.argmax(1).to(dev)) for x, y in te]
state = torch.load('/home/ryan/checkpoints/quant_final_s%d.pt' % seed)['state']
out = []
for F in (0, 16):
    simulate.BN_STATS_F = F; torch.manual_seed(1000 + seed)
    m = CNNModelQ().to(dev); m.load_state_dict(state)
    m.train()
    with torch.no_grad():
        for x, y in tl: m(augment_batch(x.to(dev).double()))   # BN updates running stats; weights untouched
    v = torch.cat([b.running_var for b in m.modules() if isinstance(b, simulate.QBatchNorm2d)])
    m.eval(); c = n = 0
    with torch.no_grad():
        for x, t in test: c += (m(x).argmax(1) == t).sum().item(); n += x.shape[0]
    out.append('F=%d top1 %.2f (running_var<=0: %d/%d)' % (F, 100 * c / n, (v <= 0).sum().item(), v.numel()))
print('seed', seed, ' | '.join(out))
