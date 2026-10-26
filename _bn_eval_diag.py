import torch, sys, simulate
from simulate import CNNModelQ, load_data, augment_batch, to_fixed, ring_truncate
# §21.6 diagnostic on the saved f=8 models: is the running-stat eval gap the WEIGHTS or the BN STATISTICS?
#  stored : running stats as accumulated during training (2^-8 grid, momentum update) -- the protocol in evalfix
#  recal  : stats re-estimated after training in float64 over all train batches (aug), encoded at 2^-8
#  fold16 : recal stats, BN folded to per-channel scale/shift held at 2^-16 (MPC: a one-off precomputation;
#           then one secret multiply + truncation per activation, payload ~2^(8+16+mag) << 2^31)
#  batch  : old protocol (each test batch's own stats), for reference
seed = int(sys.argv[1]); dev = torch.device('cuda')
simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True; simulate.BITs = 8; simulate.B_IV = 8
simulate.BN_EARLY_DIV_BITS = 5; simulate.RING_BITS = 32; simulate.DIV_PUBLIC_RING = 1; simulate.LIN_GRAD_PRESHIFT = 2
torch.manual_seed(seed); tl, te = load_data(128)
test = [(x.to(dev).double(), y.argmax(1).to(dev)) for x, y in te]
m = CNNModelQ().to(dev); m.load_state_dict(torch.load('/home/ryan/checkpoints/quant_final_s%d.pt' % seed)['state'])
bns = [b for b in m.modules() if isinstance(b, simulate.QBatchNorm2d)]
def acc(mode):
    m.train() if mode == 'batch' else m.eval(); c = n = 0
    keep = [b.clone() for b in m.buffers()]
    with torch.no_grad():
        for x, t in test: c += (m(x).argmax(1) == t).sum().item(); n += x.shape[0]
    for b, s in zip(m.buffers(), keep): b.copy_(s)
    return 100 * c / n
stored = [(b.running_mean.clone(), b.running_var.clone()) for b in bns]
r = {'stored': acc('eval'), 'batch': acc('batch')}
# recalibrate: exact per-channel mean / unbiased var of each BN input over the training set (fixed-point forward)
sums = {id(b): [0, 0, 0] for b in bns}
def hook(b, inp, out):
    x = inp[0]; d = [0, 2, 3]; s = sums[id(b)]
    s[0] += x.sum(d); s[1] += (x * x).sum(d); s[2] += x.numel() // x.shape[1]
hs = [b.register_forward_hook(hook) for b in bns]
m.train(); keep = [b.clone() for b in m.buffers()]
with torch.no_grad():
    for x, y in tl: m(augment_batch(x.to(dev).double()))
for b, s in zip(m.buffers(), keep): b.copy_(s)
for h in hs: h.remove()
# NOTE: this is the population over all train batches given each layer's input under batch-stat
# normalisation upstream -- the same quantity torch's running stats estimate.
for b in bns:
    s0, s1, n = sums[id(b)]; mu = s0 / n; var = (s1 / n - mu * mu) * n / (n - 1)
    b.running_mean.copy_(mu); b.running_var.copy_(var)
r['recal'] = acc('eval')
# folded BN at 2^-16 for the per-channel constants
F2 = 16
def folded(self, x):
    if self.training: return orig(self, x)
    sc = self.weight / torch.sqrt(self.running_var + 1e-5); sh = self.bias - self.running_mean * sc
    xi, = to_fixed(x); sci = torch.round(sc * 2**F2).long().view(1,-1,1,1); shi = torch.round(sh * 2**simulate.BITs).long().view(1,-1,1,1)
    return (ring_truncate(xi * sci, F2) + shi).double() / 2**simulate.BITs
orig = simulate.QBatchNorm2d.forward; simulate.QBatchNorm2d.forward = folded
r['fold16_recal'] = acc('eval')
for b, (mu, v) in zip(bns, stored): b.running_mean.copy_(mu); b.running_var.copy_(v)
r['fold16_stored'] = acc('eval')
simulate.QBatchNorm2d.forward = orig
# how far are the stored (training-accumulated, 2^-8 grid) stats from the recalibrated ones?
print('seed', seed, ' '.join('%s %.2f' % kv for kv in r.items()))
