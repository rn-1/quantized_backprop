import torch, sys, simulate
from simulate import CNNModelBaseline, load_data, augment_batch
# §21.6 control: give the float baseline the SAME post-hoc BN recalibration as the fixed-point models
# (population mean / unbiased var over all augmented train batches), so the comparison stays like-for-like.
seed = int(sys.argv[1]); snap = sys.argv[2]; dev = torch.device('cuda')
simulate.STAGE_BLOCKS = [5,5,5]; simulate.GLOBAL_AVG_POOL = True
torch.manual_seed(seed); tl, te = load_data(128)
m = CNNModelBaseline().to(dev).double()
missing = m.load_state_dict({k: v.to(dev).double() for k, v in torch.load(snap)['final_params'].items()}, strict=False)
assert not missing.unexpected_keys and all('running' in k or 'num_batches' in k for k in missing.missing_keys)
bns = [b for b in m.modules() if isinstance(b, torch.nn.BatchNorm2d)]
for b in bns: b.reset_running_stats(); b.momentum = None        # momentum=None -> cumulative (population) average
m.train()
with torch.no_grad():
    for x, y in tl: m(augment_batch(x.to(dev).double()))
m.eval(); c = n = 0
with torch.no_grad():
    for x, y in te: c += (m(x.to(dev).double()).argmax(1) == y.argmax(1).to(dev)).sum().item(); n += x.shape[0]
print('seed', seed, 'float recal %.2f' % (100 * c / n))
