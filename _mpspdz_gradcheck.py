import torch, torch.fx, re, sys, math
import torch.nn as nn, torch.nn.functional as F

# BITS_REPORT §21.4 (check 3a): compare MP-SPDZ emulator gradients (gradcheck.mpc) with torch float64.
# The model classes are exec'd from the .mpc source itself, so both sides build the identical network.
# Reference matches MP-SPDZ semantics: BN normalises with the UNBIASED batch variance; loss is the SUM
# over the batch (MP-SPDZ defers the 1/B to the update). Usage: _mpspdz_gradcheck.py <seq|res>
mode = sys.argv[1]
src = open('/home/ryan/mp-spdz-0.4.3/Programs/Source/gradcheck.mpc').read()
cls_src = src[src.index('class Block'):src.index('net = Res()')]
exec(cls_src)
d = torch.load('/home/ryan/checkpoints/mpspdz_gradcheck_%s.pt' % mode)
net = (Res() if mode == 'res' else Seq()).double()
net.load_state_dict(d['state'])

def bn_unbiased(self, x):
    mu = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), unbiased=True, keepdim=True)
    return (x - mu) / torch.sqrt(var + self.eps) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.forward = bn_unbiased.__get__(m)

x, y = d['x'].double(), d['y']
net.zero_grad()
F.cross_entropy(net(x), y, reduction='sum').backward()

# MP-SPDZ gradients, keyed by layer index
mp = {}
for line in open('/home/ryan/checkpoints/logs/mpspdz/gradcheck_%s.txt' % mode):
    m_ = re.match(r'GRAD (\d+) (\w+) (\d+) \[(.*)\]', line.strip())
    if m_:
        mp.setdefault(int(m_.group(1)), {})[int(m_.group(3))] = (m_.group(2), torch.tensor([float(v) for v in m_.group(4).split(',')], dtype=torch.float64))

# torch modules with parameters, in torch.fx call order (= MP-SPDZ layer order)
mods = dict(net.named_modules())
order = [n.target for n in torch.fx.symbolic_trace(net).graph.nodes if n.op == 'call_module']
tmods = [(t, mods[t]) for t in order if any(True for _ in mods[t].parameters(recurse=False))]
mlayers = [i for i in sorted(mp)]
assert len(tmods) == len(mlayers), (len(tmods), len(mlayers))
print('%-12s %-6s %-5s | %8s %8s | %s' % ('torch module', 'MP', 'param', 'cosine', 'ratio', 'rel.err'))
worst = 1.0
for (name, mod), li in zip(tmods, mlayers):
    for j, (pname, p) in enumerate(mod.named_parameters(recurse=False)):
        typ, g_mp = mp[li][j]
        g = p.grad.detach()
        if isinstance(mod, nn.Conv2d) and pname == 'weight': g = g.permute(0, 2, 3, 1)   # [out,kh,kw,in]
        if isinstance(mod, nn.Linear) and pname == 'weight': g = g.t()                   # [in,out]
        g = g.reshape(-1)
        assert g.numel() == g_mp.numel(), (name, pname, g.numel(), g_mp.numel())
        cos = (g @ g_mp / (g.norm() * g_mp.norm() + 1e-300)).item()
        ratio = (g_mp.norm() / (g.norm() + 1e-300)).item()
        rel = ((g_mp - g).norm() / (g.norm() + 1e-300)).item()
        worst = min(worst, cos)
        print('%-12s %-6s %-5s | %8.5f %8.4f | %.2e' % (name, typ[:6], pname[:5], cos, ratio, rel))
print('WORST cosine (%s): %.5f' % (mode, worst))
