import torch, sys, math, json, os, time
import torch.nn.functional as F
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, augment_batch, cross_entropy_q

# §23 ablation probe: how do skip connections and depth move the two edges of the loss-scale window?
#   lower edge (underflow): backward gradients that fall below the 2^-BITs grid after scaling by S.
#   upper edge (overflow):  the largest payload entering a truncation, vs 2^(l-1).
# No training of the fixed-point model: for each architecture we take two weight points,
#   init      = seed-0 init
#   pre       = after PRE epochs of plain float SGD (lr 0.1, mom 0.9, wd 5e-4, aug), to get past init transients
# and run identical, un-augmented batches through a float64 reference and the fixed-point model (final
# recipe, f=8) at several S. Per (arch, point, S) we record:
#   act_subgrid[d]  fraction of backward ACTIVATION gradients at block d with |S*g| < 2^-BITs (float64 ref:
#                   what the fixed-point grid cannot represent; d = block index from input to output)
#   act_med[d]      median |S*g| at block d, in grid steps
#   w_zero          fraction of weight-gradient elements that are exactly 0 in fixed point where float64 is not
#   cos / ratio     global weight-gradient cosine and norm ratio, fixed point vs float64 (Gate B metric)
#   payload         max |input| to ring_truncate (fwd and bwd) and the top site (RING_BITS=57: nothing wraps)
# Usage: _ablation_probe.py <out.json>   env: ARCHS (default 222r,222p,555r,555p,999r,999p), LOGS_LIST
#        (default 12,14,15,16,17,18,20), NB (batches/point, default 4), PRE (float epochs, default 3), DATASET
OUT = sys.argv[1]
ARCHS = os.environ.get('ARCHS', '222r,222p,555r,555p,999r,999p').split(',')
LOGS_LIST = [int(v) for v in os.environ.get('LOGS_LIST', '12,14,15,16,17,18,20').split(',')]
NB = int(os.environ.get('NB', 4)); PRE = int(os.environ.get('PRE', 3))
dev = torch.device('cuda')
simulate.set_dataset(os.environ.get('DATASET', 'cifar100')); simulate.GLOBAL_AVG_POOL = True
simulate.BITs = simulate.B_IV = 8; simulate.SOFTMAX_EXTRA_BITS = 9
simulate.BN_EARLY_DIV_BITS = 5; simulate.DIV_PUBLIC_RING = 1; simulate.LIN_GRAD_PRESHIFT = 2; simulate.BN_STATS_F = 16
simulate.RING_BITS = 57
G = 2.0 ** -simulate.BITs

def log(s): sys.stderr.write(s + '\n'); sys.stderr.flush()
torch.manual_seed(0)
tl, _ = load_data(128)
train = [(x.to(dev), y.to(dev).argmax(1)) for x, y in tl]
probe = [(x.double(), t) for x, t in train[:NB]]

PHASE = ['fwd']; PEAK = {}; orig_rt = simulate.ring_truncate
def spy(t, m):
    f = sys._getframe(1)
    while f is not None and f.f_code.co_name in ('p_truncate',): f = f.f_back
    k = (PHASE[0], '%s:%d' % (f.f_code.co_name, f.f_lineno)); v = t.abs().max().item()
    if v > PEAK.get(k, 0): PEAK[k] = v
    return orig_rt(t, m)

def blocks_of(m):   # residual-stage BasicBlocks in input->output order
    return [b for st in m.stages for b in st]

def float_pretrain(blocks, residual):
    simulate.STAGE_BLOCKS = blocks; simulate.RESIDUAL = residual
    torch.manual_seed(0); m = CNNModelBaseline().to(dev)
    init = {k: v.detach().clone() for k, v in m.state_dict().items()}
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    m.train()
    for _ in range(PRE):
        for x, t in train:
            opt.zero_grad(); F.cross_entropy(m(augment_batch(x)), t).backward(); opt.step()
    return init, {k: v.detach().clone() for k, v in m.state_dict().items()}

def act_grads(mb, S):
    # backward activation gradients at each block output of the float64 reference, scaled by S
    store = []
    def fwd_hook(mod, i, o):          # must return None: a forward hook's return value replaces the output
        o.register_hook(lambda g: store.append(g.detach()))
    hs = [b.register_forward_hook(fwd_hook) for b in blocks_of(mb)]
    sub = [[] for _ in blocks_of(mb)]; med = [[] for _ in blocks_of(mb)]; wgrads = []
    for x, t in probe:
        store.clear(); mb.zero_grad(); F.cross_entropy(mb(x), t).backward()
        for d, g in enumerate(reversed(store)):          # hooks fire output->input; reverse to input->output
            a = (S * g).abs().flatten()
            sub[d].append((a < G).double().mean().item()); med[d].append(a.median().item() / G)
        wgrads.append({n: p.grad.detach().clone() for n, p in mb.named_parameters()})
    for h in hs: h.remove()
    return [sum(v) / len(v) for v in sub], [sum(v) / len(v) for v in med], wgrads

def q_grads(mq, S):
    out = []
    for x, t in probe:
        mq.zero_grad(); PHASE[0] = 'fwd'; o = mq(x); loss = cross_entropy_q(o, t)
        PHASE[0] = 'bwd'; (S * loss).backward()
        out.append({n: (p.grad.detach() / S).clone() for n, p in mq.named_parameters() if p.grad is not None})
    return out

results = []
simulate.ring_truncate = spy
for arch in ARCHS:
    blocks = [int(c) for c in arch[:3]]; residual = arch[3] == 'r'
    t0 = time.time(); init, pre = float_pretrain(blocks, residual)
    log('AB arch %s (%s) pretrained %d float epochs in %.0fs' % (blocks, 'skips' if residual else 'PLAIN', PRE, time.time() - t0))
    for point, sd in (('init', init), ('pre', pre)):
        simulate.STAGE_BLOCKS = blocks; simulate.RESIDUAL = residual
        mb = CNNModelBaseline().to(dev); mb.load_state_dict(sd); mb = mb.double(); mb.train()
        mq = CNNModelQ().to(dev); sync_models(mb, mq); mq.train()
        bufs = {k: v.clone() for k, v in mq.state_dict().items() if 'running' in k}
        for logS in LOGS_LIST:
            S = 2.0 ** logS; PEAK.clear()
            sub, med, gf = act_grads(mb, S)
            mq.load_state_dict(bufs, strict=False)        # identical BN running buffers for every S
            gq = q_grads(mq, S)
            dot = qq = ff = 0.0; zero = nz = 0
            for a_all, b_all in zip(gq, gf):
                for n, b in b_all.items():
                    a = a_all.get(n)
                    if a is None: continue
                    dot += (a * b).sum().item(); qq += (a * a).sum().item(); ff += (b * b).sum().item()
                    m = b != 0; zero += ((a == 0) & m).sum().item(); nz += m.sum().item()
            pk = {ph: max([(v, k[1]) for k, v in PEAK.items() if k[0] == ph], default=(0, '-')) for ph in ('fwd', 'bwd')}
            r = dict(arch=arch, blocks=blocks, residual=residual, point=point, logS=logS,
                     act_subgrid=sub, act_med=med, w_zero=zero / max(nz, 1),
                     cos=dot / math.sqrt(qq * ff + 1e-300), ratio=math.sqrt(qq / (ff + 1e-300)),
                     payload_fwd=math.log2(pk['fwd'][0] or 1), payload_bwd=math.log2(pk['bwd'][0] or 1),
                     payload_bwd_site=pk['bwd'][1])
            results.append(r)
            log('AB %s %-4s S=2^%-2d | act sub-grid first/mid/last %.3f/%.3f/%.3f | med steps first/last %.1f/%.1f'
                ' | w-grad zeroed %.3f | cos %.4f ratio %.3f | payload fwd 2^%.2f bwd 2^%.2f (%s)'
                % (arch, point, logS, sub[0], sub[len(sub) // 2], sub[-1], med[0], med[-1], r['w_zero'],
                   r['cos'], r['ratio'], r['payload_fwd'], r['payload_bwd'], r['payload_bwd_site']))
        del mb, mq; torch.cuda.empty_cache()
simulate.ring_truncate = orig_rt
json.dump(results, open(OUT, 'w'), indent=1)
log('AB DONE -> %s' % OUT)
