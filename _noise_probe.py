import torch, sys, math
import simulate
from simulate import CNNModelBaseline, CNNModelQ, sync_models, load_data, to_fixed

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

# line number -> human label for every live p_truncate call site (fast_rsqrt 65-82 is dead)
SITE = {
    129: 'inv_sqrt a*u', 130: 'inv_sqrt (p-b)*u', 138: 'inv_sqrt g*m (result)',
    452: 'Conv.fwd output', 559: 'Conv.bwd grad_in/kernel',
    588: 'Linear.fwd result', 621: 'Linear.bwd grad_in/weight',
    678: 'BN.fwd variance', 685: 'BN.fwd var_unbiased', 700: 'BN.fwd running_var',
    701: 'BN.fwd running_mean', 757: 'BN.fwd x_norm', 776: 'BN.fwd result',
    818: 'BN.bwd grad_weight', 831: 'BN.bwd grad_output', 835: 'BN.bwd grad_input',
    847: 'BN.bwd grad_mean', 853: 'BN.bwd grad_std a', 860: 'BN.bwd grad_std b',
    863: 'BN.bwd grad_std c',
}
def group(label):
    if label.startswith('inv_sqrt'): return 'inv_sqrt'
    if label.startswith('Conv'):     return 'Conv'
    if label.startswith('Linear'):   return 'Linear'
    return 'BN(other)'

STATS = {}
_orig_ptrunc = simulate.p_truncate
def ptrunc_spy(*tensors):
    line = sys._getframe(1).f_lineno
    out = _orig_ptrunc(*tensors)                       # passive: identical to real path
    for t, o in zip(tensors, out):
        # p_truncate computes y = (tensor >> m)/2^m, i.e. tensor/2^(2m) as a real value; its
        # exact (un-rounded) result is tensor/2^(2*BITs). noise = realized - exact.
        ideal = t.to(torch.float64) / (2 ** (2 * simulate.BITs))
        noise = o - ideal                                    # quantization noise this truncation injects
        a = STATS.setdefault(line, {'sig2': 0.0, 'noise2': 0.0, 'n': 0, 'zero': 0, 'absideal': 0.0})
        a['sig2']    += (ideal * ideal).sum().item()
        a['noise2']  += (noise * noise).sum().item()
        a['absideal']+= ideal.abs().sum().item()
        a['zero']    += ((o == 0) & (ideal != 0)).sum().item()
        a['n']       += t.numel()
    return out

# ============================================================================
# PART A -- per-op injected-noise SNR on a live forward+backward
# ============================================================================
def part_A(bits, warmup=10):
    simulate.BITs = bits
    torch.manual_seed(0)
    mb = CNNModelBaseline().to(dev)
    mq = CNNModelQ().to(dev)
    sync_models(mb, mq)
    mq.train()
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(mq.parameters(), lr=0.001, momentum=0.9)

    torch.manual_seed(0)
    tl, _ = load_data(128)
    it = iter(tl)
    batches = [next(it) for _ in range(warmup + 1)]

    for i in range(warmup):                            # reach a realistic weight/grad regime
        x, y = batches[i]
        x = x.to(dev).to(torch.float64); tgt = y.to(dev).argmax(1)
        opt.zero_grad(); loss = ce(mq(x), tgt); loss.backward(); opt.step()

    STATS.clear()
    simulate.p_truncate = ptrunc_spy
    try:
        x, y = batches[warmup]
        x = x.to(dev).to(torch.float64); tgt = y.to(dev).argmax(1)
        opt.zero_grad(); loss = ce(mq(x), tgt); loss.backward()
    finally:
        simulate.p_truncate = _orig_ptrunc

    log('')
    log('NP === PART A: per-op injected-noise SNR  (BITs=%d, after %d warmup batches) ===' % (bits, warmup))
    log('NP grid resolution 2^-BITs = %.3e' % (2.0 ** -bits))
    log('NP %-26s %10s %11s %11s %9s %8s %7s' %
        ('op (call site)', 'elems', 'RMS signal', 'RMS noise', 'rel.noise', 'SNR dB', 'zero%'))
    log('NP ' + '-' * 90)
    rows = []
    for line, a in STATS.items():
        if a['n'] == 0: continue
        rms_sig = math.sqrt(a['sig2'] / a['n'])
        rms_noise = math.sqrt(a['noise2'] / a['n'])
        rel = rms_noise / rms_sig if rms_sig > 0 else float('inf')
        snr = 20 * math.log10(rms_sig / rms_noise) if rms_noise > 0 else float('inf')
        rows.append((rel, line, a['n'], rms_sig, rms_noise, snr, a['zero']))
    for rel, line, n, rs, rn, snr, z in sorted(rows, reverse=True):   # worst (noisiest) first
        log('NP %-26s %10d %11.3e %11.3e %8.2f%% %8.1f %6.2f%%' %
            (SITE.get(line, 'line %d' % line), n, rs, rn, 100 * rel, snr, 100 * z / n))

    # roll up by group
    gg = {}
    for rel, line, n, rs, rn, snr, z in rows:
        g = group(SITE.get(line, ''))
        d = gg.setdefault(g, {'noise2': 0.0, 'sig2': 0.0, 'n': 0})
        d['noise2'] += rn * rn * n; d['sig2'] += rs * rs * n; d['n'] += n
    log('NP --- by op group (share of total injected noise power) ---')
    total_np = sum(d['noise2'] for d in gg.values())
    for g, d in sorted(gg.items(), key=lambda kv: -kv[1]['noise2']):
        share = 100 * d['noise2'] / total_np if total_np > 0 else 0
        log('NP %-12s noise-power share %6.2f%%   (%d elems truncated)' % (g, share, d['n']))

# ============================================================================
# PART B -- controlled inverse-sqrt underflow map across BITs
# ============================================================================
def part_B(bits_list):
    log('')
    log('NP === PART B: inverse_sqrt underflow vs variance magnitude (controlled) ===')
    log('NP for each BITs, feed var = 2^k and report where 1/sqrt(var) rounds to 0 on the grid')
    variances = [2.0 ** k for k in range(0, 44, 2)]
    for bits in bits_list:
        simulate.BITs = bits
        first_zero = None
        checks = []
        for var in variances:
            xf, = to_fixed(torch.tensor([var], dtype=torch.float64, device=dev))
            r = simulate.inverse_sqrt(xf)
            true = 1.0 / math.sqrt(var)
            got = r.abs().max().item()
            zero = (got == 0.0)
            if zero and first_zero is None:
                first_zero = var
            checks.append((var, true, got, zero))
        thr = ('2^%.0f' % math.log2(first_zero)) if first_zero else '>2^42'
        pred = 2.0 ** (2 * bits + 2)     # the floor from the report: var > 2^(2*BITs+2)
        log('NP BITs=%-2d  grid 2^-%-2d  underflow onset var=%-7s  (predicted ~2^%.0f = var>%.2g)'
            % (bits, bits, thr, math.log2(pred), pred))

log('NP device %s' % dev)
part_A(22)
part_A(16)
part_B([7, 13, 16, 19, 22])
simulate.BITs = 22
log('NP DONE')
