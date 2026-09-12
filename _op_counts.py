import torch, sys
import simulate
from simulate import (CNNModelQ, load_data, reset_op_counts, snapshot_op_counts,
                      attach_relu_counter, OP_COUNTS)

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH = 128

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

log('OC device %s  BITs=%d  batch=%d' % (dev, simulate.BITs, BATCH))

torch.manual_seed(0)
train_loader, _ = load_data(BATCH)
x, y = next(iter(train_loader))
x = x.to(dev).to(torch.float64); target = y.to(dev).argmax(1)

model = CNNModelQ().to(dev)
model.train()
attach_relu_counter(model)
ce = torch.nn.CrossEntropyLoss()

# ---- one forward, snapshot, then backward ----
reset_op_counts()
out = model(x)
fwd = snapshot_op_counts()
loss = ce(out, target)
loss.backward()
tot = snapshot_op_counts()

OPS = [('probabilistic truncations', 'p_truncate'),
       ('inverse-sqrt (BatchNorm)',  'inv_sqrt'),
       ('ReLU comparisons (DReLU)',  'relu')]

def row(label, key):
    fc, tc = fwd[key + '_calls'], tot[key + '_calls']
    fe, te = fwd[key + '_elems'], tot[key + '_elems']
    bc, be = tc - fc, te - fe
    return (label, fc, bc, tc, fe, be, te)

log('')
log('OC === per training step (batch=%d), one fwd + one bwd ===' % BATCH)
log('OC %-28s | %8s %8s %8s | %14s %14s %14s' %
    ('operation', 'fwd', 'bwd', 'TOTAL', 'fwd elems', 'bwd elems', 'TOTAL elems'))
log('OC ' + '-' * 108)
tot_calls = tot_elems = 0
for label, key in OPS:
    l, fc, bc, tc, fe, be, te = row(label, key)
    log('OC %-28s | %8d %8d %8d | %14d %14d %14d' % (l, fc, bc, tc, fe, be, te))
    tot_calls += tc; tot_elems += te
log('OC ' + '-' * 108)
log('OC %-28s | %8s %8s %8d | %14s %14s %14d' % ('ALL (round-ish / bandwidth)', '', '', tot_calls, '', '', tot_elems))

log('')
log('OC === normalised ===')
log('OC truncations / image      : %.1f' % (tot['p_truncate_calls'] / BATCH))
log('OC truncated elems / image  : %.0f' % (tot['p_truncate_elems'] / BATCH))
log('OC inv-sqrt calls (all fwd) : %d  (%d elems)' % (tot['inv_sqrt_calls'], tot['inv_sqrt_elems']))
log('OC relu calls (all fwd)     : %d  (%d elems)' % (tot['relu_calls'], tot['relu_elems']))
log('OC DONE')
