import torch, sys, math
import simulate
from simulate import (CNNModelBaseline, CNNModelQ, sync_models, load_data,
                      paired_tensors, calculateFinalParamError)

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR = 0.001

def log(s):
    sys.stderr.write(s + '\n'); sys.stderr.flush()

log('EV device %s  BITs=%d' % (dev, simulate.BITs))

# --- cache one full epoch of training batches (identical order for both models) ---
torch.manual_seed(0)
train_loader, test_loader = load_data(128)
train_batches = [(x.clone(), y.clone()) for x, y in train_loader]
test_batches  = [(x.clone(), y.clone()) for x, y in test_loader]
log('EV cached %d train batches, %d test batches' % (len(train_batches), len(test_batches)))

# --- build + sync so both start from identical weights ---
torch.manual_seed(0)
model_base = CNNModelBaseline().to(dev)
model_Q = CNNModelQ().to(dev)
sync_models(model_base, model_Q)

ce = torch.nn.CrossEntropyLoss()

def train(model, dtype):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    model.train()
    losses = []
    for x, y in train_batches:
        x = x.to(dev).to(dtype); target = y.to(dev).argmax(1)
        opt.zero_grad(); out = model(x); loss = ce(out, target)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

log('EV training baseline...')
bl = train(model_base, torch.float32)
log('EV training quantized...')
ql = train(model_Q, torch.float64)
log('EV baseline  loss first %.4f -> last %.4f  min %.4f' % (bl[0], bl[-1], min(bl)))
log('EV quantized loss first %.4f -> last %.4f  min %.4f' % (ql[0], ql[-1], min(ql)))

# --- per-layer relative param error (RMSE per layer / baseline layer RMS) ---
log('EV --- per-layer parameter error (quantized vs baseline) ---')
log('EV %-14s %10s %12s %12s' % ('param', 'numel', 'rel_rmse', '|base|_rms'))
for name, base, quant in paired_tensors(model_base, model_Q, include_running=False):
    b = base.detach().cpu().to(torch.float64)
    q = quant.detach().cpu().to(torch.float64)
    rmse = math.sqrt(torch.mean((b - q) ** 2).item())
    ref = math.sqrt(torch.mean(b ** 2).item())
    rel = rmse / ref if ref > 1e-12 else float('nan')
    log('EV %-14s %10d %11.3f%% %12.4g' % (name, b.numel(), 100 * rel, ref))
log('EV aggregate param RMSE (relative): %.3f%%' % (100 * calculateFinalParamError(model_base, model_Q)))

# --- classification accuracy on the test set ---
# both models forward in train() mode so BatchNorm uses batch statistics: the quantized
# BatchNorm has no running-stats eval path (training=True is hardcoded), so this keeps the
# two comparable rather than confounding quantization error with a BN-mode difference.
def top1_top5(model, dtype):
    model.train()
    c1 = c5 = n = 0
    with torch.no_grad():
        for x, y in test_batches:
            x = x.to(dev).to(dtype)
            logits = model(x)
            target = y.argmax(1).to(dev)
            pred1 = logits.argmax(1)
            c1 += (pred1 == target).sum().item()
            top5 = logits.topk(5, dim=1).indices
            c5 += (top5 == target.unsqueeze(1)).any(1).sum().item()
            n += x.shape[0]
    return 100 * c1 / n, 100 * c5 / n, n

log('EV --- test-set classification accuracy (CIFAR-100, %d classes) ---' % 100)
b1, b5, n = top1_top5(model_base, torch.float32)
q1, q5, _ = top1_top5(model_Q, torch.float64)
log('EV baseline   top1 %.2f%%  top5 %.2f%%  (n=%d)' % (b1, b5, n))
log('EV quantized  top1 %.2f%%  top5 %.2f%%  (n=%d)' % (q1, q5, n))
log('EV chance     top1 %.2f%%  top5 %.2f%%' % (1.0, 5.0))
log('EV DONE')
