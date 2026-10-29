import numpy as np, os, sys
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Convert the CS231n Tiny-ImageNet release (data/tiny-imagenet-200/) into one uint8 cache,
# data/tiny-imagenet-200.npz: x_train (100000,64,64,3), y_train, x_val (10000,64,64,3), y_val.
# Labels index the sorted wnids.txt. The labelled val split is used as the test set (the official test split
# has no labels). Some images are greyscale, so all are converted to RGB.
R = '/home/ryan/data/tiny-imagenet-200'
wnids = sorted(open(f'{R}/wnids.txt').read().split())
cls = {w: i for i, w in enumerate(wnids)}

def load(p):
    return np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)

def run(paths):
    with ProcessPoolExecutor(32) as ex:
        return np.stack(list(ex.map(load, paths, chunksize=256)))

tr = [(f'{R}/train/{w}/images/{f}', cls[w]) for w in wnids for f in sorted(os.listdir(f'{R}/train/{w}/images'))]
va = [(f'{R}/val/images/{l.split()[0]}', cls[l.split()[1]]) for l in open(f'{R}/val/val_annotations.txt')]
x_tr = run([p for p, _ in tr]); y_tr = np.array([c for _, c in tr], dtype=np.int64)
x_va = run([p for p, _ in va]); y_va = np.array([c for _, c in va], dtype=np.int64)
assert x_tr.shape == (100000, 64, 64, 3) and x_va.shape == (10000, 64, 64, 3), (x_tr.shape, x_va.shape)
assert np.bincount(y_tr).tolist() == [500] * 200 and np.bincount(y_va).tolist() == [50] * 200
np.savez('/home/ryan/data/tiny-imagenet-200.npz', x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va)
m = x_tr.reshape(-1, 3).mean(0) / 255; s = x_tr.reshape(-1, 3).std(0) / 255
print('saved; train mean', m.round(4).tolist(), 'std', s.round(4).tolist())
