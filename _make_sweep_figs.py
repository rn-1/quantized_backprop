import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

OUT = '/home/ryan/report_figs'
with open(OUT + '/sweep_curves_data.json') as f:
    D = json.load(f)
ep = D['epochs']

SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; MUTED='#898781'; GRID='#e1e0d9'; AXIS='#c3c2b7'
# S value -> (color, label). baseline is a gray dashed reference, not a category.
CFG = [
    ('S8',  '#e34948', 'S=2⁸  (below knee)'),
    ('S10', '#eb6834', 'S=2¹⁰ (knee)'),
    ('S12', '#2a78d6', 'S=2¹² (recommended)'),
    ('S26', '#1baf7a', 'S=2²⁶ (ceiling pressure)'),
]
ESTOP = 9

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':11,
    'figure.facecolor':SURF,'axes.facecolor':SURF,'text.color':INK,'axes.labelcolor':SEC,
    'xtick.color':MUTED,'ytick.color':MUTED})

def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)

# ---- Fig 1: loss ----
fig, ax = plt.subplots(figsize=(7.6,4.6), dpi=150); style(ax)
ax.plot(ep, D['baseline']['loss'], color=SEC, lw=2, ls=(0,(4,3)), zorder=2, label='baseline (float32)')
for key, col, lab in CFG:
    ax.plot(ep, D[key]['loss'], color=col, lw=2, marker='o', ms=4, zorder=3, label=lab)
ax.set_xlabel('epoch'); ax.set_ylabel('training loss (CrossEntropy)')
ax.set_title('Training loss across loss-scale S  (BITs=13)', color=INK, fontsize=12.5, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(0,3.7); ax.set_xlim(0.6,12.4)
ax.legend(frameon=False, loc='upper right', fontsize=9)
fig.tight_layout(); fig.savefig(OUT+'/sweep_loss.png', facecolor=SURF, bbox_inches='tight'); plt.close(fig)

# ---- Fig 2: top-1 accuracy ----
fig, ax = plt.subplots(figsize=(7.6,4.6), dpi=150); style(ax)
ax.axvline(ESTOP, color=MUTED, lw=1, ls=(0,(4,4)), zorder=1)
ax.text(ESTOP-0.15, 20, 'early stop', color=MUTED, fontsize=9, va='center', ha='right')
ax.plot(ep, D['baseline']['t1'], color=SEC, lw=2, ls=(0,(4,3)), zorder=2, label='baseline (float32)')
for key, col, lab in CFG:
    ax.plot(ep, D[key]['t1'], color=col, lw=2, marker='o', ms=4, zorder=3, label=lab)
ax.set_xlabel('epoch'); ax.set_ylabel('test top-1 accuracy (%)')
ax.set_title('Test top-1 accuracy across loss-scale S  (BITs=13)', color=INK, fontsize=12.5, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(8,44); ax.set_xlim(0.6,12.4)
ax.legend(frameon=False, loc='lower right', fontsize=9)
fig.tight_layout(); fig.savefig(OUT+'/sweep_accuracy.png', facecolor=SURF, bbox_inches='tight'); plt.close(fig)

# ---- Fig 3: degradation (top-1 gap vs baseline) ----
fig, ax = plt.subplots(figsize=(7.6,4.6), dpi=150); style(ax)
ax.axhspan(-0.66, 0.66, color=MUTED, alpha=0.12, zorder=0)
ax.text(12.4, 0.66, ' noise\n band', color=MUTED, fontsize=8.5, va='center', ha='left')
ax.axhline(0, color=AXIS, lw=1, zorder=1)
for key, col, lab in CFG:
    gap = [q - b for q, b in zip(D[key]['t1'], D['baseline']['t1'])]
    ax.plot(ep, gap, color=col, lw=2, marker='o', ms=4, zorder=3, label=lab)
ax.set_xlabel('epoch'); ax.set_ylabel('top-1 gap: quantized − baseline (pts)')
ax.set_title('Quantization degradation across loss-scale S  (BITs=13)', color=INK, fontsize=12.5, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_xlim(0.6,12.9)
ax.legend(frameon=False, loc='lower left', fontsize=9)
fig.tight_layout(); fig.savefig(OUT+'/sweep_degradation.png', facecolor=SURF, bbox_inches='tight'); plt.close(fig)

print('wrote sweep_loss.png, sweep_accuracy.png, sweep_degradation.png')
for key, _, lab in CFG:
    gaps = [q - b for q, b in zip(D[key]['t1'], D['baseline']['t1'])]
    print('%-4s mean gap %+.2f  worst %+.2f  final-ptrunc 2^%.1f' %
          (key, sum(gaps)/len(gaps), min(gaps), D[key]['ptrunc'][-1]))
