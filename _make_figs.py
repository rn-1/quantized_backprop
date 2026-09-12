import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# --- data from the 12-epoch CE run (_eval_multiepoch.py trace) ---
ep      = list(range(1, 13))
base_l  = [3.412, 2.838, 2.410, 2.037, 1.687, 1.408, 1.163, 0.894, 0.675, 0.512, 0.335, 0.222]
quant_l = [3.415, 2.847, 2.425, 2.050, 1.704, 1.416, 1.163, 0.884, 0.661, 0.507, 0.360, 0.238]
base_t1 = [17.08, 24.84, 29.88, 33.73, 36.35, 37.48, 38.95, 39.93, 40.88, 40.24, 40.07, 39.84]
quant_t1= [16.90, 24.76, 29.86, 33.79, 36.34, 37.54, 39.38, 40.33, 40.82, 40.50, 40.05, 40.50]
base_t5 = [42.15, 52.92, 59.46, 63.53, 65.96, 67.70, 69.01, 70.11, 70.53, 69.69, 69.46, 69.09]
quant_t5= [42.01, 53.03, 59.64, 63.31, 66.15, 67.76, 69.06, 70.19, 70.56, 69.82, 69.63, 69.65]

# --- palette (dataviz skill, validated categorical slots 1 & 2) ---
BASE  = '#2a78d6'   # blue  -> baseline
QUANT = '#eb6834'   # orange-> quantized
SURF  = '#fcfcfb'
INK   = '#0b0b0b'
SEC   = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'
AXIS  = '#c3c2b7'
ESTOP = 9           # early-stop epoch (best val top-1)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'figure.facecolor': SURF,
    'axes.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': SEC,
    'xtick.color': MUTED, 'ytick.color': MUTED,
})

def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

OUT = '/home/ryan/report_figs'
os.makedirs(OUT, exist_ok=True)

# ============ Figure 1: training loss vs epoch ============
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
style(ax)
ax.axvline(ESTOP, color=MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
ax.text(ESTOP - 0.15, 2.55, 'early stop\n(best val)', color=MUTED, fontsize=9, va='center', ha='right')
ax.plot(ep, base_l,  color=BASE,  lw=2, marker='o', ms=5, zorder=3, label='baseline (float32)')
ax.plot(ep, quant_l, color=QUANT, lw=2, marker='o', ms=5, zorder=3, label='quantized (fp57, BITs=22)')
ax.set_xlabel('epoch'); ax.set_ylabel('training loss (CrossEntropy)')
ax.set_title('Training loss — quantized tracks float32', color=INK, fontsize=13, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(0, 3.7); ax.set_xlim(0.6, 12.4)
ax.legend(frameon=False, loc='upper right', fontsize=10)
ax.annotate('%.2f' % quant_l[-1], (ep[-1], quant_l[-1]), color=QUANT, fontsize=9,
            xytext=(6, -2), textcoords='offset points', va='center')
fig.tight_layout()
fig.savefig(OUT + '/loss_curve.png', facecolor=SURF, bbox_inches='tight')
plt.close(fig)

# ============ Figure 2: test accuracy vs epoch ============
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
style(ax)
ax.axvline(ESTOP, color=MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
ax.text(ESTOP - 0.15, 54, 'early stop\n(best val)', color=MUTED, fontsize=9, va='center', ha='right')
# top-5 band
ax.plot(ep, base_t5,  color=BASE,  lw=2, marker='o', ms=5, zorder=3)
ax.plot(ep, quant_t5, color=QUANT, lw=2, marker='o', ms=5, zorder=3)
# top-1 band
ax.plot(ep, base_t1,  color=BASE,  lw=2, marker='o', ms=5, zorder=3, label='baseline (float32)')
ax.plot(ep, quant_t1, color=QUANT, lw=2, marker='o', ms=5, zorder=3, label='quantized (fp57, BITs=22)')
ax.text(12.3, base_t5[-1] + 1.5, 'Top-5', color=SEC, fontsize=10, fontweight='bold', ha='right')
ax.text(12.3, base_t1[-1] - 3.0, 'Top-1', color=SEC, fontsize=10, fontweight='bold', ha='right')
ax.set_xlabel('epoch'); ax.set_ylabel('test accuracy (%)')
ax.set_title('Test accuracy — quantized matches baseline within noise', color=INK, fontsize=13, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(10, 76); ax.set_xlim(0.6, 12.4)
ax.legend(frameon=False, loc='lower right', fontsize=10)
fig.tight_layout()
fig.savefig(OUT + '/accuracy_curve.png', facecolor=SURF, bbox_inches='tight')
plt.close(fig)

print('wrote', OUT + '/loss_curve.png', 'and', OUT + '/accuracy_curve.png')
