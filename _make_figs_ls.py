import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# --- BITs=13 + S=2^12, 12-epoch trace (_multiepoch_lossscale.py) ---
ep       = list(range(1, 13))
base_l   = [3.412,2.838,2.410,2.037,1.687,1.408,1.163,0.894,0.675,0.512,0.335,0.222]
quant_l  = [3.414,2.836,2.411,2.043,1.708,1.421,1.166,0.881,0.671,0.502,0.346,0.223]
base_t1  = [17.08,24.84,29.88,33.73,36.35,37.48,38.95,39.93,40.88,40.24,40.07,39.84]
quant_t1 = [17.07,24.78,29.84,33.85,36.33,37.41,39.13,40.11,40.68,40.07,39.76,40.07]
base_t5  = [42.15,52.92,59.46,63.53,65.96,67.70,69.01,70.11,70.53,69.69,69.46,69.09]
quant_t5 = [41.98,52.88,59.57,63.33,65.92,67.56,69.01,70.03,70.18,69.38,69.15,69.40]
gap_t1   = [q - b for q, b in zip(quant_t1, base_t1)]

BASE='#2a78d6'; QUANT='#eb6834'; SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'
MUTED='#898781'; GRID='#e1e0d9'; AXIS='#c3c2b7'; GOOD='#0ca30c'; ESTOP=9

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':11,
    'figure.facecolor':SURF,'axes.facecolor':SURF,'text.color':INK,'axes.labelcolor':SEC,
    'xtick.color':MUTED,'ytick.color':MUTED})

def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)

OUT='/home/ryan/report_figs'; os.makedirs(OUT, exist_ok=True)
QLABEL='quantized (fp57, BITs=13, S=2¹²)'

# Fig 1: loss
fig, ax = plt.subplots(figsize=(7.2,4.4), dpi=150); style(ax)
ax.axvline(ESTOP, color=MUTED, lw=1, ls=(0,(4,4)), zorder=1)
ax.text(ESTOP-0.15, 2.55, 'early stop\n(best val)', color=MUTED, fontsize=9, va='center', ha='right')
ax.plot(ep, base_l, color=BASE, lw=2, marker='o', ms=5, zorder=3, label='baseline (float32)')
ax.plot(ep, quant_l, color=QUANT, lw=2, marker='o', ms=5, zorder=3, label=QLABEL)
ax.set_xlabel('epoch'); ax.set_ylabel('training loss (CrossEntropy)')
ax.set_title('Training loss — loss-scaled 13-bit fixed point tracks float32',
             color=INK, fontsize=12.5, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(0,3.7); ax.set_xlim(0.6,12.4)
ax.legend(frameon=False, loc='upper right', fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT+'/loss_curve_ls.png', facecolor=SURF, bbox_inches='tight'); plt.close(fig)

# Fig 2: accuracy (top-1 + top-5)
fig, ax = plt.subplots(figsize=(7.2,4.4), dpi=150); style(ax)
ax.axvline(ESTOP, color=MUTED, lw=1, ls=(0,(4,4)), zorder=1)
ax.text(ESTOP-0.15, 54, 'early stop\n(best val)', color=MUTED, fontsize=9, va='center', ha='right')
for series, col in [(base_t5,BASE),(quant_t5,QUANT),(base_t1,BASE),(quant_t1,QUANT)]:
    ax.plot(ep, series, color=col, lw=2, marker='o', ms=5, zorder=3)
ax.plot([],[],color=BASE,lw=2,marker='o',ms=5,label='baseline (float32)')
ax.plot([],[],color=QUANT,lw=2,marker='o',ms=5,label=QLABEL)
ax.text(12.3, base_t5[-1]+1.4, 'Top-5', color=SEC, fontsize=10, fontweight='bold', ha='right')
ax.text(12.3, base_t1[-1]-3.0, 'Top-1', color=SEC, fontsize=10, fontweight='bold', ha='right')
ax.set_xlabel('epoch'); ax.set_ylabel('test accuracy (%)')
ax.set_title('Test accuracy — quantized matches baseline within noise',
             color=INK, fontsize=12.5, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(10,76); ax.set_xlim(0.6,12.4)
ax.legend(frameon=False, loc='lower right', fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT+'/accuracy_curve_ls.png', facecolor=SURF, bbox_inches='tight'); plt.close(fig)

# Fig 3: degradation — top-1 gap (quant - baseline) per epoch, with run-to-run noise band
fig, ax = plt.subplots(figsize=(7.2,4.0), dpi=150); style(ax)
ax.axhspan(-0.66, 0.66, color=MUTED, alpha=0.12, zorder=0)   # observed run-to-run noise band (§6.2)
ax.text(12.4, 0.66, ' run-to-run\n noise band', color=MUTED, fontsize=8.5, va='center', ha='left')
ax.axhline(0, color=AXIS, lw=1, zorder=1)
colors = [GOOD if g >= 0 else QUANT for g in gap_t1]
ax.bar(ep, gap_t1, color=colors, width=0.62, zorder=3)
ax.set_xlabel('epoch'); ax.set_ylabel('top-1 gap: quantized − baseline (pts)')
ax.set_title('Quantization degradation — top-1 gap stays within noise, no trend',
             color=INK, fontsize=12.5, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(-0.8,0.8); ax.set_xlim(0.4,13.3)
mean_gap = sum(gap_t1)/len(gap_t1)
ax.text(0.7, -0.74, 'mean %+.2f pts   worst %+.2f pts   (both ≪ noise band ±0.66)'
        % (mean_gap, min(gap_t1, key=abs) if False else min(gap_t1)),
        color=SEC, fontsize=9, ha='left')
fig.tight_layout(); fig.savefig(OUT+'/degradation_ls.png', facecolor=SURF, bbox_inches='tight'); plt.close(fig)

print('wrote loss_curve_ls.png, accuracy_curve_ls.png, degradation_ls.png')
print('mean top-1 gap %+.3f pts; max |gap| %.2f pts' % (mean_gap, max(abs(g) for g in gap_t1)))
