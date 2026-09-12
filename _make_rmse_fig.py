import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ep = list(range(1, 13))
# per-epoch param RMSE (%) from the three 12-epoch fidelity runs
rmse_b22 = [0.50,0.83,1.13,1.42,1.71,2.02,2.34,2.69,3.05,3.43,3.80,4.15]   # BITs=22 (§6.2)
rmse_s18 = [0.449,0.742,0.982,1.209,1.438,1.677,1.927,2.190,2.467,2.749,3.024,3.279]  # BITs=13 S=2^18
rmse_s12 = [0.437,0.719,0.952,1.173,1.396,1.628,1.872,2.125,2.393,2.667,2.933,3.182]  # BITs=13 S=2^12

SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; MUTED='#898781'; GRID='#e1e0d9'; AXIS='#c3c2b7'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':11,
    'figure.facecolor':SURF,'axes.facecolor':SURF,'text.color':INK,'axes.labelcolor':SEC,
    'xtick.color':MUTED,'ytick.color':MUTED})

fig, ax = plt.subplots(figsize=(7.4,4.5), dpi=150)
ax.set_facecolor(SURF)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('left','bottom'): ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1)
ax.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0)

ax.plot(ep, rmse_b22, color='#e34948', lw=2, marker='o', ms=4, zorder=3, label='BITs=22 (uniform, §6.2)')
ax.plot(ep, rmse_s18, color='#eb6834', lw=2, marker='o', ms=4, zorder=3, label='BITs=13 + S=2¹⁸')
ax.plot(ep, rmse_s12, color='#2a78d6', lw=2, marker='o', ms=4, zorder=3, label='BITs=13 + S=2¹² (recommended)')

ax.set_xlabel('epoch'); ax.set_ylabel('parameter RMSE vs float32 baseline (%)')
ax.set_title('Weight divergence over training  (does NOT affect accuracy)',
             color=INK, fontsize=12.5, fontweight='bold', loc='left', pad=12)
ax.set_xticks(ep); ax.set_ylim(0,4.5); ax.set_xlim(0.6,12.4)
ax.legend(frameon=False, loc='upper left', fontsize=9.5)
ax.text(12.3, 0.15, 'linear, bounded — no acceleration', color=MUTED, fontsize=9, ha='right')
fig.tight_layout()
OUT='/home/ryan/report_figs'; os.makedirs(OUT, exist_ok=True)
fig.savefig(OUT+'/param_rmse.png', facecolor=SURF, bbox_inches='tight')
print('wrote param_rmse.png; slopes %%/epoch: b22 %.3f  s18 %.3f  s12 %.3f'
      % ((rmse_b22[-1]-rmse_b22[0])/11, (rmse_s18[-1]-rmse_s18[0])/11, (rmse_s12[-1]-rmse_s12[0])/11))
