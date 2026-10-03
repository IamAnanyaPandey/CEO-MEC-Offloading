"""Weight-sweep graphs from any E4 CSV file(s).
"""
import argparse, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'Random': '#e74c3c', 'GA': '#2ecc71', 'PSO': '#3498db', 'CSA': '#9b59b6',
       'GA-PSO': '#7f8c8d', 'CEO': '#f39c12'}
MRK = {'Random': 'x', 'GA': 's', 'PSO': 'o', 'CSA': 'D', 'GA-PSO': '^', 'CEO': '*'}
NAME = {'CEO': 'CEO (Proposed)'}
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'savefig.bbox': 'tight', 'savefig.dpi': 300})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.png'), dpi=300)
    plt.close(fig)
    print('saved', os.path.join(out, name + '.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', default=['results/E4.csv'])
    ap.add_argument('--algos', nargs='+', default=['CEO', 'GA-PSO'])
    ap.add_argument('--out', default='results/analysis')
    ap.add_argument('--errorbars', action='store_true', help='add std error bars')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = pd.concat([pd.read_csv(f) for f in a.files], ignore_index=True)
    d = d[d.algorithm.isin(a.algos)].copy()
    if d.empty:
        raise SystemExit(f"No rows for {a.algos} in {a.files}. "
                         f"Available: {sorted(pd.concat([pd.read_csv(f) for f in a.files]).algorithm.unique())}")
    d['tcr'] *= 100
    g = d.groupby(['algorithm', 'w1']).agg(
        delay=('total_delay', 'mean'), delay_sd=('total_delay', 'std'),
        tcr=('tcr', 'mean'), tcr_sd=('tcr', 'std'),
        local=('local_tasks', 'mean'), n=('seed', 'count')).reset_index()
    print(g.round(3).to_string(index=False))
    tag = '_'.join(x.lower().replace('-', '') for x in a.algos)
    g.to_csv(os.path.join(a.out, f'weights_{tag}.csv'), index=False)

    # 1. trade-off: delay vs TCR, labelled by w1
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for alg in a.algos:
        t = g[g.algorithm == alg].sort_values('w1')
        if a.errorbars:
            ax.errorbar(t.delay, t.tcr, xerr=t.delay_sd, yerr=t.tcr_sd, fmt='none',
                        ecolor=COL.get(alg, 'k'), alpha=.35, lw=.8)
        ax.plot(t.delay, t.tcr, marker=MRK.get(alg, 'o'), color=COL.get(alg, 'k'),
                label=NAME.get(alg, alg), lw=1.2, ms=5)
        for _, r in t.iterrows():
            ax.annotate(f'{r.w1:g}', (r.delay, r.tcr), fontsize=6,
                        xytext=(4, 3), textcoords='offset points')
    ax.set_xlabel('Total delay (s)'); ax.set_ylabel('TCR (%)')
    ax.grid(alpha=.3); ax.legend()
    save(fig, a.out, f'weights_{tag}_tradeoff')

    # 2. delay and TCR against w1
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for alg in a.algos:
        t = g[g.algorithm == alg].sort_values('w1')
        for ax, col, sd, yl in [(axs[0], 'delay', 'delay_sd', 'Total delay (s)'),
                                (axs[1], 'tcr', 'tcr_sd', 'TCR (%)')]:
            if a.errorbars:
                ax.errorbar(t.w1, t[col], yerr=t[sd], color=COL.get(alg, 'k'),
                            marker=MRK.get(alg, 'o'), capsize=2, lw=1.2, ms=5,
                            label=NAME.get(alg, alg))
            else:
                ax.plot(t.w1, t[col], marker=MRK.get(alg, 'o'), color=COL.get(alg, 'k'),
                        lw=1.2, ms=5, label=NAME.get(alg, alg))
            ax.set_xlabel('$w_1$ (weight of the delay term)'); ax.set_ylabel(yl)
            ax.grid(alpha=.3); ax.set_xticks(sorted(t.w1.unique()))
    axs[0].legend()
    save(fig, a.out, f'weights_{tag}_lines')

    # 3. tasks kept local
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for alg in a.algos:
        t = g[g.algorithm == alg].sort_values('w1')
        ax.plot(t.w1, t.local, marker=MRK.get(alg, 'o'), color=COL.get(alg, 'k'),
                lw=1.2, ms=5, label=NAME.get(alg, alg))
    ax.set_xlabel('$w_1$ (weight of the delay term)')
    ax.set_ylabel('Tasks executed locally (of 200)')
    ax.grid(alpha=.3); ax.legend(); ax.set_xticks(sorted(g.w1.unique()))
    save(fig, a.out, f'weights_{tag}_local')


if __name__ == '__main__':
    main()