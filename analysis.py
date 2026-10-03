
import argparse, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ALGOS = ['Random', 'GA', 'PSO', 'CSA', 'GA-PSO', 'CEO']
BASE = [a for a in ALGOS if a != 'CEO']
COL = {'Random': '#e74c3c', 'GA': '#2ecc71', 'PSO': '#3498db', 'CSA': '#9b59b6',
       'GA-PSO': '#7f8c8d', 'CEO': '#f39c12'}
# ---- Name shown in figures and tables. Change it here only (CSV files keep 'CEO').
PROPOSED_LABEL = 'CEO (Proposed)'
NAME = lambda a: PROPOSED_LABEL if a == 'CEO' else a
MRK = {'Random': 'x', 'GA': 's', 'PSO': 'o', 'CSA': 'D', 'GA-PSO': '^', 'CEO': '*'}
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'savefig.bbox': 'tight', 'savefig.dpi': 300})


def save(fig, out, name):
    """Save every figure as PNG (300 dpi)."""
    fig.savefig(os.path.join(out, f'{name}.png'), dpi=300)
    plt.close(fig)


def grouped_bars(ax, labels, algos, mean, sd=None, colors=COL):
    x = np.arange(len(labels)); w = 0.8 / len(algos)
    for k, alg in enumerate(algos):
        ax.bar(x + (k - (len(algos) - 1) / 2) * w, mean[alg], w, yerr=None if sd is None else sd[alg],
               color=colors.get(alg, '#999999'), label=NAME(alg), capsize=2, error_kw={'lw': 0.6},
               edgecolor='white', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis='y', alpha=.3)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method='zsplit').pvalue)


def pair(df, cfg, alg, metric):
    return df[(df.config == cfg) & (df.algorithm == alg)].sort_values('seed')[metric].values


def fmt_p(p):
    return '$<$0.001' if p < 1e-3 else f'{p:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results'); ap.add_argument('--out', default=None)
    a = ap.parse_args(); out = a.out or os.path.join(a.res, 'analysis'); os.makedirs(out, exist_ok=True)
    S = open(os.path.join(out, 'summary.txt'), 'w')
    say = lambda *x: (print(*x), print(*x, file=S))
    load = lambda e: pd.read_csv(os.path.join(a.res, f'{e}.csv')) if os.path.exists(os.path.join(a.res, f'{e}.csv')) else None

    # ------------------------------------------------------------------ E1
    e1 = load('E1')
    if e1 is not None:
        e1['tcr'] *= 100
        runs = e1[e1.exp == 'E1']; mem = e1[e1.exp == 'E1mem']
        say('=== E1: main comparison (mean +- std over instances) ===')
        rows = []
        for cfg in sorted(runs.config.unique()):
            m = int(runs[runs.config == cfg].n_tasks.iloc[0])
            pd_, pt_ = [], []
            for b in BASE:
                pd_.append(paired_p(pair(runs, cfg, 'CEO', 'total_delay'), pair(runs, cfg, b, 'total_delay')))
                pt_.append(paired_p(pair(runs, cfg, 'CEO', 'tcr'), pair(runs, cfg, b, 'tcr')))
            pd_, pt_ = dict(zip(BASE, holm(pd_))), dict(zip(BASE, holm(pt_)))
            for alg in ALGOS:
                g = runs[(runs.config == cfg) & (runs.algorithm == alg)]
                mm = mem[(mem.config == cfg) & (mem.algorithm == alg)].peak_mem_MB
                r = dict(config=cfg, m=m, algorithm=alg, n=len(g),
                         delay=g.total_delay.mean(), delay_sd=g.total_delay.std(),
                         tcr=g.tcr.mean(), tcr_sd=g.tcr.std(),
                         time=g.runtime_s.mean(), time_sd=g.runtime_s.std(), nfe=g.nfe.mean(),
                         local=g.local_tasks.mean(), mem=mm.mean() if len(mm) else np.nan,
                         p_delay=pd_.get(alg, np.nan), p_tcr=pt_.get(alg, np.nan))
                rows.append(r)
        T = pd.DataFrame(rows); T.to_csv(os.path.join(out, 'E1_summary.csv'), index=False)
        say(T.round(4).to_string(index=False))

        # improvements of CEO over best baseline (by mean delay)
        say('\nCEO vs best baseline (mean delay):')
        for cfg in T.config.unique():
            t = T[T.config == cfg].set_index('algorithm')
            bb = t.loc[BASE, 'delay'].idxmin()
            say(f'  {cfg}: best baseline {bb} {t.loc[bb,"delay"]:.2f}s, CEO {t.loc["CEO","delay"]:.2f}s, '
                f'reduction {(1 - t.loc["CEO","delay"]/t.loc[bb,"delay"])*100:.1f}% ; '
                f'TCR CEO {t.loc["CEO","tcr"]:.1f} vs {bb} {t.loc[bb,"tcr"]:.1f}; '
                f'all p_delay<0.05: {bool((t.loc[BASE,"p_delay"]<0.05).all())}, all p_tcr<0.05: {bool((t.loc[BASE,"p_tcr"]<0.05).all())}')

        # per-server distribution of CEO
        say('\nCEO task distribution (mean over instances): local, per-server counts, Jain index of offloaded counts')
        for cfg in sorted(runs.config.unique()):
            g = runs[(runs.config == cfg) & (runs.algorithm == 'CEO')]
            c = np.array([[int(v) for v in s.split(';')] for s in g.server_counts])
            jain = (c.sum(1) ** 2) / (c.shape[1] * (c ** 2).sum(1))
            say(f'  {cfg}: local {g.local_tasks.mean():.1f}, per-server {np.round(c.mean(0),1)}, '
                f'min {c.min()}, max {c.max()}, Jain {jain.mean():.3f}')

        # LaTeX table

        # ---------- Wilcoxon table: CEO vs each baseline, per scenario and metric
        # sign: '+' CEO significantly better, '-' significantly worse, '=' no significant difference
        cfgs_w = list(T.config.unique()); W = []
        for b in BASE:
            r = {'baseline': b}
            for c in cfgs_w:
                t = T[T.config == c].set_index('algorithm')
                for met, pcol, better in [('delay', 'p_delay', -1), ('tcr', 'p_tcr', +1)]:
                    p = t.loc[b, pcol]; diff = t.loc['CEO', met] - t.loc[b, met]
                    sign = '=' if p >= 0.05 else ('+' if np.sign(diff) == better else '-')
                    r[f'{c}_{met}_p'] = p; r[f'{c}_{met}_sign'] = sign
            W.append(r)
        W = pd.DataFrame(W); W.to_csv(os.path.join(out, 'wilcoxon.csv'), index=False)
        say('\n=== Wilcoxon signed-rank test, CEO vs baseline (Holm-adjusted p; + CEO better, - worse, = n.s.) ===')
        for _, r in W.iterrows():
            say('  ' + f'{r.baseline:7s} ' + '  '.join(
                f'{c}: delay {fmt_p(r[f"{c}_delay_p"]).replace("$<$", "<")} ({r[f"{c}_delay_sign"]}), '
                f'TCR {fmt_p(r[f"{c}_tcr_p"]).replace("$<$", "<")} ({r[f"{c}_tcr_sign"]})' for c in cfgs_w))
        wins = {k: sum((W[f'{c}_{m}_sign'] == k).sum() for c in cfgs_w for m in ['delay', 'tcr']) for k in '+=-'}
        say(f'  totals over {len(BASE)*len(cfgs_w)*2} comparisons: +{wins["+"]} / ={wins["="]} / -{wins["-"]}')

        # PNG version of the Wilcoxon table (for slides / quick viewing)
        fig, ax = plt.subplots(figsize=(7.0, 0.35 * (2 * len(BASE) + 2))); ax.axis('off')
        cell, colors = [], []
        for _, r in W.iterrows():
            for met, mlab in [('delay', 'Delay'), ('tcr', 'TCR')]:
                row = [r.baseline if met == 'delay' else '', mlab]; rc = ['w', 'w']
                for c in cfgs_w:
                    p, sg = r[f'{c}_{met}_p'], r[f'{c}_{met}_sign']
                    row.append(('<0.001' if p < 1e-3 else f'{p:.3f}') + f'  ({sg})')
                    rc.append({'+': '#d5f5e3', '=': '#fdebd0', '-': '#fadbd8'}[sg])
                cell.append(row); colors.append(rc)
        tb = ax.table(cellText=cell, cellColours=colors, colLabels=['Baseline', 'Metric'] + cfgs_w,
                      loc='center', cellLoc='center')
        tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.25)
        ax.set_title(f'Wilcoxon signed-rank test, {NAME("CEO")} vs baselines (Holm-adjusted p; '
                     f'+ better, = n.s., - worse)', fontsize=8)
        save(fig, out, 'fig_wilcoxon')

        cfgs = list(T.config.unique())
        lab = [f'{c}\n({int(T[T.config == c].m.iloc[0])} tasks)' for c in cfgs]
        get = lambda col: {alg: [T[(T.config == c) & (T.algorithm == alg)][col].iloc[0] for c in cfgs] for alg in ALGOS}
        for col, sdc, yl, name in [('delay', 'delay_sd', 'Total delay (s)', 'fig_delay'),
                                   ('tcr', 'tcr_sd', 'TCR (%)', 'fig_tcr'),
                                   ('time', 'time_sd', 'Runtime per run (s)', 'fig_runtime')]:
            fig, ax = plt.subplots(figsize=(3.5, 2.4))
            grouped_bars(ax, lab, ALGOS, get(col), get(sdc)); ax.set_ylabel(yl)
            ax.legend(ncol=3, fontsize=6, loc='upper left' if col != 'tcr' else 'upper right')
            if col == 'tcr':
                ax.set_ylim(0, 100)
            else:                       # head-room so the legend does not cover the bars
                top = max(np.nanmax(np.array(get(col)[a_]) + np.array(get(sdc)[a_])) for a_ in ALGOS)
                ax.set_ylim(0, top * 1.35)
            fig.tight_layout(); save(fig, out, name)

        # tasks per server (CEO), stacked: local + each server
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        bottom = np.zeros(len(cfgs)); nsv = None
        loc = [runs[(runs.config == c) & (runs.algorithm == 'CEO')].local_tasks.mean() for c in cfgs]
        ax.bar(lab, loc, color='#bdc3c7', label='Local'); bottom += loc
        per = []
        for c in cfgs:
            g = runs[(runs.config == c) & (runs.algorithm == 'CEO')]
            per.append(np.array([[int(v) for v in q.split(';')] for q in g.server_counts]).mean(0))
        per = np.array(per)
        cm = plt.cm.viridis(np.linspace(0.15, 0.9, per.shape[1]))
        for j in range(per.shape[1]):
            ax.bar(lab, per[:, j], bottom=bottom, color=cm[j], label=f'Server {j+1}'); bottom += per[:, j]
        ax.set_ylabel(f'Tasks ({NAME("CEO")}, mean)'); ax.legend(fontsize=6, ncol=2); ax.grid(axis='y', alpha=.3)
        fig.tight_layout(); save(fig, out, 'fig_distribution')

        # convergence vs NFE
        cp = os.path.join(a.res, 'E1_convergence.pkl')
        if os.path.exists(cp):
            conv = pickle.load(open(cp, 'rb'))
            cfgs = sorted(runs.config.unique())
            fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4)); axs = axs.ravel()
            for ax, cfg in zip(axs, cfgs):
                budget = int(runs[runs.config == cfg].budget.iloc[0])
                grid = np.linspace(0.02 * budget, budget, 200)
                for alg in ALGOS:
                    if (cfg, alg) not in conv:
                        continue
                    curves = []
                    for nfe, fit in conv[(cfg, alg)]:
                        idx = np.searchsorted(nfe, grid, side='right') - 1
                        c = np.where(idx >= 0, fit[np.clip(idx, 0, None)], np.nan)
                        curves.append(c)
                    mu = np.nanmean(np.array(curves), 0)
                    ax.plot(grid, mu, color=COL[alg], label=NAME(alg), lw=1.2, marker=MRK[alg], markevery=25, ms=3.5)
                m = int(runs[runs.config == cfg].n_tasks.iloc[0])
                ax.set_title(f'{cfg} ({m} tasks)'); ax.set_xlabel('Function evaluations'); ax.set_ylabel('Best fitness')
                ax.grid(alpha=.3)
            axs[0].legend(ncol=3, loc='upper right')
            fig.tight_layout(); save(fig, out, 'fig_convergence')

    # ------------------------------------------------------------------ E2 ablation
    e2 = load('E2')
    if e2 is not None:
        e2['tcr'] *= 100
        say('\n=== E2: ablation ===')
        order = list(dict.fromkeys(e2.algorithm))
        pref = ['CEO (full)', 'w/o greedy init', 'w/o greedy+OBL', 'w/o memory adh.', 'w/o Levy',
                'w/o elite LS', 'w/o adaptive AP', 'w/o restart', 'CSA + OBL only', 'Discrete CSA']
        order = [v for v in pref if v in order]
        rows = []
        for v in order:
            r = {'variant': v}
            for cfg in sorted(e2.config.unique()):
                g = e2[(e2.config == cfg) & (e2.algorithm == v)]
                r[f'{cfg}_delay'] = g.total_delay.mean(); r[f'{cfg}_tcr'] = g.tcr.mean()
                r[f'{cfg}_fit'] = g.fitness.mean()
                r[f'{cfg}_p'] = np.nan if v == 'CEO (full)' else paired_p(pair(e2, cfg, 'CEO (full)', 'fitness'), pair(e2, cfg, v, 'fitness'))
            rows.append(r)
        A = pd.DataFrame(rows)
        for cfg in sorted(e2.config.unique()):
            A[f'{cfg}_p'] = np.r_[np.nan, holm(A[f'{cfg}_p'].values[1:])]
        A.to_csv(os.path.join(out, 'E2_summary.csv'), index=False); say(A.round(4).to_string(index=False))
        cfgs = sorted(e2.config.unique())

        fig, axs = plt.subplots(len(cfgs), 2, figsize=(7.0, 2.3 * len(cfgs)), squeeze=False)
        yy = np.arange(len(A))
        for r_, c in enumerate(cfgs):
            for k_, (key, xl) in enumerate([('delay', 'Total delay (s)'), ('tcr', 'TCR (%)')]):
                ax = axs[r_, k_]
                vals = A[f'{c}_{key}'].values
                ax.barh(yy, vals, color=['#f39c12'] + ['#7f8c8d'] * (len(A) - 1))
                ax.set_yticks(yy); ax.set_yticklabels(A.variant if k_ == 0 else [])
                ax.invert_yaxis(); ax.grid(axis='x', alpha=.3)
                span = vals.max() - vals.min()
                ax.set_xlim(max(0, vals.min() - 0.15 * span), vals.max() + 0.1 * span)
                ax.axvline(vals[0], color='#f39c12', ls='--', lw=0.8)
                ax.set_xlabel(xl); ax.set_title(f'{c}', fontsize=8)
        fig.tight_layout(); save(fig, out, 'fig_ablation')

    # ------------------------------------------------------------------ E3 scalability
    e3 = load('E3')
    if e3 is not None:
        e3['tcr'] *= 100
        say('\n=== E3: scalability ===')
        cfgs = list(dict.fromkeys(e3.sort_values(['n_servers', 'n_tasks']).config))
        rows = []
        for cfg in cfgs:
            ps = holm([paired_p(pair(e3, cfg, 'CEO', 'total_delay'), pair(e3, cfg, b, 'total_delay')) for b in BASE])
            for alg in ALGOS:
                g = e3[(e3.config == cfg) & (e3.algorithm == alg)]
                rows.append(dict(config=cfg, algorithm=alg, n=len(g), budget=g.budget.iloc[0],
                                 delay=g.total_delay.mean(), tcr=g.tcr.mean(), time=g.runtime_s.mean(),
                                 local=g.local_tasks.mean(),
                                 p=np.nan if alg == 'CEO' else dict(zip(BASE, ps))[alg]))
        T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(out, 'E3_summary.csv'), index=False)
        say(T3.round(3).to_string(index=False))
        show = ['GA', 'PSO', 'GA-PSO', 'CEO']

        algs3 = [x for x in ALGOS if x in set(T3.algorithm)]
        cols = ['n_tasks', 'n_servers', 'algorithm', 'total_delay', 'tcr']
        allr = e3[cols]                          # e3 tcr is already in per cent
        if e1 is not None:                       # add the E1 points (50-200 tasks, 5 servers)
            r1 = e1[e1.exp == 'E1'][cols]        # e1 tcr is already in per cent as well
            allr = pd.concat([allr, r1[r1.algorithm.isin(algs3)]])
        fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4))
        panels = [(allr.n_servers == 5, 'n_tasks', 'Number of tasks (5 servers)'),
                  (allr.n_tasks == 200, 'n_servers', 'Number of servers (200 tasks)')]
        for col, (sel, xcol, xl) in enumerate(panels):
            sub = allr[sel]
            for row, (metric, yl) in enumerate([('total_delay', 'Total delay (s)'),
                                                ('tcr', 'TCR (%)')]):
                ax = axs[row, col]
                for alg in algs3:
                    g = sub[sub.algorithm == alg].groupby(xcol)[metric].mean()
                    ax.plot(g.index, g.values, marker=MRK[alg], color=COL[alg],
                            label=NAME(alg), lw=1.2, ms=4)
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
                ticks = sorted(sub[xcol].unique())      # only the values actually simulated
                ax.set_xticks(ticks); ax.set_xticklabels([str(int(t)) for t in ticks])
        axs[0, 0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); save(fig, out, 'fig_scalability')

    # ------------------------------------------------------------------ E4 weights
    e4 = load('E4')
    if e4 is not None:
        e4['tcr'] *= 100
        say('\n=== E4: weight sweep (S4) ===')
        W = e4.groupby(['w1', 'algorithm'])[['total_delay', 'tcr', 'local_tasks']].mean().reset_index()
        say(W.round(3).to_string(index=False)); W.to_csv(os.path.join(out, 'E4_summary.csv'), index=False)
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        for alg in W.algorithm.unique():
            w = W[W.algorithm == alg].sort_values('w1')
            ax.plot(w.total_delay, w.tcr, marker=MRK.get(alg, 'o'), color=COL.get(alg, 'k'), label=NAME(alg), lw=1)
            if alg == 'CEO':
                for _, r in w.iterrows():
                    ax.annotate(f'{r.w1:g}', (r.total_delay, r.tcr), fontsize=6, xytext=(3, 2), textcoords='offset points')
        ax.set_xlabel('Total delay (s)'); ax.set_ylabel('TCR (%)'); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); save(fig, out, 'fig_weights')

    # ------------------------------------------------------------------ E5 tuning
    e5 = load('E5')
    if e5 is not None:
        say('\n=== E5: CEO parameter grid on tuning instances (mean fitness, lower is better) ===')
        G = e5.groupby('algorithm').fitness.agg(['mean', 'std']).sort_values('mean')
        say(G.round(5).to_string())
        d = 'gf=0.2,ef=0.3,Ps=8'
        if d in G.index:
            say(f'default {d}: rank {list(G.index).index(d)+1} of {len(G)}, '
                f'gap to best {(G.loc[d,"mean"]/G["mean"].iloc[0]-1)*100:.2f}%')
        G.to_csv(os.path.join(out, 'E5_summary.csv'))
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        cols = ['#f39c12' if i == 0 else ('#e74c3c' if n == d else '#95a5a6') for i, n in enumerate(G.index)]
        ax.bar(range(len(G)), G['mean'], color=cols)
        ax.set_xlabel('Parameter configuration (sorted)'); ax.set_ylabel('Mean fitness')
        ax.set_ylim(G['mean'].min() * 0.98, G['mean'].max() * 1.01); ax.grid(axis='y', alpha=.3)
        ax.set_title(f'orange: best ({G.index[0]}),  red: submitted default', fontsize=7)
        fig.tight_layout(); save(fig, out, 'fig_tuning')
    S.close()


if __name__ == '__main__':
    main()