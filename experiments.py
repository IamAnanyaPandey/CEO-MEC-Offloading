import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
import os
import io
from contextlib import redirect_stdout

from system_model import SimulationConfig, MECEnvironment
from algorithms import (
    RandomOffloading, GeneticAlgorithm,
    ParticleSwarmOptimization, CrowSearchAlgorithm
)
from ceo import CEO

warnings.filterwarnings('ignore')

# Plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

COLORS = {
    'RandomOffloading':          '#e74c3c',
    'GeneticAlgorithm':          '#2ecc71',
    'ParticleSwarmOptimization': '#3498db',
    'CrowSearchAlgorithm':       '#9b59b6',
    'CEO':                       '#f39c12'
}

LABELS = {
    'RandomOffloading':          'Random',
    'GeneticAlgorithm':          'GA',
    'ParticleSwarmOptimization': 'PSO',
    'CrowSearchAlgorithm':       'CSA',
    'CEO':                       'CEO'
}

MARKERS = {
    'RandomOffloading':          'X',
    'GeneticAlgorithm':          's',
    'ParticleSwarmOptimization': 'o',
    'CrowSearchAlgorithm':       'D',
    'CEO':                       '*'
}

# Fixed parameters

TASK_COUNTS   = [50, 100, 150, 200]
FIXED_SERVERS = 5     
POP_SIZE      = 50
MAX_ITER      = 100
W1            = 0.5   # weight for delay
W2            = 0.5   # weight for tcr

ALGO_KEYS = [
    'RandomOffloading',
    'GeneticAlgorithm',
    'ParticleSwarmOptimization',
    'CrowSearchAlgorithm',
    'CEO'
]


CONVERGENCE_ALGO_KEYS = [
    'GeneticAlgorithm',
    'ParticleSwarmOptimization',
    'CrowSearchAlgorithm',
    'CEO'
]

N_RUNS     = 30
CONV_SEEDS = list(range(1, N_RUNS + 1))   # seeds 1..30

# Scenario labels for task experiments
TASK_SCENARIO_LABELS = [
    'S1\n(50 Tasks)',
    'S2\n(100 Tasks)',
    'S3\n(150 Tasks)',
    'S4\n(200 Tasks)'
]


# Shared algorithm factory (same hyperparameters used everywhere, only `seed` varies)
def _build_algorithms(env, seed, max_iter=MAX_ITER, pop_size=POP_SIZE):
   
    return {
        'RandomOffloading': RandomOffloading(
            env, num_trials=pop_size * max_iter, seed=seed
        ),
        'GeneticAlgorithm': GeneticAlgorithm(
            env, max_iterations=max_iter, population_size=pop_size,
            crossover_rate=0.9, mutation_rate=0.05, tournament_size=3,
            elite_count=2, seed=seed
        ),
        'ParticleSwarmOptimization': ParticleSwarmOptimization(
            env, max_iterations=max_iter, population_size=pop_size,
            w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, seed=seed
        ),
        'CrowSearchAlgorithm': CrowSearchAlgorithm(
            env, max_iterations=max_iter, population_size=pop_size,
            awareness_prob=0.1, flight_length=2.0, seed=seed
        ),
        'CEO': CEO(
            env, max_iterations=max_iter, population_size=pop_size,
            awareness_prob=0.1, flight_length=2.0,
            levy_beta=1.5, greedy_fraction=0.2, elite_fraction=0.3,
            stagnation_patience=8, restart_fraction=0.3, seed=seed
        ),
    }


def run_single_experiment(env, max_iter=MAX_ITER, pop_size=POP_SIZE, seed=50):
    results = {}
    algos = _build_algorithms(env, seed, max_iter, pop_size)
    # dict preserves insertion order -> identical execution order to before:
    # Random, GA, PSO, CSA, CEO
    for name, algo in algos.items():
        results[name] = algo.optimize()
    return results


class _SuppressStdout:
   
    def __enter__(self):
        self._buf = io.StringIO()
        self._ctx = redirect_stdout(self._buf)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False


def collect_convergence_statistics(env, seeds=CONV_SEEDS,
                                    max_iter=MAX_ITER, pop_size=POP_SIZE,
                                    verbose_every=10):
   
    metrics = {
        algo: {'fitness': [], 'total_delay': [], 'tcr': [], 'execution_time': []}
        for algo in ALGO_KEYS
    }
    histories = {k: [] for k in CONVERGENCE_ALGO_KEYS}

    n_seeds = len(seeds)
    for run_idx, seed in enumerate(seeds):
        with _SuppressStdout():
            algos = _build_algorithms(env, seed, max_iter, pop_size)
            for name, algo in algos.items():
                res = algo.optimize()
                metrics[name]['fitness'].append(res['fitness'])
                metrics[name]['total_delay'].append(res['total_delay'])
                metrics[name]['tcr'].append(res['tcr'])
                metrics[name]['execution_time'].append(res['execution_time'])
                if name in histories:
                    histories[name].append(res['convergence_history'])

        if (run_idx + 1) % verbose_every == 0 or (run_idx + 1) == n_seeds:
            print(f"    [{run_idx + 1:2d}/{n_seeds}] independent runs completed")

    stats = {}
    for algo in ALGO_KEYS:
        m          = metrics[algo]
        fitness_arr = np.array(m['fitness'])
        delay_arr   = np.array(m['total_delay'])
        tcr_arr     = np.array(m['tcr'])          # fraction, 0-1
        time_arr    = np.array(m['execution_time'])

        stats[algo] = {
            'fitness_mean': float(fitness_arr.mean()),
            'fitness_std':  float(fitness_arr.std()),
            'delay_mean':   float(delay_arr.mean()),
            'delay_std':    float(delay_arr.std()),
            'tcr_mean':     float(tcr_arr.mean() * 100.0),   # percent
            'tcr_std':      float(tcr_arr.std() * 100.0),    # percent
            'time_mean':    float(time_arr.mean()),
            'time_std':     float(time_arr.std()),
        }

        if algo in histories:
            arr = np.array(histories[algo])  # shape (n_seeds, max_iter)
            stats[algo]['conv_mean'] = arr.mean(axis=0)
            stats[algo]['conv_std']  = arr.std(axis=0)
            stats[algo]['conv_runs'] = arr

    return stats


def _plot_grouped_bar(data, algo_keys, scenario_labels, ylabel, title,
                      savepath, value_fmt='{:.1f}', ylim=None, err=None):
    n_algos     = len(algo_keys)
    n_scenarios = len(scenario_labels)

    bar_w = 0.15
    x     = np.arange(n_scenarios)

    fig, ax = plt.subplots(figsize=(13, 6))

    for a_idx, algo in enumerate(algo_keys):
        offsets  = x + (a_idx - (n_algos - 1) / 2) * bar_w
        vals     = data[algo]
        errs     = err[algo] if err is not None else None
        bars     = ax.bar(
            offsets, vals,
            width=bar_w,
            color=COLORS[algo],
            label=LABELS[algo],
            edgecolor='white',
            linewidth=0.4,
            alpha=0.9,
            yerr=errs,
            capsize=3,
            error_kw={'linewidth': 1, 'ecolor': '#333333', 'alpha': 0.7}
        )
        for b_idx, (bar, val) in enumerate(zip(bars, vals)):
            err_h = errs[b_idx] if errs is not None else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + err_h + (0.3 if 'TCR' in ylabel else 1.0),
                value_fmt.format(val),
                ha='center', va='bottom', fontsize=7.5, rotation=90
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.set_xlabel('Scenario')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title='Algorithm', loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def _print_table(stats, label):
    print(f"\n{'='*95}")
    print(f"  SCENARIO: {label}   (mean \u00b1 std over {N_RUNS} independent runs)")
    print(f"{'='*95}")
    print(f"  {'Algorithm':<10} | {'Fitness':>16} | "
          f"{'Total Delay (s)':>18} | {'TCR (%)':>14} | {'Time (s)':>12}")
    print(f"  {'-'*88}")
    for algo in ALGO_KEYS:
        r = stats[algo]
        print(f"  {LABELS[algo]:<10} | "
              f"{r['fitness_mean']:>8.4f} \u00b1 {r['fitness_std']:<5.4f} | "
              f"{r['delay_mean']:>9.2f} \u00b1 {r['delay_std']:<6.2f} | "
              f"{r['tcr_mean']:>6.1f} \u00b1 {r['tcr_std']:<5.1f} | "
              f"{r['time_mean']:>6.2f} \u00b1 {r['time_std']:<3.2f}")

    ceo_delay = stats['CEO']['delay_mean']
    csa_delay = stats['CrowSearchAlgorithm']['delay_mean']
    ceo_tcr   = stats['CEO']['tcr_mean']
    csa_tcr   = stats['CrowSearchAlgorithm']['tcr_mean']

    delay_imp = (csa_delay - ceo_delay) / csa_delay * 100
    tcr_imp   = (ceo_tcr - csa_tcr) / csa_tcr * 100 if csa_tcr > 0 else float('inf')

    print(f"\n  CEO vs CSA → Mean delay reduced by {delay_imp:.2f}%  |  "
          f"Mean TCR improved by {tcr_imp:.2f}%")
    print(f"{'='*95}")


# EXPERIMENT: Fixed servers = 5, Tasks: 50 / 100 / 150 / 200

def experiment_task_scalability(output_dir):
    print("\n" + "=" * 70)
    print("  TASK SCALABILITY EXPERIMENT")
    print("  Fixed Servers = 5  |  Tasks = 50, 100, 150, 200")
    print(f"  Weights: w1={W1}, w2={W2}")
    print("=" * 70)

    all_results   = {}   # per scenario -> full 30-run stats dict (all 5 algos, all metrics)
    delay_data    = {algo: [] for algo in ALGO_KEYS}
    delay_err     = {algo: [] for algo in ALGO_KEYS}
    tcr_data      = {algo: [] for algo in ALGO_KEYS}
    tcr_err       = {algo: [] for algo in ALGO_KEYS}

    for idx, n_tasks in enumerate(TASK_COUNTS):
        print(f"\n  Running S{idx+1}: {n_tasks} tasks, {FIXED_SERVERS} servers ...")
        config  = SimulationConfig(
            num_tasks=n_tasks, num_edge_servers=FIXED_SERVERS,
            w1=W1, w2=W2, seed=50
        )
        env     = MECEnvironment(config)

        print(f"    Collecting {N_RUNS}-run statistics for S{idx+1}...")
        stats = collect_convergence_statistics(
            env, seeds=CONV_SEEDS, max_iter=MAX_ITER, pop_size=POP_SIZE
        )
        all_results[n_tasks] = stats

        for algo in ALGO_KEYS:
            delay_data[algo].append(stats[algo]['delay_mean'])
            delay_err[algo].append(stats[algo]['delay_std'])
            tcr_data[algo].append(stats[algo]['tcr_mean'])
            tcr_err[algo].append(stats[algo]['tcr_std'])

    
    fig, axes  = plt.subplots(2, 2, figsize=(14, 10))
    axes       = axes.flatten()
    iterations = np.arange(1, MAX_ITER + 1)

    for idx, n_tasks in enumerate(TASK_COUNTS):
        ax      = axes[idx]
        stats   = all_results[n_tasks]
        s_label = f'S{idx+1} ({n_tasks} Tasks)'

        
        rf = stats['RandomOffloading']['fitness_mean']
        ax.axhline(y=rf, color=COLORS['RandomOffloading'], linestyle='--',
                   linewidth=1.5, label=f'Random ({rf:.2f})')

        for algo in CONVERGENCE_ALGO_KEYS:
            mean = stats[algo]['conv_mean']
            std  = stats[algo]['conv_std']

            ax.plot(iterations, mean,
                    color=COLORS[algo], label=LABELS[algo],
                    linewidth=2, marker=MARKERS[algo],
                    markevery=10, markersize=6)
            ax.fill_between(iterations, mean - std, mean + std,
                             color=COLORS[algo], alpha=0.15, linewidth=0)

        ax.set_title(f'Convergence — {s_label}, {FIXED_SERVERS} Servers ({N_RUNS} runs)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Convergence Comparison — Task Scalability ({FIXED_SERVERS} Servers, '
        f'mean \u00b1 std over {N_RUNS} runs)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_convergence.png'))
    plt.close()
    print("\n  Saved: fig1_convergence.png")

    # Total Delay bar chart (mean ± std over 30 runs)
    _plot_grouped_bar(
        data            = delay_data,
        algo_keys       = ALGO_KEYS,
        scenario_labels = TASK_SCENARIO_LABELS,
        ylabel          = 'Total Delay (s)',
        title           = f'Total Delay with ({FIXED_SERVERS} Servers, mean \u00b1 std, {N_RUNS} runs)',
        savepath        = os.path.join(output_dir, 'fig2a_total_delay.png'),
        value_fmt       = '{:.1f}',
        err             = delay_err
    )
    print("  Saved: fig2a_total_delay.png")

    # TCR bar chart (mean ± std over 30 runs)
    _plot_grouped_bar(
        data            = tcr_data,
        algo_keys       = ALGO_KEYS,
        scenario_labels = TASK_SCENARIO_LABELS,
        ylabel          = 'TCR (%)',
        title           = f'Task Completion Ratio with ({FIXED_SERVERS} Servers, mean \u00b1 std, {N_RUNS} runs)',
        savepath        = os.path.join(output_dir, 'fig2b_tcr.png'),
        value_fmt       = '{:.1f}',
        ylim            = (0, 65),
        err             = tcr_err
    )
    print("  Saved: fig2b_tcr.png")

    # Print result tables (mean ± std over 30 runs)
    for idx, n_tasks in enumerate(TASK_COUNTS):
        _print_table(all_results[n_tasks],
                     f"S{idx+1}: {n_tasks} Tasks, {FIXED_SERVERS} Servers")

    return all_results, delay_data, tcr_data


# Main function
def main():
    output_dir = r'C:\Users\CSE\Desktop\Python\CO-01'
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    print("\n" + "#" * 70)
    print("#   MEC OFFLOADING EXPERIMENTS                                #")
    print("#   Algorithms : Random, GA, PSO, CSA, CEO                   #")
    print(f"#   Population : {POP_SIZE}  |  Iterations : {MAX_ITER}"
          + " " * 21 + "#")
    print(f"#   Weights    : w1={W1}, w2={W2}"
          + " " * 33 + "#")
    print("#" * 70)

    experiment_task_scalability(output_dir)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETED in {time.time() - start:.1f}s")
    print(f"  Output → {output_dir}/")
    print(f"")
    print(f"  Task Scalability (5 servers fixed, tasks vary):")
    print(f"    fig1_convergence.png")
    print(f"    fig2a_total_delay.png")
    print(f"    fig2b_tcr.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
