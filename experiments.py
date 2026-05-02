import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
import os

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

# Scenario labels for task experiments
TASK_SCENARIO_LABELS = [
    'S1\n(50 Tasks)',
    'S2\n(100 Tasks)',
    'S3\n(150 Tasks)',
    'S4\n(200 Tasks)'
]


# Running all algorithms
def run_single_experiment(env, max_iter=MAX_ITER, pop_size=POP_SIZE, seed=50):
    results = {}

    rand = RandomOffloading(env, num_trials=pop_size * max_iter, seed=seed)
    results['RandomOffloading'] = rand.optimize()

    ga = GeneticAlgorithm(
        env, max_iterations=max_iter, population_size=pop_size,
        crossover_rate=0.9, mutation_rate=0.05, tournament_size=3,
        elite_count=2, seed=seed
    )
    results['GeneticAlgorithm'] = ga.optimize()

    pso = ParticleSwarmOptimization(
        env, max_iterations=max_iter, population_size=pop_size,
        w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, seed=seed
    )
    results['ParticleSwarmOptimization'] = pso.optimize()

    csa = CrowSearchAlgorithm(
        env, max_iterations=max_iter, population_size=pop_size,
        awareness_prob=0.1, flight_length=2.0, seed=seed
    )
    results['CrowSearchAlgorithm'] = csa.optimize()

    ceo = CEO(
        env, max_iterations=max_iter, population_size=pop_size,
        awareness_prob=0.1, flight_length=2.0,
        levy_beta=1.5, greedy_fraction=0.2, elite_fraction=0.3,
        stagnation_patience=8, restart_fraction=0.3, seed=seed
    )
    results['CEO'] = ceo.optimize()

    return results


def _plot_grouped_bar(data, algo_keys, scenario_labels, ylabel, title,
                      savepath, value_fmt='{:.1f}', ylim=None):
    n_algos     = len(algo_keys)
    n_scenarios = len(scenario_labels)

    bar_w = 0.15
    x     = np.arange(n_scenarios)

    fig, ax = plt.subplots(figsize=(13, 6))

    for a_idx, algo in enumerate(algo_keys):
        offsets = x + (a_idx - (n_algos - 1) / 2) * bar_w
        vals    = data[algo]
        bars    = ax.bar(
            offsets, vals,
            width=bar_w,
            color=COLORS[algo],
            label=LABELS[algo],
            edgecolor='white',
            linewidth=0.4,
            alpha=0.9
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + (0.3 if 'TCR' in ylabel else 1.0),
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


def _print_table(results, label):
    print(f"\n{'='*85}")
    print(f"  SCENARIO: {label}")
    print(f"{'='*85}")
    print(f"  {'Algorithm':<10} | {'Fitness':>10} | "
          f"{'Total Delay (s)':>16} | {'TCR (%)':>8} | {'Time (s)':>9}")
    print(f"  {'-'*75}")
    for algo in ALGO_KEYS:
        r = results[algo]
        print(f"  {LABELS[algo]:<10} | {r['fitness']:>10.4f} | "
              f"{r['total_delay']:>16.2f} | "
              f"{r['tcr']*100:>8.1f} | {r['execution_time']:>9.2f}")

    ceo_delay = results['CEO']['total_delay']
    csa_delay = results['CrowSearchAlgorithm']['total_delay']
    ceo_tcr   = results['CEO']['tcr'] * 100
    csa_tcr   = results['CrowSearchAlgorithm']['tcr'] * 100

    delay_imp = (csa_delay - ceo_delay) / csa_delay * 100
    tcr_imp   = (ceo_tcr - csa_tcr) / csa_tcr * 100 if csa_tcr > 0 else float('inf')

    print(f"\n  CEO vs CSA → Delay reduced by {delay_imp:.2f}%  |  "
          f"TCR improved by {tcr_imp:.2f}%")
    print(f"{'='*85}")


# EXPERIMENT: Fixed servers = 5, Tasks: 50 / 100 / 150 / 200

def experiment_task_scalability(output_dir):
    print("\n" + "=" * 70)
    print("  TASK SCALABILITY EXPERIMENT")
    print("  Fixed Servers = 5  |  Tasks = 50, 100, 150, 200")
    print(f"  Weights: w1={W1}, w2={W2}")
    print("=" * 70)

    all_results = {}
    delay_data  = {algo: [] for algo in ALGO_KEYS}
    tcr_data    = {algo: [] for algo in ALGO_KEYS}

    for idx, n_tasks in enumerate(TASK_COUNTS):
        print(f"\n  Running S{idx+1}: {n_tasks} tasks, {FIXED_SERVERS} servers ...")
        config  = SimulationConfig(
            num_tasks=n_tasks, num_edge_servers=FIXED_SERVERS,
            w1=W1, w2=W2, seed=50
        )
        env     = MECEnvironment(config)
        results = run_single_experiment(env, seed=50)

        all_results[n_tasks] = results
        for algo in ALGO_KEYS:
            delay_data[algo].append(results[algo]['total_delay'])
            tcr_data[algo].append(results[algo]['tcr'] * 100)

    # Convergence plot (2x2 grid for S1-S4)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, n_tasks in enumerate(TASK_COUNTS):
        ax      = axes[idx]
        res     = all_results[n_tasks]
        s_label = f'S{idx+1} ({n_tasks} Tasks)'

        for algo in ALGO_KEYS:
            if algo == 'RandomOffloading':
                rf = res[algo]['fitness']
                ax.axhline(y=rf, color=COLORS[algo], linestyle='--',
                           linewidth=1.5, label=f'Random ({rf:.2f})')
            else:
                history = res[algo]['convergence_history']
                ax.plot(range(1, len(history) + 1), history,
                        color=COLORS[algo], label=LABELS[algo],
                        linewidth=2, marker=MARKERS[algo],
                        markevery=10, markersize=6)

        ax.set_title(f'Convergence — {s_label}, {FIXED_SERVERS} Servers')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Convergence Comparison — Task Scalability ({FIXED_SERVERS} Servers)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_convergence.png'))
    plt.close()
    print("\n  Saved: fig1_convergence.png")

    # Total Delay bar chart
    _plot_grouped_bar(
        data            = delay_data,
        algo_keys       = ALGO_KEYS,
        scenario_labels = TASK_SCENARIO_LABELS,
        ylabel          = 'Total Delay (s)',
        title           = f'Total Delay with ({FIXED_SERVERS} Servers)',
        savepath        = os.path.join(output_dir, 'fig2a_total_delay.png'),
        value_fmt       = '{:.1f}'
    )
    print("  Saved: fig2a_total_delay.png")

    # TCR bar chart 
    _plot_grouped_bar(
        data            = tcr_data,
        algo_keys       = ALGO_KEYS,
        scenario_labels = TASK_SCENARIO_LABELS,
        ylabel          = 'TCR (%)',
        title           = f'Task Completion Ratio with ({FIXED_SERVERS} Servers)',
        savepath        = os.path.join(output_dir, 'fig2b_tcr.png'),
        value_fmt       = '{:.1f}',
        ylim            = (0, 65)
    )
    print("  Saved: fig2b_tcr.png")

    # Print result tables 
    for idx, n_tasks in enumerate(TASK_COUNTS):
        _print_table(all_results[n_tasks],
                     f"S{idx+1}: {n_tasks} Tasks, {FIXED_SERVERS} Servers")

    return all_results, delay_data, tcr_data


# Main function
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
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
