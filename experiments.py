
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, csv, pickle, time, tracemalloc, itertools, platform
from multiprocessing import Pool
import numpy as np

from system_model import SimulationConfig, MECEnvironment
from algorithms import (RandomOffloading, GeneticAlgorithm, ParticleSwarmOptimization,
                        CrowSearchAlgorithm, GAPSO)
from ceo import CEO
POP = 50                      # population size, all algorithms
NFE_MAX = 15000               # identical evaluation budget, all algorithms and experiments
W1_DEFAULT = 0.5              # w2 = 1 - w1
NORMALIZE = True              # objective uses T_total / T_ref (T_ref = all-local total delay)
EDGE_PROC_RANGE = (40e9, 50e9)   # C_j = U[40, 50] GHz  
VM_RANGE = (40, 100)             # K_j = U[40, 100]     

# CEO parameters. g_f, e_f, P_s were selected by the E5 grid on tuning instances
# 1001-1005 (never used for reported results). All other values as in the paper.
CEO_PARAMS = dict(awareness_prob=0.1, flight_length=2.0, levy_beta=1.5,
                  greedy_fraction=0.3, elite_fraction=0.3, stagnation_patience=5,
                  restart_fraction=0.3)

TASKS_E1 = [50, 100, 150, 200]
SERVERS_E1 = 5
SCALE_CONFIGS = [(300, 5), (500, 5), (200, 10), (200, 15), (200, 20)]
W1_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9]
TUNING_SEEDS = [1001, 1002, 1003, 1004, 1005]

ABLATION = {                                        # variant name -> CEO switches
    'CEO (full)':      {},
    'w/o greedy init': {'init_mode': 'obl'},
    'w/o greedy+OBL':  {'init_mode': 'random'},
    'w/o memory adh.': {'use_memory': False},
    'w/o Levy':        {'use_levy': False},
    'w/o elite LS':    {'use_elite': False},
    'w/o adaptive AP': {'use_adaptive_AP': False},
    'w/o restart':     {'use_restart': False},
    'CSA + OBL only':  {'init_mode': 'obl', 'use_memory': False, 'use_levy': False,
                        'use_elite': False, 'use_adaptive_AP': False, 'use_restart': False},
    'Discrete CSA':    {'init_mode': 'random', 'use_memory': False, 'use_levy': False,
                        'use_elite': False, 'use_adaptive_AP': False, 'use_restart': False},
}
ALGOS = ['Random', 'GA', 'PSO', 'CSA', 'GA-PSO', 'CEO']
# =========================================================================


def make_env(n_tasks, n_servers, seed, w1=W1_DEFAULT):
    cfg = SimulationConfig(num_tasks=n_tasks, num_edge_servers=n_servers,
                           w1=w1, w2=1 - w1, normalize_delay=NORMALIZE, seed=seed,
                           edge_processing_range=EDGE_PROC_RANGE, vm_range=VM_RANGE)
    return MECEnvironment(cfg)


def make_algo(name, env, seed, ceo_kwargs=None):
    B = NFE_MAX
    if name == 'Random':
        return RandomOffloading(env, seed=seed, max_nfe=B)
    if name == 'GA':
        return GeneticAlgorithm(env, population_size=POP, crossover_rate=0.9, mutation_rate=0.05,
                                tournament_size=3, elite_count=2, seed=seed, max_nfe=B)
    if name == 'PSO':
        return ParticleSwarmOptimization(env, population_size=POP, w_start=0.9, w_end=0.4,
                                         c1=2.0, c2=2.0, seed=seed, max_nfe=B)
    if name == 'CSA':
        return CrowSearchAlgorithm(env, population_size=POP, awareness_prob=0.1,
                                   flight_length=2.0, seed=seed, max_nfe=B)
    if name == 'GA-PSO':
        return GAPSO(env, population_size=POP, w=0.8, c1=1.5, c2=2.5, c3=2.0,
                     mutation_rate=0.01, count_max=7, seed=seed, max_nfe=B)
    if name == 'CEO':
        kw = dict(CEO_PARAMS)
        kw.update(ceo_kwargs or {})
        return CEO(env, population_size=POP, seed=seed, max_nfe=B, **kw)
    raise ValueError(name)


def run_job(job):
    exp, label, m, n, w1, algo, variant, kw, seed, mem = job
    env = make_env(m, n, seed, w1)
    opt = make_algo(algo, env, seed, kw)
    if mem:
        tracemalloc.start()
    r = opt.optimize()
    peak = tracemalloc.get_traced_memory()[1] / 1e6 if mem else float('nan')
    if mem:
        tracemalloc.stop()
    row = dict(exp=exp, config=label, n_tasks=m, n_servers=n, w1=w1, algorithm=variant,
               seed=seed, budget=NFE_MAX, fitness=r['fitness'], total_delay=r['total_delay'],
               tcr=r['tcr'], nfe=r['nfe'], runtime_s=r['execution_time'], peak_mem_MB=peak,
               local_tasks=r['local_tasks'],
               server_counts=';'.join(str(c) for c in r['edge_task_distribution']))
    conv = (np.array(r['nfe_history']), np.array(r['convergence_history']))
    return row, conv


def build_jobs(exp, seeds):
    J = []
    if exp == 'E1':
        for k, m in enumerate(TASKS_E1):
            for a, s in itertools.product(ALGOS, seeds):
                J.append(('E1', f'S{k+1}', m, SERVERS_E1, W1_DEFAULT, a, a, None, s, False))
        for k, m in enumerate(TASKS_E1):    # peak memory: one instance, tracemalloc (slow)
            for a in ALGOS:
                J.append(('E1mem', f'S{k+1}', m, SERVERS_E1, W1_DEFAULT, a, a, None, seeds[0], True))
    elif exp == 'E2':
        for k, m in [(2, 100), (4, 200)]:
            for (v, kw), s in itertools.product(ABLATION.items(), seeds):
                J.append(('E2', f'S{k}', m, SERVERS_E1, W1_DEFAULT, 'CEO', v, kw, s, False))
    elif exp == 'E3':
        for m, n in SCALE_CONFIGS:
            for a, s in itertools.product(ALGOS, seeds):
                J.append(('E3', f'{m}T-{n}S', m, n, W1_DEFAULT, a, a, None, s, False))
    elif exp == 'E4':
        for w1 in W1_SWEEP:
            for a, s in itertools.product(['CEO', 'GA-PSO'], seeds):
                J.append(('E4', f'w1={w1}', 200, SERVERS_E1, w1, a, a, None, s, False))
    elif exp == 'E5':
        for gf, ef, ps in itertools.product([0.1, 0.2, 0.3], [0.1, 0.3, 0.5], [5, 8, 12]):
            kw = {'greedy_fraction': gf, 'elite_fraction': ef, 'stagnation_patience': ps}
            for s in TUNING_SEEDS:
                J.append(('E5', 'S2-tuning', 100, SERVERS_E1, W1_DEFAULT, 'CEO',
                          f'gf={gf},ef={ef},Ps={ps}', kw, s, False))
    return J


def run_experiment(exp, seeds, workers, out):
    t0 = time.time()
    print(f"\n=== {exp} ===")
    jobs = build_jobs(exp, seeds)
    part = os.path.join(out, f'{exp}_partial.csv')
    convf = os.path.join(out, f'{exp}_conv_partial.pkl')
    done = set()
    if os.path.exists(part):
        with open(part) as f:
            for r in csv.DictReader(f):
                done.add((r['exp'], r['config'], r['algorithm'], int(r['seed'])))
    todo = [j for j in jobs if (j[0], j[1], j[6], j[8]) not in done]
    print(f"  {len(jobs)} runs in total, {len(todo)} remaining, {workers} worker(s)")
    # memory jobs are run last and alone so that other processes do not disturb them
    normal = [j for j in todo if not j[9]]
    memjobs = [j for j in todo if j[9]]

    def save(row, conv):
        new = not os.path.exists(part)
        with open(part, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        if row['exp'] == 'E1':
            with open(convf, 'ab') as f:
                pickle.dump(((row['config'], row['algorithm']), conv), f)

    n_done = 0
    if normal:
        with Pool(workers) as pool:
            for row, conv in pool.imap_unordered(run_job, normal, chunksize=1):
                save(row, conv); n_done += 1
                if n_done % 50 == 0 or n_done == len(normal):
                    print(f"  {n_done}/{len(normal)} runs done ({time.time()-t0:.0f} s)")
    for j in memjobs:
        save(*run_job(j))

    with open(part) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(out, f'{exp}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if os.path.exists(convf):
        convs = {}
        with open(convf, 'rb') as f:
            while True:
                try:
                    k, c = pickle.load(f)
                except EOFError:
                    break
                convs.setdefault(k, []).append(c)
        with open(os.path.join(out, 'E1_convergence.pkl'), 'wb') as f:
            pickle.dump(convs, f)
    print(f"  {exp} COMPLETE -> {os.path.join(out, exp + '.csv')} ({len(rows)} rows, {time.time()-t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='all', help='E1 | E2 | E3 | E4 | E5 | all (default: all)')
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--out', default='results')
    ap.add_argument('--quick', action='store_true', help='2 instances only (test)')
    a = ap.parse_args()
    seeds = [1, 2] if a.quick else list(range(1, a.runs + 1))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'environment.txt'), 'w') as f:
        f.write(f"python {platform.python_version()} | numpy {np.__version__} | "
                f"{platform.platform()} | {platform.processor()} | workers={a.workers}\n")
    exps = ['E5', 'E1', 'E2', 'E3', 'E4'] if a.exp == 'all' else [a.exp]
    for e in exps:
        run_experiment(e, seeds, a.workers, a.out)


if __name__ == '__main__':      
    main()