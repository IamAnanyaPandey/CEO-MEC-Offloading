
import numpy as np
import time
from math import gamma, sin, pi
from typing import List, Optional
from system_model import MECEnvironment


class CEO:
    def __init__(self, env: MECEnvironment, max_iterations: int = 100,
                 population_size: int = 50, awareness_prob: float = 0.1,
                 flight_length: float = 2.0, levy_beta: float = 1.5,
                 greedy_fraction: float = 0.2, elite_fraction: float = 0.3,
                 stagnation_patience: int = 8, restart_fraction: float = 0.3,
                 seed: Optional[int] = None, max_nfe: Optional[int] = None,
                 init_mode: str = 'greedy_obl', use_memory: bool = True,
                 use_levy: bool = True, use_elite: bool = True,
                 use_adaptive_AP: bool = True, use_restart: bool = True):
        self.env = env
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.max_nfe = max_nfe
        self.rng = np.random.RandomState(seed)
        self.num_tasks = env.config.num_tasks
        self.num_options = env.get_solution_space_size()
        self.AP, self.fl = awareness_prob, flight_length
        self.levy_beta = levy_beta
        self.greedy_count = max(2, int(population_size * greedy_fraction)) if init_mode == 'greedy_obl' else 0
        self.elite_count = max(2, int(population_size * elite_fraction))
        self.stagnation_patience = stagnation_patience
        self.restart_count = max(2, int(population_size * restart_fraction))
        self.init_mode = init_mode
        self.use_memory, self.use_levy, self.use_elite = use_memory, use_levy, use_elite
        self.use_adaptive_AP, self.use_restart = use_adaptive_AP, use_restart

        b = levy_beta
        self.levy_sigma = ((gamma(1 + b) * sin(pi * b / 2)) /
                           (gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))) ** (1 / b)

        self.convergence_history: List[float] = []
        self.nfe_history: List[int] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        self.execution_time: float = 0.0
        self.n_restarts = 0

    # ---- helpers ----------------------------------------------------------
    def _should_stop(self, it):
        if self.max_nfe is not None:
            return self.env.nfe >= self.max_nfe
        return it >= self.max_iterations

    def _progress(self, it):
        if self.max_nfe is not None:
            return min(1.0, self.env.nfe / self.max_nfe)
        return it / self.max_iterations

    def _evaluate(self, s):
        return self.env.fitness(s)

    def _random_discrete(self):
        return self.rng.randint(0, self.num_options, size=self.num_tasks)

    def _levy_sample(self) -> int:
        u = self.rng.normal(0, self.levy_sigma)
        v = self.rng.normal(0, 1)
        step = abs(u / (abs(v) ** (1 / self.levy_beta)))
        return max(1, min(int(step * 2), self.num_tasks))

    def _discrete_obl(self, x):
        return (self.num_options - 1) - x          # x_bar = n - x

    # ---- 1. greedy + OBL initialisation ------------------------------------
    def _greedy_solution(self):
        
        env = self.env
        sol = np.zeros(self.num_tasks, dtype=int)
        for i, task in enumerate(env.tasks):
            cycles = task.data_size * task.cpu_cycles_per_byte
            best_opt, best_delay = 0, cycles / task.device_processing_power
            for j, server in enumerate(env.servers):
                proc = cycles / (server.processing_power / server.num_vms)
                rate = env.config.bandwidth * np.log2(
                    1 + task.transmit_power * task.channel_gains[j] / env.config.noise_power)
                d = proc + task.data_size / max(rate, 1e-10)
                if d < best_delay:
                    best_opt, best_delay = j + 1, d
            sol[i] = best_opt
        genes = self.rng.choice(self.num_tasks, size=max(1, self.num_tasks // 10), replace=False)
        for g in genes:
            sol[g] = self.rng.randint(0, self.num_options)
        return sol

    def _initialize_population(self):
        pop, fit = [], []
        for _ in range(self.greedy_count):
            x = self._greedy_solution()
            pop.append(x); fit.append(self._evaluate(x))
        for _ in range(self.population_size - self.greedy_count):
            x = self._random_discrete()
            if self.init_mode == 'random':
                pop.append(x); fit.append(self._evaluate(x)); continue
            xo = self._discrete_obl(x)
            fx, fo = self._evaluate(x), self._evaluate(xo)
            pop.append(x if fx <= fo else xo); fit.append(min(fx, fo))
        return np.array(pop), np.array(fit)

    # ---- 2. discrete memory adherence ---------------------------------------
    def _discrete_follow(self, crow, memory_j, progress):
        new = crow.copy()
        n_copy = max(1, int(self.num_tasks * (0.6 * (1 - progress) + 0.1)))
        g = self.rng.choice(self.num_tasks, size=n_copy, replace=False)
        new[g] = memory_j[g]
        return new

    def _csa_follow(self, crow, memory_j):          # ablation: original CSA rule + rounding
        x = crow + self.rng.random() * self.fl * (memory_j - crow)
        return np.clip(np.round(x).astype(int), 0, self.num_options - 1)

    # ---- 3. Lévy exploration ------------------------------------------------
    def _levy_exploration(self, crow):
        new = crow.copy()
        g = self.rng.choice(self.num_tasks, size=self._levy_sample(), replace=False)
        new[g] = self.rng.randint(0, self.num_options, size=len(g))
        return new

    # ---- 4. elite local search ----------------------------------------------
    def _local_search(self, sol, fit):
        best_sol, best_fit = sol.copy(), fit
        gene = self.rng.randint(0, self.num_tasks)            # (a) all n+1 values of one gene
        for val in range(self.num_options):
            if val == sol[gene]:
                continue
            c = sol.copy(); c[gene] = val
            f = self._evaluate(c)
            if f < best_fit:
                best_sol, best_fit = c, f
        if self.best_solution is not None:                    # (b) crossover with OD*
            c = sol.copy()
            mask = self.rng.random(self.num_tasks) < 0.3
            c[mask] = self.best_solution[mask]
            f = self._evaluate(c)
            if f < best_fit:
                best_sol, best_fit = c, f
        n_shift = self.rng.randint(3, min(6, self.num_tasks + 1))   # (c) +-1 on 3-5 genes
        g = self.rng.choice(self.num_tasks, size=n_shift, replace=False)
        c = sol.copy()
        c[g] = np.clip(c[g] + self.rng.choice([-1, 1], size=n_shift), 0, self.num_options - 1)
        f = self._evaluate(c)
        if f < best_fit:
            best_sol, best_fit = c, f
        return best_sol, best_fit

    # ---- 6. stagnation restart (fixed) -------------------------------------
    def _restart(self, pop, pop_fit, mem, mem_fit):
        worst = np.argsort(mem_fit)[-self.restart_count:]
        for idx in worst:
            new = self._random_discrete()
            mask = self.rng.random(self.num_tasks) < 0.5
            new[mask] = self.best_solution[mask]
            obl = self._discrete_obl(new)
            fn, fo = self._evaluate(new), self._evaluate(obl)
            s, f = (new, fn) if fn <= fo else (obl, fo)
            pop[idx], pop_fit[idx] = s, f
            mem[idx], mem_fit[idx] = s.copy(), f
        self.n_restarts += 1

    # ---- main loop ------------------------------------------------------------
    def optimize(self) -> dict:
        self.env.reset_nfe()
        t0 = time.time()
        N = self.population_size
        pop, pop_fit = self._initialize_population()
        mem, mem_fit = pop.copy(), pop_fit.copy()
        b = np.argmin(mem_fit)
        self.best_fitness, self.best_solution = mem_fit[b], mem[b].copy()
        stag, prev = 0, self.best_fitness
        it = 0
        while not self._should_stop(it):
            p = self._progress(it)
            AP_t = self.AP + (0.5 - self.AP) * p if self.use_adaptive_AP else self.AP
            for i in range(N):
                j = self.rng.randint(0, N)
                while j == i:
                    j = self.rng.randint(0, N)
                if self.rng.random() >= AP_t:
                    if self.use_memory:
                        new = self._discrete_follow(pop[i], mem[j], p)
                        if self.rng.random() < p * 0.8:
                            g = self.rng.choice(self.num_tasks, max(1, int(self.num_tasks * 0.15)), replace=False)
                            new[g] = self.best_solution[g]
                    else:
                        new = self._csa_follow(pop[i], mem[j])
                else:
                    new = self._levy_exploration(pop[i]) if self.use_levy else self._random_discrete()
                pop[i] = new
                f = self._evaluate(new)
                pop_fit[i] = f                                   # BUG FIX
                if f < mem_fit[i]:
                    mem[i], mem_fit[i] = new.copy(), f
                if f < self.best_fitness:
                    self.best_fitness, self.best_solution = f, new.copy()

            if self.use_elite:
                for idx in np.argsort(mem_fit)[:self.elite_count]:
                    s, f = self._local_search(mem[idx], mem_fit[idx])
                    if f < mem_fit[idx]:
                        mem[idx], mem_fit[idx] = s, f
                        if f < self.best_fitness:
                            self.best_fitness, self.best_solution = f, s.copy()

            if abs(self.best_fitness - prev) < 1e-10:
                stag += 1
            else:
                stag, prev = 0, self.best_fitness
            if self.use_restart and stag >= self.stagnation_patience:
                self._restart(pop, pop_fit, mem, mem_fit)
                stag = 0

            self.convergence_history.append(self.best_fitness)
            self.nfe_history.append(self.env.nfe)
            it += 1
        self.execution_time = time.time() - t0
        return self.get_results()

    def get_results(self) -> dict:
        r = self.env.evaluate_solution(self.best_solution)
        r['convergence_history'] = self.convergence_history
        r['nfe_history'] = self.nfe_history
        r['nfe'] = self.env.nfe
        r['execution_time'] = self.execution_time
        r['algorithm'] = 'CEO'
        r['n_restarts'] = self.n_restarts
        return r
