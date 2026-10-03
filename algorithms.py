"""Baselines: Random, GA, PSO, CSA, and GA-PSO.
"""
import numpy as np
import time
from typing import List, Tuple, Optional
from system_model import MECEnvironment


class BaseOptimizer:
    def __init__(self, env: MECEnvironment, max_iterations: int = 100,
                 population_size: int = 50, seed: Optional[int] = None,
                 max_nfe: Optional[int] = None):
        self.env = env
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.max_nfe = max_nfe
        self.rng = np.random.RandomState(seed)
        self.num_tasks = env.config.num_tasks
        self.num_options = env.get_solution_space_size()  # 0 = local, 1..n = servers
        self.convergence_history: List[float] = []
        self.nfe_history: List[int] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        self.execution_time: float = 0.0

    # ---- budget helpers -------------------------------------------------
    def _should_stop(self, it: int) -> bool:
        if self.max_nfe is not None:
            return self.env.nfe >= self.max_nfe
        return it >= self.max_iterations

    def _progress(self, it: int) -> float:
        if self.max_nfe is not None:
            return min(1.0, self.env.nfe / self.max_nfe)
        return it / self.max_iterations

    def _record(self):
        self.convergence_history.append(self.best_fitness)
        self.nfe_history.append(self.env.nfe)

    # ---- common ---------------------------------------------------------
    def _random_solution(self) -> np.ndarray:
        return self.rng.randint(0, self.num_options, size=self.num_tasks)

    def _evaluate(self, solution: np.ndarray) -> float:
        return self.env.fitness(solution)

    def _clip_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(np.round(solution).astype(int), 0, self.num_options - 1)

    def optimize(self) -> dict:
        raise NotImplementedError

    def get_results(self) -> dict:
        r = self.env.evaluate_solution(self.best_solution)
        r['convergence_history'] = self.convergence_history
        r['nfe_history'] = self.nfe_history
        r['nfe'] = self.env.nfe
        r['execution_time'] = self.execution_time
        r['algorithm'] = self.__class__.__name__
        return r


# 1. RANDOM OFFLOADING -------------------------------------------------------
class RandomOffloading(BaseOptimizer):
    def __init__(self, env, num_trials: int = 1000, seed=None, max_nfe=None):
        super().__init__(env, max_iterations=1, population_size=num_trials,
                         seed=seed, max_nfe=max_nfe)
        self.num_trials = max_nfe if max_nfe is not None else num_trials

    def optimize(self) -> dict:
        self.env.reset_nfe()
        t0 = time.time()
        step = max(1, self.num_trials // 100)
        for trial in range(self.num_trials):
            s = self._random_solution()
            f = self._evaluate(s)
            if f < self.best_fitness:
                self.best_fitness, self.best_solution = f, s.copy()
            if (trial + 1) % step == 0:
                self._record()
        self.execution_time = time.time() - t0
        return self.get_results()


# 2. GENETIC ALGORITHM (GA) — ---------------------------------
class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, env, max_iterations=100, population_size=50,
                 crossover_rate=0.9, mutation_rate=0.05, tournament_size=3,
                 elite_count=2, seed=None, max_nfe=None):
        super().__init__(env, max_iterations, population_size, seed, max_nfe)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count

    def _tournament_selection(self, pop, fit):
        c = self.rng.choice(self.population_size, size=self.tournament_size, replace=False)
        return pop[c[np.argmin(fit[c])]].copy()

    def _uniform_crossover(self, p1, p2):
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        mask = self.rng.random(self.num_tasks) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)

    def _mutate(self, ind):
        m = ind.copy()
        for i in range(self.num_tasks):
            if self.rng.random() < self.mutation_rate:
                m[i] = self.rng.randint(0, self.num_options)
        return m

    def optimize(self) -> dict:
        self.env.reset_nfe()
        t0 = time.time()
        pop = np.array([self._random_solution() for _ in range(self.population_size)])
        fit = np.array([self._evaluate(ind) for ind in pop])
        b = np.argmin(fit)
        self.best_solution, self.best_fitness = pop[b].copy(), fit[b]
        it = 0
        while not self._should_stop(it):
            new_pop = [pop[i].copy() for i in np.argsort(fit)[:self.elite_count]]
            while len(new_pop) < self.population_size:
                c1, c2 = self._uniform_crossover(self._tournament_selection(pop, fit),
                                                 self._tournament_selection(pop, fit))
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(c2))
            pop = np.array(new_pop[:self.population_size])
            fit = np.array([self._evaluate(ind) for ind in pop])
            g = np.argmin(fit)
            if fit[g] < self.best_fitness:
                self.best_fitness, self.best_solution = fit[g], pop[g].copy()
            self._record()
            it += 1
        self.execution_time = time.time() - t0
        return self.get_results()


# 3. PARTICLE SWARM OPTIMIZATION (PSO) —  -------------------
class ParticleSwarmOptimization(BaseOptimizer):
    def __init__(self, env, max_iterations=100, population_size=50,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, seed=None, max_nfe=None):
        super().__init__(env, max_iterations, population_size, seed, max_nfe)
        self.w_start, self.w_end, self.c1, self.c2 = w_start, w_end, c1, c2
        self.v_max = (self.num_options - 1) / 2

    def optimize(self) -> dict:
        self.env.reset_nfe()
        t0 = time.time()
        N, m, ub = self.population_size, self.num_tasks, self.num_options - 1
        pos = self.rng.uniform(0, ub, size=(N, m))
        vel = self.rng.uniform(-self.v_max, self.v_max, size=(N, m))
        pbest, pfit = pos.copy(), np.full(N, np.inf)
        gbest, gfit = None, np.inf
        for i in range(N):
            f = self._evaluate(self._clip_solution(pos[i]))
            pfit[i] = f
            if f < gfit:
                gfit, gbest = f, pos[i].copy()
        self.best_fitness, self.best_solution = gfit, self._clip_solution(gbest)
        it = 0
        while not self._should_stop(it):
            w = self.w_start - (self.w_start - self.w_end) * self._progress(it)
            for i in range(N):
                r1, r2 = self.rng.random(m), self.rng.random(m)
                vel[i] = w * vel[i] + self.c1 * r1 * (pbest[i] - pos[i]) + self.c2 * r2 * (gbest - pos[i])
                vel[i] = np.clip(vel[i], -self.v_max, self.v_max)
                pos[i] = np.clip(pos[i] + vel[i], 0, ub)
                f = self._evaluate(self._clip_solution(pos[i]))
                if f < pfit[i]:
                    pfit[i], pbest[i] = f, pos[i].copy()
                if f < gfit:
                    gfit, gbest = f, pos[i].copy()
            self.best_fitness, self.best_solution = gfit, self._clip_solution(gbest)
            self._record()
            it += 1
        self.execution_time = time.time() - t0
        return self.get_results()


# 4. CROW SEARCH ALGORITHM (CSA) — ---------------------------
class CrowSearchAlgorithm(BaseOptimizer):
    def __init__(self, env, max_iterations=100, population_size=50,
                 awareness_prob=0.1, flight_length=2.0, seed=None, max_nfe=None):
        super().__init__(env, max_iterations, population_size, seed, max_nfe)
        self.AP, self.fl = awareness_prob, flight_length

    def optimize(self) -> dict:
        self.env.reset_nfe()
        t0 = time.time()
        N, m, ub = self.population_size, self.num_tasks, self.num_options - 1
        pos = self.rng.uniform(0, ub, size=(N, m))
        mem = pos.copy()
        mfit = np.array([self._evaluate(self._clip_solution(pos[i])) for i in range(N)])
        b = np.argmin(mfit)
        self.best_fitness, self.best_solution = mfit[b], self._clip_solution(mem[b])
        it = 0
        while not self._should_stop(it):
            new = np.zeros_like(pos)
            for i in range(N):
                j = self.rng.randint(0, N)
                while j == i:
                    j = self.rng.randint(0, N)
                r_j, r_i = self.rng.random(), self.rng.random()
                if r_j >= self.AP:
                    new[i] = pos[i] + r_i * self.fl * (mem[j] - pos[i])
                else:
                    new[i] = self.rng.uniform(0, ub, size=m)
                new[i] = np.clip(new[i], 0, ub)
            pos = new
            for i in range(N):
                d = self._clip_solution(pos[i])
                f = self._evaluate(d)
                if f < mfit[i]:
                    mem[i], mfit[i] = pos[i].copy(), f
                if f < self.best_fitness:
                    self.best_fitness, self.best_solution = f, d.copy()
            self._record()
            it += 1
        self.execution_time = time.time() - t0
        return self.get_results()


# 5. GA-PSO — ---------
class GAPSO(BaseOptimizer):

    def __init__(self, env, max_iterations=100, population_size=50,
                 w=0.8, c1=1.5, c2=2.5, c3=2.0, mutation_rate=0.01,
                 count_max=7, savior_fraction=0.2, seed=None, max_nfe=None):
        super().__init__(env, max_iterations, population_size, seed, max_nfe)
        self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3
        self.sigma, self.count_max, self.savior_fraction = mutation_rate, count_max, savior_fraction
        self.v_max = (self.num_options - 1) / 2

    def optimize(self) -> dict:
        self.env.reset_nfe()
        t0 = time.time()
        N, m, ub = self.population_size, self.num_tasks, self.num_options - 1
        ev = lambda x: self._evaluate(self._clip_solution(x))
        X = self.rng.uniform(0, ub, size=(N, m))
        V = self.rng.uniform(-self.v_max, self.v_max, size=(N, m))
        F = np.array([ev(X[i]) for i in range(N)])
        P, PF = X.copy(), F.copy()                       # personal bests
        g = np.argmin(PF); G, GF = P[g].copy(), PF[g]    # global best
        count = np.zeros(N, dtype=int)
        n_sub = max(2, int(round(self.savior_fraction * N)))
        it = 0
        while not self._should_stop(it):
            # ---------------- GA stage: crossover, mutation, selection
            for i in range(N):
                k = self.rng.randint(0, N)
                if PF[i] < PF[k]:
                    phi = self.rng.random(m)
                    O = phi * P[i] + (1 - phi) * G
                else:
                    O = P[k].copy()
                mut = self.rng.random(m) < self.sigma
                O[mut] = self.rng.uniform(0, ub, size=int(mut.sum()))
                fo = ev(O)
                if fo < F[i]:
                    X[i], F[i] = O, fo
                    count[i] = 0
                else:
                    count[i] += 1
                if F[i] < PF[i]:
                    P[i], PF[i] = X[i].copy(), F[i]
                if F[i] < GF:
                    G, GF = X[i].copy(), F[i]
            # ---------------- Savior
            for i in np.nonzero(count > self.count_max)[0]:
                sub = self.rng.choice(N, size=n_sub, replace=False)
                b = sub[np.argmin(F[sub])]
                X[i], F[i], count[i] = X[b].copy(), F[b], 0
            # ---------------- PSO stage (Eq. 14)
            for i in range(N):
                p1, p2, p3 = self.rng.random(m), self.rng.random(m), self.rng.random(m)
                y = (self.c1 * p1 * P[i] + self.c2 * p2 * G) / (self.c1 * p1 + self.c2 * p2 + 1e-12)
                V[i] = np.clip(self.w * (V[i] + self.c3 * p3 * (y - X[i])), -self.v_max, self.v_max)
                X[i] = np.clip(X[i] + V[i], 0, ub)
                F[i] = ev(X[i])
                if F[i] < PF[i]:
                    P[i], PF[i] = X[i].copy(), F[i]
                if F[i] < GF:
                    G, GF = X[i].copy(), F[i]
            self.best_fitness, self.best_solution = GF, self._clip_solution(G)
            self._record()
            it += 1
        self.best_fitness, self.best_solution = GF, self._clip_solution(G)
        self.execution_time = time.time() - t0
        return self.get_results()
