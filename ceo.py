import numpy as np
import time
from math import gamma, sin, pi
from typing import List, Optional
from system_model import MECEnvironment


class CEO:
   
    def __init__(self, env: MECEnvironment, max_iterations: int = 100,
                 population_size: int = 50, awareness_prob: float = 0.1,
                 flight_length: float = 2.0,
                 # CEO-specific parameters
                 levy_beta: float = 1.5,
                 greedy_fraction: float = 0.2,
                 elite_fraction: float = 0.3,
                 stagnation_patience: int = 8,
                 restart_fraction: float = 0.3,
                 seed: Optional[int] = None):

        self.env = env
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.rng = np.random.RandomState(seed)

        self.num_tasks = env.config.num_tasks
        self.num_options = env.get_solution_space_size()  # 0 to n_servers

        # Standard CSA parameters
        self.AP = awareness_prob
        self.fl = flight_length

        # CEO parameters
        self.levy_beta = levy_beta
        self.greedy_count = max(2, int(population_size * greedy_fraction))
        self.elite_count = max(2, int(population_size * elite_fraction))
        self.stagnation_patience = stagnation_patience
        self.restart_count = max(2, int(population_size * restart_fraction))

        # Precompute Lévy sigma (Mantegna's algorithm)
        beta = self.levy_beta
        num = gamma(1 + beta) * sin(pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        self.levy_sigma = (num / den) ** (1 / beta)

        # Tracking
        self.convergence_history: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        self.execution_time: float = 0.0

    
    def _evaluate(self, solution: np.ndarray) -> float:
        return self.env.fitness(solution)

    def _random_discrete(self) -> np.ndarray:
        return self.rng.randint(0, self.num_options, size=self.num_tasks)

    def _levy_sample(self) -> int:
        u = self.rng.normal(0, self.levy_sigma)
        v = self.rng.normal(0, 1)
        step = abs(u / (abs(v) ** (1 / self.levy_beta)))
        # Scale to gene count
        n_genes = max(1, min(int(step * 2), self.num_tasks))
        return n_genes

    # Greedy + OBL Initialization
    def _greedy_solution(self) -> np.ndarray:
        """Greedy initialization: assign each task to its minimal-delay option.
        Uses simplified rate (no interference) since full offloading vector
        is not yet determined at initialization stage.
        """
        solution = np.zeros(self.num_tasks, dtype=int)

        for i in range(self.num_tasks):
            task = self.env.tasks[i]
            cycles = task.data_size * task.cpu_cycles_per_byte
            best_opt = 0
            best_delay = cycles / task.device_processing_power  # local delay

            for j in range(self.env.config.num_edge_servers):
                server = self.env.servers[j]
                vm_power = server.processing_power / server.num_vms
                proc = cycles / vm_power
                rate = self.env.config.bandwidth * np.log2(
                    1 + task.transmit_power * task.channel_gains[j]
                    / self.env.config.noise_power
                )
                trans = task.data_size / max(rate, 1e-10)

                if proc + trans < best_delay:
                    best_delay = proc + trans
                    best_opt = j + 1

            solution[i] = best_opt

        # Randomly perturb a few genes for diversity among greedy crows
        n_perturb = max(1, self.num_tasks // 10)
        genes = self.rng.choice(self.num_tasks, size=n_perturb, replace=False)
        for g in genes:
            solution[g] = self.rng.randint(0, self.num_options)

        return solution

    def _discrete_obl(self, x: np.ndarray) -> np.ndarray:
        """Discrete OBL: x_bar = n - x (paper Algorithm 1, line 7)."""
        return (self.num_options - 1) - x

    def _initialize_population(self) -> tuple:
        
        population = []
        fitness_vals = []

        for _ in range(self.greedy_count):
            x = self._greedy_solution()
            population.append(x)
            fitness_vals.append(self._evaluate(x))

        for _ in range(self.population_size - self.greedy_count):
            x = self._random_discrete()
            x_opp = self._discrete_obl(x)

            f_x = self._evaluate(x)
            f_opp = self._evaluate(x_opp)

            if f_x <= f_opp:
                population.append(x)
                fitness_vals.append(f_x)
            else:
                population.append(x_opp)
                fitness_vals.append(f_opp)

        return np.array(population), np.array(fitness_vals)

    # Discrete memory following
    def _discrete_follow(self, crow: np.ndarray, memory_j: np.ndarray,
                         progress: float) -> np.ndarray:
        
        new_crow = crow.copy()

        copy_rate = 0.6 * (1 - progress) + 0.1
        n_copy = max(1, int(self.num_tasks * copy_rate))

        genes_to_copy = self.rng.choice(self.num_tasks, size=n_copy, replace=False)
        new_crow[genes_to_copy] = memory_j[genes_to_copy]

        return new_crow

    # Levy controlled exploration
    def _levy_exploration(self, crow: np.ndarray) -> np.ndarray:
        new_crow = crow.copy()

        n_change = self._levy_sample()
        genes = self.rng.choice(self.num_tasks, size=n_change, replace=False)
        for g in genes:
            new_crow[g] = self.rng.randint(0, self.num_options)

        return new_crow

    # Elite local search
    def _local_search(self, solution: np.ndarray, fitness: float) -> tuple:
        best_sol = solution.copy()
        best_fit = fitness
        gene = self.rng.randint(0, self.num_tasks)
        original_val = solution[gene]
        for val in range(self.num_options):
            if val == original_val:
                continue
            candidate = solution.copy()
            candidate[gene] = val
            f = self._evaluate(candidate)
            if f < best_fit:
                best_sol = candidate.copy()
                best_fit = f

        if self.best_solution is not None:
            candidate = solution.copy()
            mask = self.rng.random(self.num_tasks) < 0.3
            candidate[mask] = self.best_solution[mask]
            f = self._evaluate(candidate)
            if f < best_fit:
                best_sol = candidate.copy()
                best_fit = f

        n_shift = self.rng.randint(3, min(6, self.num_tasks + 1))
        genes = self.rng.choice(self.num_tasks, size=n_shift, replace=False)
        candidate = solution.copy()
        for g in genes:
            shift = self.rng.choice([-1, 1])
            candidate[g] = np.clip(candidate[g] + shift, 0, self.num_options - 1)
        f = self._evaluate(candidate)
        if f < best_fit:
            best_sol = candidate.copy()
            best_fit = f

        return best_sol, best_fit

    # Elite guided restart
    def _elite_guided_restart(self, population: np.ndarray,
                               fitness_vals: np.ndarray) -> tuple:
      
        worst_indices = np.argsort(fitness_vals)[-self.restart_count:]

        for idx in worst_indices:
            new_sol = self._random_discrete()

            if self.best_solution is not None:
                mask = self.rng.random(self.num_tasks) < 0.5
                new_sol[mask] = self.best_solution[mask]

            # Also try the OBL opposite
            obl_sol = self._discrete_obl(new_sol)

            f_new = self._evaluate(new_sol)
            f_obl = self._evaluate(obl_sol)

            if f_new <= f_obl:
                population[idx] = new_sol
                fitness_vals[idx] = f_new
            else:
                population[idx] = obl_sol
                fitness_vals[idx] = f_obl

        return population, fitness_vals

    
    def optimize(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  CROW SEARCH BASED EFFICIENT OFFLOADING (CEO)")
        print(f"  Flock: {self.population_size} | Iters: {self.max_iterations}")
        print(f"  AP: {self.AP} | FL: {self.fl} | Levy beta: {self.levy_beta}")
        print(f"  Greedy: {self.greedy_count} | Elite: {self.elite_count}"
              f" | Patience: {self.stagnation_patience}")
        print(f"{'='*60}")

        start_time = time.time()

        
        population, pop_fitness = self._initialize_population()
        memory = population.copy()
        mem_fitness = pop_fitness.copy()
        best_idx = np.argmin(mem_fitness)
        self.best_fitness = mem_fitness[best_idx]
        self.best_solution = memory[best_idx].copy()
        stagnation_counter = 0
        prev_best = self.best_fitness

        for iteration in range(self.max_iterations):
            progress = iteration / self.max_iterations
            current_AP = self.AP + (0.5 - self.AP) * progress

            for i in range(self.population_size):
                # Select a random crow j to follow (j ≠ i)
                j = self.rng.randint(0, self.population_size)
                while j == i:
                    j = self.rng.randint(0, self.population_size)

                r_j = self.rng.random()

                if r_j >= current_AP:

                    new_crow = self._discrete_follow(
                        population[i], memory[j], progress
                    )

                    
                    if self.best_solution is not None and self.rng.random() < progress * 0.8:
                        n_gbest = max(1, int(self.num_tasks * 0.15))
                        gb_genes = self.rng.choice(self.num_tasks, n_gbest, replace=False)
                        new_crow[gb_genes] = self.best_solution[gb_genes]

                else:
                    new_crow = self._levy_exploration(population[i])

                # Update position
                population[i] = new_crow

                # Evaluate new position
                fit = self._evaluate(population[i])

                # Update personal memory if improved
                if fit < mem_fitness[i]:
                    memory[i] = population[i].copy()
                    mem_fitness[i] = fit

                # Update global best
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = population[i].copy()

            elite_indices = np.argsort(mem_fitness)[:self.elite_count]
            for idx in elite_indices:
                improved_sol, improved_fit = self._local_search(
                    memory[idx], mem_fitness[idx]
                )
                if improved_fit < mem_fitness[idx]:
                    memory[idx] = improved_sol
                    mem_fitness[idx] = improved_fit

                    if improved_fit < self.best_fitness:
                        self.best_fitness = improved_fit
                        self.best_solution = improved_sol.copy()

            if abs(self.best_fitness - prev_best) < 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                prev_best = self.best_fitness

            if stagnation_counter >= self.stagnation_patience:
                population, pop_fitness = self._elite_guided_restart(
                    population, pop_fitness
                )
                
                for idx in np.argsort(mem_fitness)[-self.restart_count:]:
                    mem_fitness[idx] = pop_fitness[idx]
                    memory[idx] = population[idx].copy()
                stagnation_counter = 0

            self.convergence_history.append(self.best_fitness)

            if (iteration + 1) % 20 == 0:
                print(f"  Iter {iteration+1:3d}/{self.max_iterations}: "
                      f"Best Fitness = {self.best_fitness:.6f}")

        self.execution_time = time.time() - start_time
        print(f"  Final Best Fitness: {self.best_fitness:.6f}")
        print(f"  Time: {self.execution_time:.2f}s")

        return self.get_results()

    def get_results(self) -> dict:
        """Return comprehensive results."""
        eval_results = self.env.evaluate_solution(self.best_solution)
        eval_results['convergence_history'] = self.convergence_history
        eval_results['execution_time'] = self.execution_time
        eval_results['algorithm'] = 'CEO'
        return eval_results




if __name__ == "__main__":
    from system_model import SimulationConfig

    config = SimulationConfig(num_tasks=50, num_edge_servers=5, seed=50)
    env = MECEnvironment(config)

    ceo = CEO(
        env, max_iterations=100, population_size=50, seed=50
    )
    res = ceo.optimize()

    print(f"\nFitness: {res['fitness']:.6f}")
    print(f"Total Delay: {res['total_delay']:.4f} s")
    print(f"TCR: {res['tcr']*100:.1f}%")
