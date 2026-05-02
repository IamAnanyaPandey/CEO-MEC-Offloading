import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from system_model import MECEnvironment

class BaseOptimizer:
    
    def __init__(self, env: MECEnvironment, max_iterations: int = 100,
                 population_size: int = 50, seed: Optional[int] = None):
        self.env = env
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.rng = np.random.RandomState(seed)
        
        self.num_tasks = env.config.num_tasks
        self.num_options = env.get_solution_space_size()  # 0 to n_servers
        
        # Tracking
        self.convergence_history: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        self.execution_time: float = 0.0
    
    def _random_solution(self) -> np.ndarray:
        """Generate random offloading decision vector.
        Encoding: OD[i] = 0 means local execution,
                  OD[i] = j (1..n) means offload to server j.
        Equivalent to the binary matrix OD_ij in the paper.
        """
        return self.rng.randint(0, self.num_options, size=self.num_tasks)
    
    def _evaluate(self, solution: np.ndarray) -> float:
        return self.env.fitness(solution)
    
    def _clip_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(np.round(solution).astype(int), 0, self.num_options - 1)
    
    def optimize(self) -> dict:
        raise NotImplementedError
    
    def get_results(self) -> dict:
        
        eval_results = self.env.evaluate_solution(self.best_solution)
        eval_results['convergence_history'] = self.convergence_history
        eval_results['execution_time'] = self.execution_time
        eval_results['algorithm'] = self.__class__.__name__
        return eval_results


# 1. RANDOM OFFLOADING

class RandomOffloading(BaseOptimizer):
    
    def __init__(self, env: MECEnvironment, num_trials: int = 1000,
                 seed: Optional[int] = None):
        super().__init__(env, max_iterations=1, population_size=num_trials, seed=seed)
        self.num_trials = num_trials
    
    def optimize(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  RANDOM OFFLOADING (Trials: {self.num_trials})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        best_fitness = float('inf')
        best_solution = None
        fitness_history = []
        
        for trial in range(self.num_trials):
            solution = self._random_solution()
            fitness = self._evaluate(solution)
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()
            
            # Record best fitness at regular intervals
            if (trial + 1) % (self.num_trials // min(100, self.num_trials)) == 0:
                fitness_history.append(best_fitness)
        
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.convergence_history = fitness_history
        self.execution_time = time.time() - start_time
        
        print(f"  Best Fitness: {self.best_fitness:.6f}")
        print(f"  Time: {self.execution_time:.2f}s")
        
        return self.get_results()


# 2. GENETIC ALGORITHM (GA) — Li and Zhu [9], 2020
#    Parameters from Table III:
#    N=50, Tmax=100, Crossover=0.9, Mutation=0.05, Tournament=3, Elite=2

class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, env: MECEnvironment, max_iterations: int = 100,
                 population_size: int = 50, crossover_rate: float = 0.9,
                 mutation_rate: float = 0.05, tournament_size: int = 3,
                 elite_count: int = 2, seed: Optional[int] = None):
        super().__init__(env, max_iterations, population_size, seed)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population. Shape: (pop_size, num_tasks)"""
        return np.array([self._random_solution()
                         for _ in range(self.population_size)])
    
    def _tournament_selection(self, population: np.ndarray,
                               fitness_values: np.ndarray) -> np.ndarray:
        """Select parent via tournament selection."""
        candidates = self.rng.choice(
            self.population_size, size=self.tournament_size, replace=False
        )
        best_candidate = candidates[np.argmin(fitness_values[candidates])]
        return population[best_candidate].copy()
    
    def _uniform_crossover(self, parent1: np.ndarray,
                            parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        mask = self.rng.random(self.num_tasks) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutant = individual.copy()
        for i in range(self.num_tasks):
            if self.rng.random() < self.mutation_rate:
                mutant[i] = self.rng.randint(0, self.num_options)
        return mutant
    
    def optimize(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  GENETIC ALGORITHM")
        print(f"  Pop: {self.population_size} | Gens: {self.max_iterations}")
        print(f"  Crossover: {self.crossover_rate} | Mutation: {self.mutation_rate}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        population = self._initialize_population()
        fitness_values = np.array([self._evaluate(ind) for ind in population])
       
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        
        for gen in range(self.max_iterations):
            new_population = []
            
            # Elitism - preserve best individuals
            elite_indices = np.argsort(fitness_values)[:self.elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Crossover
                child1, child2 = self._uniform_crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Evaluate new population
            population = np.array(new_population[:self.population_size])
            fitness_values = np.array([self._evaluate(ind) for ind in population])
            
            # Update global best
            gen_best_idx = np.argmin(fitness_values)
            if fitness_values[gen_best_idx] < self.best_fitness:
                self.best_fitness = fitness_values[gen_best_idx]
                self.best_solution = population[gen_best_idx].copy()
            
            self.convergence_history.append(self.best_fitness)
            
            if (gen + 1) % 20 == 0:
                print(f"  Gen {gen+1:3d}/{self.max_iterations}: "
                      f"Best Fitness = {self.best_fitness:.6f}")
        
        self.execution_time = time.time() - start_time
        print(f"  Final Best Fitness: {self.best_fitness:.6f}")
        print(f"  Time: {self.execution_time:.2f}s")
        
        return self.get_results()


# 3. PARTICLE SWARM OPTIMIZATION (PSO) — You and Tang [19], 2021
#    Parameters from Table III:
#    N=50, Tmax=100, w=0.9→0.4, c1=2.0, c2=2.0

class ParticleSwarmOptimization(BaseOptimizer):
    
    def __init__(self, env: MECEnvironment, max_iterations: int = 100,
                 population_size: int = 50, w_start: float = 0.9,
                 w_end: float = 0.4, c1: float = 2.0, c2: float = 2.0,
                 seed: Optional[int] = None):
        super().__init__(env, max_iterations, population_size, seed)
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = (self.num_options - 1) / 2
    
    def optimize(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  PARTICLE SWARM OPTIMIZATION")
        print(f"  Swarm: {self.population_size} | Iters: {self.max_iterations}")
        print(f"  w: {self.w_start}→{self.w_end} | c1: {self.c1} | c2: {self.c2}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Initialize particles (continuous, discretized for evaluation)
        positions = self.rng.uniform(
            0, self.num_options - 1,
            size=(self.population_size, self.num_tasks)
        )
        
        velocities = self.rng.uniform(
            -self.v_max, self.v_max,
            size=(self.population_size, self.num_tasks)
        )
        
        # Personal bests
        pbest_positions = positions.copy()
        pbest_fitness = np.full(self.population_size, float('inf'))
        
        # Global best
        gbest_position = None
        gbest_fitness = float('inf')
        
        # Evaluate initial positions
        for i in range(self.population_size):
            discrete_pos = self._clip_solution(positions[i])
            fit = self._evaluate(discrete_pos)
            pbest_fitness[i] = fit
            pbest_positions[i] = positions[i].copy()
            
            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_position = positions[i].copy()
        
        self.best_fitness = gbest_fitness
        self.best_solution = self._clip_solution(gbest_position)
        
        # Iterative optimization
        for iteration in range(self.max_iterations):
            # Linearly decreasing inertia weight
            w = self.w_start - (self.w_start - self.w_end) * (
                iteration / self.max_iterations
            )
            
            for i in range(self.population_size):
                r1 = self.rng.random(self.num_tasks)
                r2 = self.rng.random(self.num_tasks)
                
                # Update velocity
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Boundary handling
                positions[i] = np.clip(positions[i], 0, self.num_options - 1)
                
                # Evaluate
                discrete_pos = self._clip_solution(positions[i])
                fit = self._evaluate(discrete_pos)
                
                # Update personal best
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest_positions[i] = positions[i].copy()
                
                # Update global best
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = positions[i].copy()
            
            self.best_fitness = gbest_fitness
            self.best_solution = self._clip_solution(gbest_position)
            self.convergence_history.append(self.best_fitness)
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iter {iteration+1:3d}/{self.max_iterations}: "
                      f"Best Fitness = {self.best_fitness:.6f}")
        
        self.execution_time = time.time() - start_time
        print(f"  Final Best Fitness: {self.best_fitness:.6f}")
        print(f"  Time: {self.execution_time:.2f}s")
        
        return self.get_results()


# 4. CROW SEARCH ALGORITHM (CSA) — Askarzadeh [2], 2016
#    Standard CSA with FIXED AP and FIXED fl as per original paper.
#    Parameters from Table III: AP=0.1, fl=2.0 (both constant throughout)
#    Operates in continuous space with rounding — this is the standard
#    approach; CEO's paper (Section V-A) explicitly critiques this limitation.

class CrowSearchAlgorithm(BaseOptimizer):
    
    def __init__(self, env: MECEnvironment, max_iterations: int = 100,
                 population_size: int = 50, awareness_prob: float = 0.1,
                 flight_length: float = 2.0, seed: Optional[int] = None):
        super().__init__(env, max_iterations, population_size, seed)
        self.AP = awareness_prob    # Fixed AP as per Askarzadeh (2016)
        self.fl = flight_length     # Fixed fl as per Askarzadeh (2016)
    
    def optimize(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  CROW SEARCH ALGORITHM (Standard Askarzadeh 2016)")
        print(f"  Flock: {self.population_size} | Iters: {self.max_iterations}")
        print(f"  AP: {self.AP} (fixed) | FL: {self.fl} (fixed)")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Initialize crow positions (continuous representation)
        positions = self.rng.uniform(
            0, self.num_options - 1,
            size=(self.population_size, self.num_tasks)
        )
        
        # Initialize memory (best known positions)
        memory = positions.copy()
        memory_fitness = np.full(self.population_size, float('inf'))
        
        for i in range(self.population_size):
            discrete_pos = self._clip_solution(positions[i])
            fit = self._evaluate(discrete_pos)
            memory_fitness[i] = fit
        
        # Global best
        best_idx = np.argmin(memory_fitness)
        self.best_fitness = memory_fitness[best_idx]
        self.best_solution = self._clip_solution(memory[best_idx])
        
        # Iterative optimization
        for iteration in range(self.max_iterations):
            new_positions = np.zeros_like(positions)
            
            for i in range(self.population_size):
                # Randomly select a crow j to follow
                j = self.rng.randint(0, self.population_size)
                while j == i:
                    j = self.rng.randint(0, self.population_size)
                
                r_j = self.rng.random()  # Awareness check
                r_i = self.rng.random()  # Step size
                
                if r_j >= self.AP:
                    # Crow j doesn't know it's being followed
                    # Standard CSA position update (Eq. 2 in Askarzadeh 2016):
                    # x_i^(t+1) = x_i^t + r_i * fl * (m_j^t - x_i^t)
                    new_positions[i] = (
                        positions[i] +
                        r_i * self.fl * (memory[j] - positions[i])
                    )
                else:
                    # Crow j is aware — random exploration
                    new_positions[i] = self.rng.uniform(
                        0, self.num_options - 1, size=self.num_tasks
                    )
                
                # Boundary handling
                new_positions[i] = np.clip(
                    new_positions[i], 0, self.num_options - 1
                )
            
            # Evaluate new positions and update memory
            positions = new_positions
            
            for i in range(self.population_size):
                discrete_pos = self._clip_solution(positions[i])
                fit = self._evaluate(discrete_pos)
                
                # Update memory if new position is better
                if fit < memory_fitness[i]:
                    memory[i] = positions[i].copy()
                    memory_fitness[i] = fit
                
                # Update global best
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = discrete_pos.copy()
            
            self.convergence_history.append(self.best_fitness)
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iter {iteration+1:3d}/{self.max_iterations}: "
                      f"Best Fitness = {self.best_fitness:.6f}")
        
        self.execution_time = time.time() - start_time
        print(f"  Final Best Fitness: {self.best_fitness:.6f}")
        print(f"  Time: {self.execution_time:.2f}s")
        
        return self.get_results()


if __name__ == "__main__":
    from system_model import SimulationConfig
    
    config = SimulationConfig(num_tasks=20, num_edge_servers=3, seed=50)
    env = MECEnvironment(config)
    
    print("Testing all algorithms on small instance (20 tasks, 3 servers)...\n")
    
    # Random
    rand = RandomOffloading(env, num_trials=100, seed=50)
    rand.optimize()
    
    # GA
    ga = GeneticAlgorithm(env, max_iterations=100, population_size=50, seed=50)
    ga.optimize()
    
    # PSO
    pso = ParticleSwarmOptimization(env, max_iterations=100, population_size=50, seed=50)
    pso.optimize()
    
    # CSA
    csa = CrowSearchAlgorithm(env, max_iterations=100, population_size=50, seed=50)
    csa.optimize()
