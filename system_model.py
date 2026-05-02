import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Data Classes for Tasks and Servers
@dataclass
class Task:
    task_id: int
    data_size: float          # Di: input data size in bytes
    cpu_cycles_per_byte: float  # Ci: cycles/byte
    deadline: float           # φi: deadline in seconds
    device_processing_power: float  # C_li: local CPU freq in cycles/sec
    transmit_power: float     # pi: transmission power in Watts
    channel_gains: np.ndarray = field(default_factory=lambda: np.array([])) # hi,j for each edge server


@dataclass
class EdgeServer:
    server_id: int
    processing_power: float   # F(ej): processing capacity in cycles/sec
    num_vms: int              # Kj: number of VMs

# Simulation Parameters
class SimulationConfig:
    def __init__(
        self,
        num_tasks: int = 50,
        num_edge_servers: int = 5,
        bandwidth: float = 10e6,          # 10 MHz
        noise_power_dbm: float = -90,     # -90 dBm
        channel_gain_range: Tuple = (2e-6, 2e-5),
        edge_processing_range: Tuple = (3e9, 5e9),     # 3-5 GHz
        local_processing_range: Tuple = (100e6, 500e6), # 100-500 MHz
        task_size_range: Tuple = (300e3, 1000e3),       # 300-1000 KB in bytes
        cpu_cycles_range: Tuple = (200, 1000),          # 200-1000 cycles/byte
        transmit_power_range: Tuple = (0.1, 0.5),       # 0.1-0.5 W
        deadline_range: Tuple = (0.5, 2.0),             # 0.5-2 seconds
        vm_range: Tuple = (40, 100),      # U[40,100] VMs per server
        w1: float = 0.5,                  # weight for delay
        w2: float = 0.5,                  # weight for task failure ratio
        seed: Optional[int] = None
    ):
        self.num_tasks = num_tasks
        self.num_edge_servers = num_edge_servers
        self.bandwidth = bandwidth
        self.noise_power = 10 ** (noise_power_dbm / 10) * 1e-3  # Convert dBm to Watts
        self.channel_gain_range = channel_gain_range
        self.edge_processing_range = edge_processing_range
        self.local_processing_range = local_processing_range
        self.task_size_range = task_size_range
        self.cpu_cycles_range = cpu_cycles_range
        self.transmit_power_range = transmit_power_range
        self.deadline_range = deadline_range
        self.vm_range = vm_range
        self.w1 = w1
        self.w2 = w2
        self.seed = seed

# MEC Environment
class MECEnvironment:
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        self.tasks: List[Task] = []
        self.servers: List[EdgeServer] = []
        self._generate_environment()
    
    def _generate_environment(self):
        """Generate tasks and edge servers with random parameters."""
        cfg = self.config
        
        self.servers = []
        for j in range(cfg.num_edge_servers):
            num_vms = self.rng.randint(cfg.vm_range[0], cfg.vm_range[1] + 1)
            server = EdgeServer(
                server_id=j,
                processing_power=self.rng.uniform(*cfg.edge_processing_range),
                num_vms=num_vms
            )
            self.servers.append(server)
        
        self.tasks = []
        for i in range(cfg.num_tasks):
            channel_gains = self.rng.uniform(
                *cfg.channel_gain_range, size=cfg.num_edge_servers
            )
            task = Task(
                task_id=i,
                data_size=self.rng.uniform(*cfg.task_size_range),
                cpu_cycles_per_byte=self.rng.uniform(*cfg.cpu_cycles_range),
                deadline=self.rng.uniform(*cfg.deadline_range),
                device_processing_power=self.rng.uniform(*cfg.local_processing_range),
                transmit_power=self.rng.uniform(*cfg.transmit_power_range),
                channel_gains=channel_gains
            )
            self.tasks.append(task)
    
    def compute_data_rate(self, task_idx: int, server_idx: int,
                          offloading_decisions: np.ndarray) -> float:
        
        task = self.tasks[task_idx]
        bandwidth = self.config.bandwidth
        noise = self.config.noise_power
        
        # Signal power
        signal_power = task.transmit_power * task.channel_gains[server_idx]
        
        # Interference 
        interference = 0.0
        for i_prime in range(self.config.num_tasks):
            if i_prime != task_idx and offloading_decisions[i_prime] == (server_idx + 1):
                other_task = self.tasks[i_prime]
                interference += (other_task.transmit_power *
                                 other_task.channel_gains[server_idx])
        
        # SINR and data rate
        sinr = signal_power / (noise + interference)
        data_rate = bandwidth * np.log2(1 + sinr)
        
        return data_rate
    
    def compute_transmission_delay(self, task_idx: int, server_idx: int,
                                    offloading_decisions: np.ndarray) -> float:
        
        if server_idx == 0:  # Local execution
            return 0.0
        
        actual_server = server_idx - 1
        data_rate = self.compute_data_rate(task_idx, actual_server,
                                           offloading_decisions)
        
        if data_rate <= 0:
            return float('inf')
        
        task = self.tasks[task_idx]
        return task.data_size / data_rate
    
    def compute_processing_delay(self, task_idx: int, server_idx: int,
                                  offloading_decisions: np.ndarray) -> float:
        
        task = self.tasks[task_idx]
        total_cycles = task.data_size * task.cpu_cycles_per_byte
        
        if server_idx == 0:  
            return total_cycles / task.device_processing_power
        
        actual_server = server_idx - 1
        server = self.servers[actual_server]
        
        K_j = server.num_vms
        vm_power = server.processing_power / K_j
        
        return total_cycles / vm_power
    
    def compute_task_delay(self, task_idx: int,
                           offloading_decisions: np.ndarray) -> float:
        
        server_idx = int(offloading_decisions[task_idx])
        
        t_trans = self.compute_transmission_delay(
            task_idx, server_idx, offloading_decisions
        )
        t_proc = self.compute_processing_delay(
            task_idx, server_idx, offloading_decisions
        )
        
        return t_trans + t_proc
    
    def compute_total_delay(self, offloading_decisions: np.ndarray) -> float:
        
        total = 0.0
        for i in range(self.config.num_tasks):
            total += self.compute_task_delay(i, offloading_decisions)
        return total
    
    def compute_task_completion_indicator(self, task_idx: int,
                                          offloading_decisions: np.ndarray) -> int:
        delay = self.compute_task_delay(task_idx, offloading_decisions)
        return 1 if delay <= self.tasks[task_idx].deadline else 0
    
    def compute_tcr(self, offloading_decisions: np.ndarray) -> float:
        """Compute Task Completion Ratio (Eq. 12): TCR = (1/m) * Σ δ_i"""
        completed = sum(
            self.compute_task_completion_indicator(i, offloading_decisions)
            for i in range(self.config.num_tasks)
        )
        return completed / self.config.num_tasks
    
    def compute_capacity_penalty(self, offloading_decisions: np.ndarray) -> float:
        """Enforce constraint (14c): tasks assigned to server j must not exceed Kj VMs."""
        penalty = 0.0
        for j in range(self.config.num_edge_servers):
            tasks_on_server = int(np.sum(offloading_decisions == (j + 1)))
            K_j = self.servers[j].num_vms
            if tasks_on_server > K_j:
                # Penalty = excess tasks * large constant
                penalty += (tasks_on_server - K_j) * 10.0
        return penalty
    
    def fitness(self, offloading_decisions: np.ndarray) -> float:
        """Compute fitness (Eq. 13) with capacity penalty for constraint (14c)."""
        total_delay = self.compute_total_delay(offloading_decisions)
        tcr = self.compute_tcr(offloading_decisions)
        penalty = self.compute_capacity_penalty(offloading_decisions)
        
        obj = self.config.w1 * total_delay + self.config.w2 * (1 - tcr) + penalty
        
        return obj
    
    def evaluate_solution(self, offloading_decisions: np.ndarray) -> dict:
        total_delay = self.compute_total_delay(offloading_decisions)
        tcr = self.compute_tcr(offloading_decisions)
        fitness_val = self.fitness(offloading_decisions)
        penalty = self.compute_capacity_penalty(offloading_decisions)
        
        # Per-task delays
        task_delays = [
            self.compute_task_delay(i, offloading_decisions)
            for i in range(self.config.num_tasks)
        ]
        
        # Offloading distribution
        local_count = np.sum(offloading_decisions == 0)
        edge_counts = [
            np.sum(offloading_decisions == (j + 1))
            for j in range(self.config.num_edge_servers)
        ]
        
        return {
            'fitness': fitness_val,
            'total_delay': total_delay,
            'tcr': tcr,
            'penalty': penalty,
            'task_delays': task_delays,
            'local_tasks': int(local_count),
            'edge_task_distribution': edge_counts,
            'offloading_decisions': offloading_decisions.copy()
        }
    
    def generate_random_solution(self) -> np.ndarray:
        """Generate random offloading decision vector.
        Encoding: OD[i] = 0 means local execution,
                  OD[i] = j (1..n) means offload to server j.
        Equivalent to the binary matrix OD_ij in the paper.
        """
        return self.rng.randint(
            0, self.config.num_edge_servers + 1,
            size=self.config.num_tasks
        )
    
    def get_solution_space_size(self) -> int:
        return self.config.num_edge_servers + 1

# This is a Quick Test

if __name__ == "__main__":
    config = SimulationConfig(num_tasks=50, num_edge_servers=5, seed=50)
    env = MECEnvironment(config)
    
    print(f"Generated {len(env.tasks)} tasks and {len(env.servers)} edge servers")
    print(f"VM range: U{list(config.vm_range)}")
    
    total_vms = 0
    for s in env.servers:
        vm_power = s.processing_power / s.num_vms
        total_vms += s.num_vms
        print(f"  Server {s.server_id}: {s.processing_power/1e9:.2f} GHz, "
              f"{s.num_vms} VMs, "
              f"µ_j = {vm_power/1e9:.4f} GHz per VM")
    
    print(f"  Total VMs: {total_vms}")
    print(f"  Total VMs >= Tasks? {total_vms} >= {config.num_tasks}: "
          f"{'YES' if total_vms >= config.num_tasks else 'NO'}")
    print(f"Solution space per task: {env.get_solution_space_size()} options")
    
    # Test with random solution
    solution = env.generate_random_solution()
    results = env.evaluate_solution(solution)
    
    print(f"\nRandom Solution Test:")
    print(f"  Offloading: {solution}")
    print(f"  Fitness: {results['fitness']:.4f}")
    print(f"  Total Delay: {results['total_delay']:.4f} s")
    print(f"  TCR: {results['tcr']:.2%}")
    print(f"  Penalty: {results['penalty']:.2f}")
    print(f"  Local tasks: {results['local_tasks']}")
    print(f"  Edge distribution: {results['edge_task_distribution']}")
