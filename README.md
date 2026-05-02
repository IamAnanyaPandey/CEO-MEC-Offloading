# CEO: Crow-inspired Efficient Offloading for Mobile Edge Computing

This repository contains the official Python implementation of the **Crow-inspired Efficient Offloading (CEO)** algorithm for bi-objective task offloading in multi-server Mobile Edge Computing (MEC) environments.

> **Note:** The associated research paper is currently under review. Full citation information, DOI, and links will be added once the paper is published.

## Overview

CEO is a discrete metaheuristic optimization algorithm that extends the classical Crow Search Algorithm (CSA) with four targeted enhancements designed for the combinatorial nature of MEC task offloading:

- **Greedy + Opposition-Based Learning (OBL) initialization** — seeds the population with high-quality starting solutions
- **Lévy-flight controlled exploration** — heavy-tailed random jumps to escape local optima
- **Elite local search** — focused refinement of top solutions every iteration
- **Elite-guided restart** — re-injects diversity when stagnation is detected

The algorithm jointly minimizes total task completion delay and maximizes the Task Completion Ratio (TCR) under deadline constraints, formulated as a weighted bi-objective fitness function with a VM capacity penalty.

## Repository Contents

| File | Description |
|------|-------------|
| `system_model.py` | MEC system model: Task and EdgeServer data classes, SINR-based data rate, transmission and processing delay, fitness function with capacity penalty |
| `algorithms.py` | Baseline algorithms: Random Offloading, Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Crow Search Algorithm (CSA) |
| `ceo.py` | Proposed CEO algorithm |
| `experiments.py` | Experiment driver: runs all five algorithms across S1–S4 (50 / 100 / 150 / 200 tasks) with 5 fixed servers; generates convergence and bar plots |

## Requirements

- Python 3.9 or higher
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Clone the repository and run the main experiment:

```bash
git clone https://github.com/IamAnanyaPandey/CEO-MEC-Offloading.git
cd CEO-MEC-Offloading
pip install -r requirements.txt
python experiments.py
```

This runs the full task scalability experiment (50, 100, 150, 200 tasks with 5 fixed edge servers) and saves three figures to `results/`:

- `fig1_convergence.png` — Convergence comparison across S1–S4
- `fig2a_total_delay.png` — Total delay grouped bar chart
- `fig2b_tcr.png` — Task Completion Ratio grouped bar chart

A summary table is also printed to the console for each scenario.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Population size | 50 |
| Maximum iterations | 100 |
| Bandwidth | 10 MHz |
| Noise power | -90 dBm |
| Edge server CPU range | 3–5 GHz |
| Local CPU range | 100–500 MHz |
| Task data size | 300–1000 KB |
| CPU cycles per byte | 200–1000 |
| Transmit power | 0.1–0.5 W |
| Deadlines | 0.5–2.0 s |
| VMs per server | U[40, 100] |
| Fitness weights | w₁ = w₂ = 0.5 |
| Random seed | 50 |

All parameters can be modified in the `SimulationConfig` class in `system_model.py`.

## Algorithms and Hyperparameters

| Algorithm | Key parameters |
|-----------|----------------|
| Random | num_trials = pop_size × max_iter |
| GA | crossover = 0.9, mutation = 0.05, tournament = 3, elite = 2 |
| PSO | w: 0.9 → 0.4 (linearly decreasing), c1 = c2 = 2.0 |
| CSA (Askarzadeh, 2016) | AP = 0.1 (fixed), fl = 2.0 (fixed) |
| **CEO (proposed)** | AP = 0.1 → 0.5 (adaptive), Lévy β = 1.5, greedy = 20%, elite = 30%, patience = 8, restart = 30% |

## Reproducibility

All experiments use a fixed random seed (`seed=50`). Running `python experiments.py` reproduces the exact figures and tables presented in the manuscript.

## Citation

A formal citation will be added once the paper is published. Until then, if you use this code, please cite this repository directly:

```bibtex
@misc{ceo_mec_2026,
  author = {Ananya Pandey},
  title  = {CEO: Crow-inspired Efficient Offloading for Mobile Edge Computing},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/IamAnanyaPandey/CEO-MEC-Offloading}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
