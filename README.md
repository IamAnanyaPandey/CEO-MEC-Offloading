# CEO: Crow search based Efficient Offloading for Mobile Edge Computing

This repository contains the Python implementation of CEO (Crow search based Efficient Offloading), a discrete metaheuristic for bi-objective computational offloading in multi-server Mobile Edge Computing (MEC) environments.

> **Note:** The associated research paper is under review. Citation details will be added after publication.

## Overview

CEO extends the Crow Search Algorithm (CSA) to the discrete offloading problem with six components:

- Greedy and Opposition-Based Learning (OBL) initialization
- Discrete memory adherence
- Lévy-controlled exploration
- Elite local search
- Adaptive awareness probability (0.1 to 0.5)
- Stagnation-guided restart

The objective is a weighted sum of the normalized total delay and the task failure ratio (1 - TCR), with a penalty for violating the VM capacity of the edge servers.

## Repository contents

| File | Description |
|---|---|
| `system_model.py` | MEC system model: tasks, edge servers, SINR-based data rate, delay model, fitness function with capacity penalty |
| `algorithms.py` | Baselines: Random, GA, PSO, CSA, GA-PSO |
| `ceo.py` | Proposed CEO algorithm, including switches for the ablation variants |
| `experiments.py` | Experiment driver (E1 to E5) |
| `analysis.py` | Tables, statistical tests (Wilcoxon signed-rank with Holm adjustment) and figures from the result files |
| `plot_weights.py` | Figures for the weight sensitivity experiment |
| `results/` | Result files of the reported runs |

## Requirements

Python 3.9 or higher with NumPy, pandas, SciPy and Matplotlib.

```
pip install -r requirements.txt
```

## Usage

Run all experiments (30 instances per scenario, budget of 15,000 fitness evaluations per run):

```
python experiments.py --exp all --runs 30
```

Single experiments can be selected with `--exp E1`, `E2`, `E3`, `E4` or `E5`. A short test run with two instances is available with `--quick`. The number of parallel processes is set with `--workers`.

| Experiment | Content |
|---|---|
| E1 | Comparison with the baselines for 50, 100, 150 and 200 tasks and 5 servers |
| E2 | Ablation of the CEO components for 100 and 200 tasks |
| E3 | Scalability with up to 500 tasks and up to 20 servers |
| E4 | Sensitivity to the objective weights |
| E5 | Grid search of the CEO parameters on tuning instances |

Create the tables and figures from the result files:

```
python analysis.py
python plot_weights.py
```

The outputs are written to `results/analysis/`.

## Simulation parameters

| Parameter | Value |
|---|---|
| Bandwidth | 10 MHz |
| Noise power | -90 dBm |
| Channel gain | 2e-6 to 2e-5 |
| Task size | 300 to 1000 KB |
| CPU cycles per byte | 200 to 1000 |
| Edge server processing power | 40 to 50 GHz |
| Local device processing power | 100 to 500 MHz |
| VMs per server | U[40, 100] |
| Transmission power | 0.1 to 0.5 W |
| Deadline | 0.5 to 2.0 s |
| Weights | w1 = w2 = 0.5 |

All parameters can be modified in `SimulationConfig` in `system_model.py`.

## Algorithm parameters

All algorithms use a population size of 50 and a budget of 15,000 fitness evaluations.

| Algorithm | Parameters |
|---|---|
| GA | crossover 0.9, mutation 0.05, tournament 3, elite 2 |
| PSO | w from 0.9 to 0.4, c1 = c2 = 2.0 |
| CSA | AP = 0.1, fl = 2.0 |
| GA-PSO | w = 0.8, (c1, c2, c3) = (1.5, 2.5, 2.0), sigma = 0.01, count_max = 7, savior fraction 0.2 |
| CEO | AP from 0.1 to 0.5, Lévy beta = 1.5, greedy fraction 0.3, elite fraction 0.3, stagnation patience 5, restart fraction 0.3 |

The CEO parameters greedy fraction, elite fraction and stagnation patience were selected with experiment E5 on tuning instances (seeds 1001 to 1005), which are not used for the reported results.

## Reproducibility

The reported results use instance seeds 1 to 30. The software versions and the machine used for a run are written to `results/environment.txt`.

## License

MIT License. See the `LICENSE` file.