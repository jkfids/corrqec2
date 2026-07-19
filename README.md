# corrqec2
> Preprint version. Potentially out of date!

Repository containing code and data for *Spatiotemporal Pauli process: Quantum combs for modelling correlated noise in quantum error correction*.

## Links and abstract

- **DOI** – 
https://doi.org/10.48550/arXiv.2603.05474
- **arXiv** – [arXiv:2603.05474v2](https://arxiv.org/abs/2603.05474v2)

**Abstract**
> Correlated noise is a critical failure mode in quantum error correction (QEC), as temporal memory and spatial structure concentrate faults into error bursts that undermine standard threshold assumptions. Yet, a fundamental gap persists between the stochastic Pauli models ubiquitous in QEC and the microscopic, non-Markovian descriptions of physical device dynamics. We close this gap by introducing *Spatiotemporal Pauli Processes* (SPPs). By applying a multi-time Pauli twirl–operationally realised by Pauli-frame randomisation–to a general process tensor, we map arbitrary multi-time, non-Markovian dynamics to a multi-time Pauli process. This process is represented by a process-separable comb, or equivalently, a well-defined joint probability distribution over Pauli trajectories in spacetime. We show that SPPs inherit efficient tensor network representations whose bond dimensions are bounded by the environment's Liouville-space dimension. To interpret these structures, we develop transfer operator diagnostics linking spectra to correlation decay, and exact hidden Markov representations for suitable classes of SPPs. We demonstrate the framework via surface code memory and stability simulations of up to distance $19$ for (i) a temporally correlated "storm" model that tunes correlation length at fixed marginal error rates, and (ii) a genuinely spatiotemporal 2D quantum cellular automaton bath that maps exactly to a nonlinear probabilistic cellular automaton under twirling. Tuning coherent bath interactions drives the system into a pseudo-critical regime, exhibiting critical slowing down and macroscopic error avalanches that cause a complete breakdown of surface code distance scaling.

## Repository overview-

- **`src/`** – Source code.
- **`data/`** – Input data for reproducing results.
- **`scripts/`** – Scripts for generating the paper's results and figures.
- **`slurm/`** – SLURM batch scripts for running the full sweeps on a cluster.
- **`project/paper`** – LaTeX source files.

## Reproduce

**Clone**
```bash
git clone https://github.com/jkfids/corrqec2.git
cd corrqec2
```

**Install** (in project root)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .  # For devs:  pip install -e ".[dev]"
```

**Run**

Example: locally run experiment 1 Monte Carlo simulations (may need to modify filepaths).

```bash
# Increase number of shots for reduced statistical uncertainty.
python scripts/experiment1.py --num-workers 4 --max-shots 10000 --batch-size 1000
```

Generate experiment 1 plots from included or generated data.
```bash
python scripts/experiment1_figures.py
```

The other experiments follow the same pattern. Full production sweeps are compute-heavy;
the SLURM scripts under `slurm/` show the parameters used for the paper.

## Contact

**Email** – john.kam@monash.edu