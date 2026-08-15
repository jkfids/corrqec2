# corrqec2

Code and data for *Spatiotemporal Pauli processes: Quantum combs for modeling correlated noise in quantum error correction*, by John F. Kam, Angus Southwell, Spiro Gicev, Muhammad Usman, and Kavan Modi.

## Links and abstract

- **DOI** – https://doi.org/10.48550/arXiv.2603.05474
- **arXiv** – [arXiv:2603.05474](https://arxiv.org/abs/2603.05474)

**Abstract**
> Correlated noise is a critical failure mode in quantum error correction (QEC), yet a gap remains between the stochastic Pauli models used for scalable QEC analyses and the microscopic, non-Markovian descriptions of noise in physical devices. We bridge this gap by introducing *Spatiotemporal Pauli Processes* (SPPs): the natural multi-time generalization of Pauli channels—joint probability distributions over Pauli faults across space and time—obtained exactly from any noise process, however non-Markovian, under standard Pauli-frame randomization. SPPs thereby provide a single language in which correlated noise can be recast, compared, and extended, compatible with both microscopic open systems modeling and the stabilizer workflow of QEC design and analysis. Our central result is constructive: the multi-time Pauli twirl acts by local contractions on a process tensor network, yielding an explicit classical tensor network whose virtual bonds encode memory, bounded by the environment's Liouville-space dimension. Transfer operator diagnostics link memory spectra to correlation decay, and hidden Markov representations enable efficient sampling of correlated Pauli fault trajectories—mapping microscopically derived noise directly into circuit-level QEC simulation. We demonstrate this with surface code memory and stability benchmarks up to distance $19$. A temporal "storm" model with tunable correlation time shows that memory alone, at strictly fixed marginal error rates, systematically erodes the exponential error suppression expected from code distance. A genuinely spatiotemporal quantum cellular automaton bath maps exactly, under system twirling, to a nonlinear probabilistic cellular automaton; tuning its coherent interactions drives the noise into a pseudo-critical regime with critical slowing down and macroscopic error avalanches that reverse surface code distance scaling. These results establish that below-threshold average error rates alone do not guarantee scalable error suppression, and position SPPs as a practical bridge from microscopic correlated dynamics to circuit-level QEC.

## Repository overview

- **`src/`** – The `corrqec2` package: noise models, experiments, sampling, and decoding.
- **`scripts/`** – Scripts that generate the paper's results and figures.
- **`data/`** – Simulation results needed to regenerate the figures.
- **`slurm/`** – SLURM batch scripts for the full sweeps on a cluster.
- **`manuscript/`** – LaTeX source and figures.

## Reproduce

**Clone**
```bash
git clone https://github.com/jkfids/corrqec2.git
cd corrqec2
```

**Install** (Python 3.12 or newer, from the project root)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This also installs [`processtensor`](https://github.com/jkfids/process-tensor)
from GitHub — see [below](#processtensor-dependency).

**Run**

Run scripts from the project root, adjusting the `DATA_DIR`, `FIGURE_DIR`, and
`RESULTS_DIR` path constants at the top of each file as needed.

Regenerate the experiment 1 figure from the official data:
```bash
python scripts/experiment1_figures.py
```

Run the experiment 1 Monte Carlo simulations locally:
```bash
# Increase the shot count to reduce statistical uncertainty.
python scripts/experiment1.py --num-workers 4 --max-shots 10000 --batch-size 1000
```

The other experiments follow the same pattern. Full production sweeps are
compute-heavy and were run on Monash's
[M3 (MASSIVE)](https://docs.erc.monash.edu/Compute/HPC/M3/) cluster; the SLURM
scripts under `slurm/` record the parameters used for the paper.

## `processtensor` dependency

`scripts/worked_examples.py` relies on an external package,
[`processtensor`](https://github.com/jkfids/process-tensor), for the process
tensor calculations, including construction and computing measures
such as the quantum relative entropy. It is not on PyPI, so it is installed
from GitHub and pinned to the commit used for the paper's figures.

## License

MIT – see [`LICENSE`](LICENSE).

The stability circuit construction under
`src/corrqec2/experiments/_stability_builder/` is adapted from the code
accompanying C. Gidney, *Stability Experiments: The Overlooked Dual of Memory
Experiments*, [Quantum **6**, 786 (2022)](https://doi.org/10.22331/q-2022-08-24-786),
archived at [Zenodo](https://doi.org/10.5281/zenodo.6859486). Copyright (c) 2022
Craig Gidney, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
and modified for use here; provided "as is", without warranties or conditions of
any kind.

## Contact

**Email** – john.kam@monash.edu
