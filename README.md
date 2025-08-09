# SPAO-D Simulation Skeleton

This repository provides a small but documented environment for experimenting
with path selection strategies in a Starlink-like low-Earth orbit laser network.
It accompanies a set of teaching materials on secure routing and serves as a
reference implementation for the SPAO-D (Secure Path Availability and
Obfuscation - Dynamic) solver.

## Layout

- `spaod_sim/spaod/solver.py` – path selection logic with latency/privacy/risk
  scoring and Sinkhorn-based soft choice.
- `spaod_sim/run_spaod.py` – command line driver that executes the solver for
  one or more random seeds, parallelising replications with multiprocessing.
- `spaod_sim/configs/` – example configuration files.
- `spaod_sim/data/` – ground station table used by the examples.

## Usage

Install Python requirements:

```bash
pip install -r spaod_sim/requirements.txt
```

Run a few seeds of the example scenario:

```bash
python spaod_sim/run_spaod.py --seeds 4
```

Each seed runs in its own process and prints a short summary of the selected
path and risk assessment.

The project is intentionally lightweight to keep the routing logic accessible.
Contributions that improve documentation or extend the simulation features are
welcome.

