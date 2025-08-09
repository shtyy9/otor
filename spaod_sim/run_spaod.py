"""Command line driver for SPAO-D path selection experiments.

The script loads a YAML configuration, builds a Starlink-like topology and
invokes the solver to pick a routing path between two example ground stations.
Multiple replications can be evaluated in parallel via ``--seeds`` to better
utilise available CPU cores.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import yaml

from spaod.solver import AlgoCfg, select_path_for_pair
from spaod.topology import LinkCfg, TEParams, TopologyBuilder


def _run_seed(seed: int, config: str, alg: str):
    """Execute the solver for a single replication.

    Parameters
    ----------
    seed : int
        Random seed controlling stochastic components of the solver.
    config : str
        Path to the YAML configuration file.
    alg : str
        Name of the routing algorithm. Only ``"spaod"`` is implemented in this
        skeleton and will raise ``NotImplementedError`` otherwise.

    Returns
    -------
    dict
        Summary dictionary returned by :func:`spaod.solver.select_path_for_pair`.
    """

    if alg != "spaod":
        raise NotImplementedError(f"Algorithm '{alg}' is not implemented")

    cfg_path = config if os.path.isabs(config) else os.path.join(os.path.dirname(__file__), config)
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Build topology according to config
    link_cfg = LinkCfg(
        isl_max_km=cfg["links"]["isl_max_km"],
        elev_min_deg=cfg["links"]["elev_min_deg"],
    )
    te_params = TEParams(
        dt=cfg["time"]["dt"],
        horizon_s=min(cfg["time"]["horizon_s"], 60),
        pat_mean_s=cfg["links"].get("pat_mean_s", 0.5),
    )
    const = cfg["constellation"]
    planes = min(const["planes"], 2)
    sats_per_plane = min(const["sats_per_plane"], 4)
    topo = TopologyBuilder(
        planes,
        sats_per_plane,
        const["inclination_deg"],
        const["altitude_m"],
        const.get("phase_factor", 1),
        link_cfg,
        te_params,
    )

    algo_cfg = AlgoCfg(
        K_paths=cfg["algo"].get("K_paths", 6),
        temp=cfg["algo"].get("temp", 0.6),
        alpha=cfg["algo"].get("alpha", 0.5),
        beta=cfg["algo"].get("beta", 1.0),
        gamma=cfg["algo"].get("gamma", 0.1),
        zeta=cfg["algo"].get("zeta", 0.5),
        risk_W_s=cfg["risk_window"].get("W_s", 30.0),
        risk_m_max=cfg["risk_window"].get("m_max", 2),
        risk_bound=cfg["risk_window"].get("bound", 0.2),
        tau_threshold=cfg["detector"].get("tau_threshold", 10.0),
        seed=seed,
    )

    ogs_path = os.path.join(os.path.dirname(__file__), "data", "ogs.csv")
    ogs_df = pd.read_csv(ogs_path)

    res = select_path_for_pair(ogs_df, 0, 1, topo, algo_cfg)
    risk = res.get("risk_upper", float("nan"))
    print(f"[Seed {seed}] chosen_idx={res['chosen_idx']} risk_upper={risk:.3f}")
    return res



def main():
    """Entry point for the command line interface."""
    parser = argparse.ArgumentParser(description="SPAO-D simulation skeleton")
    parser.add_argument("--config", type=str, default="configs/starlink_1584.yaml")
    parser.add_argument(
        "--alg", type=str, default="spaod", choices=["spaod", "baseline_shortest", "baseline_random"]
    )
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()


    # Run each seed in its own process to better utilise CPU cores.
    seeds = list(range(args.seeds))
    if len(seeds) > 1:
        with ProcessPoolExecutor() as exe:
            exe.map(_run_seed, seeds, [args.config]*len(seeds), [args.alg]*len(seeds))
    else:
        _run_seed(seeds[0], args.config, args.alg)


if __name__ == "__main__":
    main()
