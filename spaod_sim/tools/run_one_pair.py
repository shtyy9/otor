# Minimal runnable script: load YAML & CSV, build topology, pick a path for one OGS pair
import argparse, json, yaml, os, sys
import pandas as pd

# ensure parent dir on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spaod.topology import TopologyBuilder, LinkCfg, TEParams
from spaod.solver import AlgoCfg, select_path_for_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/starlink_1584.yaml")
    ap.add_argument("--ogs", default="data/ogs.csv")
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--dst", type=int, default=5)
    ap.add_argument("--out", default="out/choice.json")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # topology
    P = cfg["constellation"]["planes"]
    S = cfg["constellation"]["sats_per_plane"]
    inc = cfg["constellation"]["inclination_deg"]
    alt = cfg["constellation"]["altitude_m"]
    pf  = cfg["constellation"]["phase_factor"]

    link_cfg = LinkCfg(isl_max_km=cfg["links"]["isl_max_km"], elev_min_deg=cfg["links"]["elev_min_deg"])
    te = TEParams(dt=cfg["time"]["dt"], horizon_s=cfg["time"]["horizon_s"], pat_mean_s=cfg["links"]["pat_mean_s"])
    topo = TopologyBuilder(P, S, inc, alt, pf, link_cfg, te)

    # algo
    algo = AlgoCfg(
        K_paths=cfg["algo"]["K_paths"],
        temp=cfg["algo"].get("temp", 0.6),
        alpha=cfg["algo"]["alpha"],
        beta =cfg["algo"]["beta"],
        gamma=cfg["algo"]["gamma"],
        zeta =cfg["algo"].get("zeta", 0.5),
        risk_W_s=cfg["risk_window"]["W_s"],
        risk_m_max=cfg["risk_window"]["m_max"],
        risk_bound=cfg["risk_window"].get("bound", 0.2),
        tau_threshold=cfg["detector"]["tau_threshold"],
    )

    ogs = pd.read_csv(args.ogs)
    res = select_path_for_pair(ogs, args.src, args.dst, topo, algo,
                               det_csv_path="data/single_node_det.csv",
                               adversary=None)
    os.makedirs("out", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
