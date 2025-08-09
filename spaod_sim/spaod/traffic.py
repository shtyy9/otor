# ====================== spaod/traffic.py ======================
# Traffic generation for OGS<->OGS under given load levels
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Dict

@dataclass
class Flow:
    flow_id: int
    start_s: float
    end_s: float
    src_g: int
    dst_g: int
    rate_mbps: float

def generate_ogs_pairs(ogs_df: pd.DataFrame, num_pairs:int, rng:np.random.Generator):
    n = len(ogs_df)
    src = rng.integers(0, n, size=num_pairs)
    dst = rng.integers(0, n, size=num_pairs)
    mask = src != dst
    return list(zip(src[mask], dst[mask]))

def generate_flows_for_load(ogs_df: pd.DataFrame, horizon_s: int, dt: float,
                            avg_arrival_per_s: float,
                            rate_range_mbps=(1.0, 40.0),
                            mean_duration_s: float = 100.0,
                            seed: int = 42) -> List[Flow]:
    """
    Simple Poisson arrivals. Each flow is OGS->OGS with IID duration & rate.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    flows: List[Flow] = []
    fid = 0
    while t < horizon_s:
        # Inter-arrival (Poisson) ~ Exp(λ=avg_arrival_per_s)
        delta = rng.exponential(1.0/max(1e-9, avg_arrival_per_s))
        t += delta
        if t >= horizon_s: break
        dur = max(dt, rng.exponential(mean_duration_s))
        end_t = min(horizon_s, t + dur)
        rate = rng.uniform(rate_range_mbps[0], rate_range_mbps[1])
        # pick a random src/dst ground pair
        src = rng.integers(0, len(ogs_df))
        dst = rng.integers(0, len(ogs_df))
        if src == dst:
            dst = (dst + 1) % len(ogs_df)
        flows.append(Flow(fid, float(t), float(end_t), int(src), int(dst), float(rate)))
        fid += 1
    return flows

def flows_by_load_levels(ogs_df: pd.DataFrame, horizon_s:int, dt:float,
                         load_levels, base_lambda: float = 70.0,
                         seed:int=42) -> Dict[float, List[Flow]]:
    """
    Create flows for each load in load_levels. base_lambda≈70/s at load=1.0 (可在配置里调小/调大)
    """
    out = {}
    for i, L in enumerate(load_levels):
        lam = max(0.0, float(L)) * base_lambda
        out[L] = generate_flows_for_load(ogs_df, horizon_s, dt, lam, seed=seed+i)
    return out
