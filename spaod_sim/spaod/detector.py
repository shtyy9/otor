# ====================== spaod/detector.py ======================
# Single-node sequential detection placeholder & CSV-driven estimates
import pandas as pd
import numpy as np
from typing import Dict

def load_single_node_det_csv(path:str) -> pd.DataFrame:
    """
    Expect columns: observer_id, threshold_tau, E_T1_seconds
    If missing, returns empty DF.
    """
    try:
        df = pd.read_csv(path)
        needed = {"observer_id","threshold_tau","E_T1_seconds"}
        if not needed.issubset(set(df.columns)):
            return pd.DataFrame(columns=list(needed))
        return df
    except Exception:
        return pd.DataFrame(columns=["observer_id","threshold_tau","E_T1_seconds"])

def default_E_T1_map(observer_ids, tau_default: float = 10.0, mean_E_T1=15.0, std=5.0, seed=123) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = np.clip(rng.normal(mean_E_T1, std, size=len(observer_ids)), 3.0, 60.0)
    return {oid: float(v) for oid, v in zip(observer_ids, vals)}

def estimate_E_T1_map(csv_path:str, observer_ids, tau_default: float = 10.0) -> Dict[str, float]:
    """
    Build {observer_id -> E[T1]} from CSV if present; otherwise make a reasonable default.
    """
    df = load_single_node_det_csv(csv_path)
    if df.empty:
        return default_E_T1_map(observer_ids, tau_default)
    # pick the closest tau to tau_default for each observer
    df["_tau_diff"] = np.abs(df["threshold_tau"] - float(tau_default))
    df = df.sort_values("_tau_diff")
    chosen = df.groupby("observer_id").head(1)
    mp = {str(r["observer_id"]): float(r["E_T1_seconds"]) for _, r in chosen.iterrows()}
    # fill missing with defaults
    missing = [oid for oid in observer_ids if oid not in mp]
    if missing:
        mp.update(default_E_T1_map(missing, tau_default))
    return mp
