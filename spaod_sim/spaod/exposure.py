# ====================== spaod/exposure.py ======================
# Exposure score calibration from single-node detection; window risk helpers
from typing import Dict, Tuple
import numpy as np

def compute_dotD_from_ET1(E_T1_map: Dict[str, float], tau: float) -> Dict[str, float]:
    """
    dotD_i ≈ tau / E[T1]_i  (units: arbitrary 'info-rate' per second)
    """
    mp = {}
    for oid, et1 in E_T1_map.items():
        et1 = max(1e-6, float(et1))
        mp[oid] = float(tau) / et1
    return mp

def distribute_exposure_to_edges(edge_usage: Dict[Tuple[str,str], float],
                                 dotD_map: Dict[str, float]) -> Dict[Tuple[str,str], float]:
    """
    edge_usage: {(u,v): time_occupied_seconds} accumulated from sim logs (任意尺度的占用时间或比例)
    dotD_map: {observer_id: dotD} where observer_id is typically a node ID (e.g., 'S123','G5')
    将每个观察点的可辨识速率按它“相关的边占用”分提到边上，得到 g[e]（暴露分）。
    一个简单可复用的缺省策略：若边(u,v)与某观察点o相邻（u==o或v==o），则该点对该边的贡献∝ edge_usage[(u,v)]。
    更精细的映射可以在将来替换。
    """
    # 先统计每个观察点相关边的总占用
    adj_sum: Dict[str, float] = {}
    for (u,v), usage in edge_usage.items():
        for o in (u,v):
            adj_sum[o] = adj_sum.get(o, 0.0) + float(usage)

    g_e: Dict[Tuple[str,str], float] = {}
    for (u,v), usage in edge_usage.items():
        contrib = 0.0
        for o in (u,v):
            if o in dotD_map and adj_sum.get(o, 0.0) > 0:
                frac = float(usage) / adj_sum[o]
                contrib += dotD_map[o] * frac
        g_e[(u,v)] = contrib
    return g_e

def p_single_window(E_T1: float, W_s: float) -> float:
    """
    Approximate single-node success probability within window W:
      p_i(W) ≈ min(1, W / E[T1]_i)
    """
    if E_T1 <= 0: return 1.0
    return float(min(1.0, W_s / E_T1))

def chernoff_upper_bound_poibin(p_list, m:int):
    """
    Simple Chernoff-style bound for Poisson-Binomial tail:
      P( sum Xi >= m ) <= inf_{t>0} exp(-t m) * Π_i (1 - p_i + p_i e^t)
    We do coarse grid over t.
    """
    p = np.array(p_list, dtype=float)
    if len(p) == 0: return 0.0
    t_grid = np.linspace(0.05, 5.0, 200)
    best = 1.0
    for t in t_grid:
        mgf = np.prod(1.0 - p + p*np.exp(t))
        bound = np.exp(-t*m) * mgf
        if bound < best: best = bound
    return float(min(1.0, max(0.0, best)))

def window_multi_node_risk(E_T1_map: Dict[str,float], W_s: float, m:int) -> float:
    """
    Compute Chernoff upper bound of '>= m nodes succeed within W'.
    """
    p_list = [p_single_window(v, W_s) for v in E_T1_map.values()]
    return chernoff_upper_bound_poibin(p_list, m)
