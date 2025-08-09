"""基于单节点检测的暴露评分校准。

此处的辅助函数把单节点检测时间映射为边级暴露分，
并提供用于分析多节点风险窗口的工具。
"""
from typing import Dict, Tuple
import numpy as np

def compute_dotD_from_ET1(E_T1_map: Dict[str, float], tau: float) -> Dict[str, float]:
    """将 ``E[T1]`` 映射转成 ``dotD``：``dotD_i ≈ tau / E[T1]_i``（单位为秒⁻¹的信息率）。"""
    mp = {}
    for oid, et1 in E_T1_map.items():
        et1 = max(1e-6, float(et1))
        mp[oid] = float(tau) / et1
    return mp

def distribute_exposure_to_edges(edge_usage: Dict[Tuple[str,str], float],
                                 dotD_map: Dict[str, float]) -> Dict[Tuple[str,str], float]:
    """将节点可辨识速率 ``dotD`` 分配到各边上以得到暴露分 ``g[e]``。

    参数说明：
    ``edge_usage`` 为 ``{(u,v): time_occupied_seconds}``，来源于仿真日志的边占用时间或比例；
    ``dotD_map`` 为 ``{observer_id: dotD}``，观察点一般为节点 ID（如 'S123'、'G5'）。
    简单可复用的默认策略是：若边 ``(u,v)`` 与观察点 ``o`` 相邻，则该点对该边的贡献与 ``edge_usage[(u,v)]`` 成比例。
    将来可替换为更精细的映射规则。
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
    """近似计算单节点在窗口 ``W`` 内成功的概率：``p_i(W) ≈ min(1, W / E[T1]_i)``。"""
    if E_T1 <= 0: return 1.0
    return float(min(1.0, W_s / E_T1))

def chernoff_upper_bound_poibin(p_list, m:int):
    """Poisson-Binomial 分布尾部的简单 Chernoff 上界。

    公式：``P(∑Xi >= m) ≤ inf_{t>0} e^{-t m} * Π_i (1 - p_i + p_i e^t)``，
    此处对 ``t`` 进行粗网格搜索。
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
    """计算 ``W`` 窗口内“至少 ``m`` 个节点成功”的 Chernoff 上界。"""
    p_list = [p_single_window(v, W_s) for v in E_T1_map.values()]
    return chernoff_upper_bound_poibin(p_list, m)
