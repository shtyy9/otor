# Glue: build candidates, score with latency+privacy, enforce risk constraints, select path
import os, json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from .topology import TopologyBuilder, LinkCfg, TEParams, candidate_paths_for_pair
from .metrics_privacy import shortest_path_nodes, path_quality_metrics, exposure_probability
from .threat_model import ThreatModel, AdversaryType, TrustLevel
from .detector import estimate_E_T1_map
from .exposure import compute_dotD_from_ET1, distribute_exposure_to_edges, window_multi_node_risk
from .algo_sinkhorn import softmin_costs, pick_path_by_probs

@dataclass
class AlgoCfg:
    K_paths:int = 6
    temp: float = 0.6              # temperature of softmin
    alpha: float = 0.5             # weight: latency
    beta: float  = 1.0             # weight: overlap_ratio (anonymity)
    gamma: float = 0.1             # weight: orbit/intra diffs (path shape diversity)
    zeta: float  = 0.5             # weight: edge exposure score (from single-node detection)
    risk_W_s: float = 30.0         # time window for multi-node risk
    risk_m_max: int = 2            # tail event: >=m nodes succeed
    risk_bound: float = 0.2        # admissible upper bound for Chernoff tail
    tau_threshold: float = 10.0    # detector threshold for mapping E[T1]
    seed:int = 123

def _sat_orbit_map(P:int, S:int) -> Dict[str, Tuple[int,int]]:
    # S<i> -> (plane, in-plane index)
    mp={}
    for p in range(P):
        for s in range(S):
            i = p*S+s
            mp[f"S{i}"]=(p,s)
    return mp

def _flatten_layered(path: List[Tuple[int,str]]) -> List[str]:
    return [name for (_layer, name) in path]

def _edge_usage_from_path(flat_nodes: List[str]) -> Dict[Tuple[str,str], float]:
    # 简单：每条边占用=1（或按边数归一化）；用于把 dotD 分摊到边
    usage={}
    for u,v in zip(flat_nodes[:-1], flat_nodes[1:]):
        usage[(u,v)] = usage.get((u,v), 0.0) + 1.0
    return usage

def _path_latency_seconds(G: nx.DiGraph, layered_path: List[Tuple[int,str]]) -> float:
    w=0.0
    for u,v in zip(layered_path[:-1], layered_path[1:]):
        w+=G[u][v].get("w",0.0)
    return float(w)

def select_path_for_pair(ogs_df: pd.DataFrame,
                         src_g: int, dst_g: int,
                         topo: TopologyBuilder,
                         algo_cfg: AlgoCfg,
                         det_csv_path: str = "data/single_node_det.csv",
                         adversary: ThreatModel = None):
    rng = np.random.default_rng(algo_cfg.seed + src_g*991 + dst_g*997)

    # 1) 构建时间展开图 + K 条候选
    G = topo.build_teg(ogs_df, feasible_edge_cb=None)
    T = int(topo.te.horizon_s / topo.te.dt)
    layered_src = (0, f"G{src_g}")
    layered_dst = (T, f"G{dst_g}")
    cand_paths = candidate_paths_for_pair(ogs_df, src_g, dst_g, topo, feasible_edge_cb=None, K=algo_cfg.K_paths)
    if not cand_paths:
        return {"chosen_idx": -1, "reason":"no_path", "paths":[]}

    # 2) 参考最短（用于匿名重叠指标）
    sp_ref = shortest_path_nodes(G, layered_src, layered_dst, weight="w")
    sp_ref_flat = _flatten_layered(sp_ref)

    # 3) 单点检测与窗口化多点风险约束
    #    3.1 估计 E[T1] 映射 -> dotD -> g[e]
    all_nodes = set()
    for P in cand_paths:
        for _, name in P:
            all_nodes.add(name)
    observer_ids = [n for n in all_nodes if n.startswith("S") or n.startswith("G")]
    E_map = estimate_E_T1_map(det_csv_path, observer_ids, tau_default=algo_cfg.tau_threshold)   # {node: E[T1]}
    # Chernoff 上界（不依赖具体路径，先作为 admission gate）
    risk_upper = window_multi_node_risk(E_map, algo_cfg.risk_W_s, algo_cfg.risk_m_max)
    if risk_upper > algo_cfg.risk_bound:
        # 过于危险，返回拒绝（真实系统可触发“加密强度/混淆度升级”或延迟重试）
        return {"chosen_idx": -1, "reason": f"risk_upper={risk_upper:.3f}>bound", "paths":[]}

    # 4) 逐路径打分：延迟 + 匿名性重叠 + 形状差异 + 暴露分（边级）
    satmap = _sat_orbit_map(topo.P, topo.S)
    costs = []
    path_feats = []
    for Pth in cand_paths:
        flat = _flatten_layered(Pth)

        # 4.1 延迟
        lat = _path_latency_seconds(G, Pth)

        # 4.2 匿名性（与最短路的重叠、轨道/轨内差异）
        q = path_quality_metrics(flat, sp_ref_flat, {f"S{i}": satmap.get(f"S{i}", None) for i in range(topo.N)})
        overlap = q["overlap_ratio"]
        orbit_diff = q["orbit_diff"]
        intra_diff = q["intra_diff"]

        # 4.3 暴露分：用占用把 dotD 分摊到边，再沿路径累加
        edge_usage = _edge_usage_from_path(flat)
        dotD = compute_dotD_from_ET1(E_map, algo_cfg.tau_threshold)
        g_e = distribute_exposure_to_edges(edge_usage, dotD)
        exposure_sum = sum(g_e.get((u,v),0.0) for (u,v) in edge_usage.keys())

        # 4.4 组合代价
        c = (algo_cfg.alpha * lat
             + algo_cfg.beta  * overlap
             + algo_cfg.gamma * (0.5*orbit_diff + 0.5*intra_diff)
             + algo_cfg.zeta  * exposure_sum)
        costs.append(c)
        path_feats.append({
            "lat_s": lat, "overlap": overlap, "orbit_diff": orbit_diff,
            "intra_diff": intra_diff, "exposure_sum": exposure_sum
        })

    # 5) 软选择（温度 temp），输出被选路径及备选排序
    probs = softmin_costs(costs, temp=algo_cfg.temp)
    idx, chosen = pick_path_by_probs(cand_paths, probs, rng=rng)
    order = np.argsort(costs).tolist()

    return {
        "chosen_idx": int(idx),
        "chosen_path": chosen,
        "probs": probs.tolist(),
        "costs": costs,
        "order": order,
        "feats": path_feats,
        "risk_upper": float(risk_upper),
    }
