"""
SPAO-D 仿真中的隐私/路径质量指标。
实现内容：
1) 与最短路径的重叠程度（匿名性下降风险）
2) 轨道面偏差与轨内偏差
3) 在给定威胁模型下的暴露概率
"""
import networkx as nx
import numpy as np

def shortest_path_nodes(G, src, dst, weight="w"):
    """返回 ``src`` 与 ``dst`` 之间最短路径的节点列表。

    若不存在路径则返回空列表而非抛出异常，方便调用方处理。
    """
    try:
        return nx.shortest_path(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return []

def path_quality_metrics(candidate_path, shortest_path, sat_orbit_map):
    """计算候选路径的重叠度与多样性指标。

    参数
    ----
    candidate_path : list[str]
        待评估的路径。
    shortest_path : list[str]
        用于衡量匿名性下降的参考最短路径。
    sat_orbit_map : dict
        ``卫星节点 ID -> 轨道索引`` 的映射，用于衡量路径多样性。
    """
    sp_set = set(shortest_path)
    cand_set = set(candidate_path)
    overlap_nodes = len(sp_set & cand_set)
    overlap_ratio = overlap_nodes / max(1, len(sp_set))
    orbits_cand = {sat_orbit_map.get(n, None) for n in cand_set if n in sat_orbit_map}
    orbits_short = {sat_orbit_map.get(n, None) for n in sp_set if n in sat_orbit_map}
    orbit_diff = len(orbits_cand ^ orbits_short)
    intra_ids_cand = {n for n in cand_set if n.startswith("S")}
    intra_ids_short = {n for n in sp_set if n.startswith("S")}
    intra_diff = len(intra_ids_cand ^ intra_ids_short)
    return {
        "overlap_ratio": overlap_ratio,
        "orbit_diff": orbit_diff,
        "intra_diff": intra_diff
    }

def exposure_probability(path, threat_model):
    """计算在 ``threat_model`` 下，路径中被观察链路的比例。"""
    observed_links = 0
    for u, v in zip(path[:-1], path[1:]):
        if threat_model.is_link_observed(u, v):
            observed_links += 1
    return observed_links / max(1, len(path)-1)
