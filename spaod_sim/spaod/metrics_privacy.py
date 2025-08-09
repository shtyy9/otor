"""
Privacy / path quality metrics for SPAO-D simulation.
Implements:
1) Path overlap with shortest path (anonymity degradation risk)
2) Orbital plane deviation and intra-plane deviation
3) Exposure probability under given threat model
"""
import networkx as nx
import numpy as np

def shortest_path_nodes(G, src, dst, weight="w"):
    try:
        return nx.shortest_path(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return []

def path_quality_metrics(candidate_path, shortest_path, sat_orbit_map):
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
    observed_links = 0
    for u, v in zip(path[:-1], path[1:]):
        if threat_model.is_link_observed(u, v):
            observed_links += 1
    return observed_links / max(1, len(path)-1)
