"""Starlink 类星座的时变拓扑构建。"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
from typing import Callable, Dict, List, Tuple, Optional

from .constellation import build_walker_constellation, geo_to_ecef, elevation_deg

DEG = np.pi/180.0
KM = 1000.0

@dataclass
class LinkCfg:
    isl_max_km: float = 3000.0
    elev_min_deg: float = 10.0

@dataclass
class TEParams:
    dt: float = 1.0
    horizon_s: int = 600
    pat_mean_s: float = 0.5  # average acquisition time surrogate

class TopologyBuilder:
    """构建瞬时的 ISL/SG 链路以及时间展开图（TEG）。"""
    def __init__(self, planes:int, sats_per_plane:int, inc_deg:float, alt_m:float, phase_factor:int,
                 link_cfg: LinkCfg, te_params: TEParams):
        self.P = planes
        self.S = sats_per_plane
        self.N = planes * sats_per_plane
        self.link_cfg = link_cfg
        self.te = te_params
        self.const = build_walker_constellation(planes, sats_per_plane, inc_deg, alt_m, phase_factor)
        # Precompute catalog indices
        self.idx_to_ps = np.array([(p, s) for p in range(self.P) for s in range(self.S)], dtype=int)

    # ---------- Instantaneous neighbors ----------
    def isl_neighbors(self, isl_max_m: float) -> Dict[int, List[int]]:
        """静态邻接模板：同平面 ±1 以及相邻平面最近相位（循环）。

        实际距离过滤在每个时刻根据卫星位置进行。
        """
        nbrs = {i: set() for i in range(self.N)}
        # same-plane ±1
        for p in range(self.P):
            for s in range(self.S):
                i = p*self.S + s
                fwd = p*self.S + ((s+1) % self.S)
                bwd = p*self.S + ((s-1) % self.S)
                nbrs[i].update([fwd, bwd])
        # adjacent-plane nearest phase
        for p in range(self.P):
            for s in range(self.S):
                i = p*self.S + s
                for dp in (-1, +1):
                    q = (p + dp) % self.P
                    # nearest phase index in plane q
                    j = q*self.S + s  # simple phase alignment; refined match can be added later
                    nbrs[i].add(j)
        return {i: list(v) for i, v in nbrs.items()}

    def visible_sg(self, sat_ecef: np.ndarray, ogs_ecef: np.ndarray, elev_min_deg: float):
        """判断卫星位置对各地面站的可见性掩码。"""
        vis = []
        for gi, gs in enumerate(ogs_ecef):
            el = elevation_deg(sat_ecef, gs)
            vis.append(el >= elev_min_deg)
        return np.array(vis, dtype=bool)

    # ---------- Build instantaneous edge list ----------
    def edges_at_time(self, t_s: float, ogs_df: pd.DataFrame) -> Tuple[List[Tuple[str,str,dict]], np.ndarray]:
        pos = self.const.eci_position(t_s)  # using ECI as quasi-ECEF within short horizon
        isl_nbrs = self.isl_neighbors(self.link_cfg.isl_max_km*KM)
        edges = []
        # ISL edges (undirected)
        for i in range(self.N):
            pi = pos[i]
            for j in isl_nbrs[i]:
                if j <= i:  # avoid duplicates
                    continue
                pj = pos[j]
                d = np.linalg.norm(pi - pj)
                if d <= self.link_cfg.isl_max_km*KM:
                    edges.append((f"S{i}", f"S{j}", {"type":"ISL", "meters": float(d)}))
        # SG edges (directed both ways)
        ogs_ecef = np.array([geo_to_ecef(row.lat, row.lon, row.alt_m) for _, row in ogs_df.iterrows()])
        for i in range(self.N):
            si = f"S{i}"
            vis = self.visible_sg(pos[i], ogs_ecef, self.link_cfg.elev_min_deg)
            for g, ok in enumerate(vis):
                if not ok: 
                    continue
                gi = f"G{g}"
                d = np.linalg.norm(pos[i] - ogs_ecef[g])
                edges.append((si, gi, {"type":"SG", "meters": float(d)}))
                edges.append((gi, si, {"type":"SG", "meters": float(d)}))
        return edges, pos

    # ---------- Time-expanded graph ----------
    def build_teg(self, ogs_df: pd.DataFrame,
                  feasible_edge_cb: Optional[Callable[[dict], bool]] = None) -> nx.DiGraph:
        """构建时间展开的有向图。

        每个时间层包含卫星和地面站节点；边表示同层物理链路，
        权重为传播时延加期望获取时间（PAT 替代），并包含允许等待的跨层保持边。
        ``feasible_edge_cb(attr)->bool`` 可进一步筛选边（例如 OSNR/容量），允许为 ``None``。
        """
        dt = self.te.dt
        T = int(self.te.horizon_s / dt)
        G = nx.DiGraph()
        # Pre-create ground nodes per layer
        gs_nodes = [f"G{g}" for g in range(len(ogs_df))]
        for layer in range(T+1):
            # Add satellite & ground nodes for this layer
            for i in range(self.N):
                G.add_node((layer, f"S{i}"))
            for gn in gs_nodes:
                G.add_node((layer, gn))

        c = 299792458.0
        pat = self.te.pat_mean_s

        # Build per-layer edges
        for layer in range(T):
            t_s = layer * dt
            edges, _pos = self.edges_at_time(t_s, ogs_df)
            for u, v, attr in edges:
                w_prop = attr["meters"] / c
                w = w_prop + pat  # simple surrogate: propagation + expected acquisition
                data = {"w": w, "kind": attr["type"], "meters": attr["meters"]}
                if feasible_edge_cb is None or feasible_edge_cb({**attr, **data}):
                    G.add_edge((layer, u), (layer+1, v), **data)
            # Add wait edges (stay at same node across time with small cost)
            for i in range(self.N):
                G.add_edge((layer, f"S{i}"), (layer+1, f"S{i}"), w=dt*0.0, kind="WAIT", meters=0.0)
            for gn in gs_nodes:
                G.add_edge((layer, gn), (layer+1, gn), w=dt*0.0, kind="WAIT", meters=0.0)
        return G

# ---------- K-shortest on TEG ----------
def k_shortest_time_expanded(G: nx.DiGraph, src: Tuple[int,str], dst: Tuple[int,str], K:int=6) -> List[List[Tuple[int,str]]]:
    """在时间展开图上执行 Yen 算法求 K 条最短路径（按总权 ``w``）。

    ``src``、``dst`` 为分层节点，如 ``(0,'G0')`` 到 ``(T,'G3')``。
    """
    def path_cost(p):
        return sum(G[u][v]["w"] for u, v in zip(p[:-1], p[1:]))
    # First shortest
    try:
        p0 = nx.shortest_path(G, src, dst, weight="w")
    except nx.NetworkXNoPath:
        return []
    A = [p0]
    A_cost = [path_cost(p0)]
    B = []
    for k in range(1, K):
        prev = A[-1]
        for i in range(len(prev)-1):
            spur_node = prev[i]
            root_path = prev[:i+1]
            removed = []
            # Remove edges that would create same root prefix
            for p in A:
                if len(p) > i and p[:i+1] == root_path and i+1 < len(p):
                    u, v = p[i], p[i+1]
                    if G.has_edge(u, v):
                        attr = G[u][v]
                        G.remove_edge(u, v)
                        removed.append((u, v, attr))
            # Compute spur path
            try:
                spur = nx.shortest_path(G, spur_node, dst, weight="w")
                cand = root_path[:-1] + spur
                B.append((cand, path_cost(cand)))
            except nx.NetworkXNoPath:
                pass
            # Restore
            for u, v, attr in removed:
                G.add_edge(u, v, **attr)
        if not B: 
            break
        # Pick lowest-cost candidate not in A
        B.sort(key=lambda x: x[1])
        while B and any(np.array_equal(B[0][0], ap) for ap in A):
            B.pop(0)
        if not B:
            break
        A.append(B[0][0]); A_cost.append(B[0][1]); B.pop(0)
    return A

# ---------- Convenience: build candidates for OGS->OGS ----------
def candidate_paths_for_pair(ogs_df: pd.DataFrame, src_g:int, dst_g:int,
                             topo: TopologyBuilder,
                             feasible_edge_cb: Optional[Callable[[dict], bool]]=None,
                             K:int=6) -> List[List[Tuple[int,str]]]:
    """先构建 TEG，再求从 ``(layer0,Gsrc)`` 到 ``(layerT,Gdst)`` 的 K 条最短路径。"""
    G = topo.build_teg(ogs_df, feasible_edge_cb=feasible_edge_cb)
    T = int(topo.te.horizon_s / topo.te.dt)
    src = (0, f"G{src_g}")
    dst = (T, f"G{dst_g}")
    return k_shortest_time_expanded(G, src, dst, K=K)

# Quick self-test (runs only when executed directly)
if __name__ == "__main__":
    # Minimal smoke test with synthetic OGS table
    ogs = pd.DataFrame([
        {"name":"LA","lat":34.05,"lon":-118.25,"alt_m":50.0,"site_avail":0.7},
        {"name":"Sydney","lat":-33.87,"lon":151.21,"alt_m":30.0,"site_avail":0.75},
    ])
    topo = TopologyBuilder(24,66,53.0,550000.0,1, LinkCfg(), TEParams(dt=1.0, horizon_s=120, pat_mean_s=0.5))
    paths = candidate_paths_for_pair(ogs, 0, 1, topo, feasible_edge_cb=None, K=3)
    print("Found", len(paths), "candidate paths; first lengths:", [len(p) for p in paths])

