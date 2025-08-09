"""Light‑weight simulation orchestration utilities.

提供一个 ``Simulation`` 类，用于协调地面站对之间的路径计算。
该实现保持简洁，适合作为更完整仿真框架的基础。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd

from .solver import AlgoCfg, select_path_for_pair
from .topology import TopologyBuilder


@dataclass
class Simulation:
    """Coordinate topology and solver to evaluate multiple traffic pairs."""

    topo: TopologyBuilder
    algo: AlgoCfg

    def run_pairs(
        self, ogs: pd.DataFrame, pairs: Iterable[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], dict]:
        """Compute a routing decision for each ``(src, dst)`` pair.

        Parameters
        ----------
        ogs:
            DataFrame describing ground stations.
        pairs:
            Iterable of ``(src_g, dst_g)`` index pairs.

        Returns
        -------
        dict
            Mapping ``(src, dst) -> solver_result`` as produced by
            :func:`spaod.solver.select_path_for_pair`.
        """

        results: Dict[Tuple[int, int], dict] = {}
        for src, dst in pairs:
            results[(src, dst)] = select_path_for_pair(
                ogs,
                src,
                dst,
                self.topo,
                self.algo,
                det_csv_path="data/single_node_det.csv",
                adversary=None,
            )
        return results


__all__ = ["Simulation"]

