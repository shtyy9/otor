"""Common metrics for evaluating paths in the SPAO-D simulator.

当前实现提供了简单的跳数与时延计算函数，可作为更复杂指标的基础。
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple


def hop_count(path: Sequence[int]) -> int:
    """Return the number of hops in a path."""
    return max(len(path) - 1, 0)


def path_latency(
    path: Sequence[int],
    link_latencies: Mapping[Tuple[int, int], float],
) -> float:
    """Compute the total latency along a path.

    Parameters
    ----------
    path:
        Sequence of node identifiers representing a path.
    link_latencies:
        Mapping ``(u, v) -> latency_ms`` for each directed link.

    Returns
    -------
    float
        Sum of the latencies for each hop; missing links count as 0.
    """

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += float(link_latencies.get((u, v), 0.0))
    return total


__all__ = ["hop_count", "path_latency"]

