"""Basic visualisation helpers for the SPAO-D simulator.

当前仅实现地面站路径的简单 2D 绘制，可为后续可视化提供起点。
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def plot_ground_path(
    ogs: pd.DataFrame, path_nodes: Sequence[str], ax=None
):
    """Plot the ground-station segments of a path on a simple lon/lat map.

    Parameters
    ----------
    ogs:
        ``pandas.DataFrame`` containing columns ``id``, ``lat_deg`` and ``lon_deg``.
    path_nodes:
        Sequence of node names as returned by the solver, e.g. ``["G0","S1",...,"G5"]``.
    ax:
        Optional :class:`matplotlib.axes.Axes` to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plotted path.
    """

    import matplotlib.pyplot as plt

    g_nodes = [n for n in path_nodes if n.startswith("G")]
    coords = []
    for node in g_nodes:
        gid = int(node[1:])
        row = ogs.loc[ogs["id"] == gid]
        if not row.empty:
            r = row.iloc[0]
            coords.append((float(r["lon_deg"]), float(r["lat_deg"])))

    if ax is None:
        _, ax = plt.subplots()

    if coords:
        xs, ys = zip(*coords)
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")

    return ax


__all__ = ["plot_ground_path"]

