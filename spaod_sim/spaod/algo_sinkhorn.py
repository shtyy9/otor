"""熵正则化路径选择的实用工具。

本模块提供在一组候选路径之间执行**软选择**的轻量级辅助函数。
核心思路是使用 Gibbs 分布根据每条路径的代价分配概率，
然后按这些概率随机抽取一条路径。
它只是对 NumPy 的一个简洁封装，供求解器在多条路由方案中使用。
"""
import numpy as np

def softmin_costs(costs, temp=0.5):
    """给定成本列表/数组，返回候选集合的 Gibbs 分布。"""
    c = np.array(costs, dtype=float)
    c = c - c.min()  # numerical stability
    w = np.exp(-c / max(1e-9, temp))
    if w.sum() <= 0:  # fallback
        w = np.ones_like(w)
    p = w / w.sum()
    return p

def pick_path_by_probs(paths, probs, rng=None):
    """按给定概率随机选择一条路径。

    参数
    ----
    paths : list
        候选路径列表。
    probs : array-like
        每条候选路径对应的选择概率，应当和为 1。
    rng : numpy.random.Generator, optional
        可选的随机数生成器，便于测试时获得确定性。

    返回
    ----
    tuple
        ``(idx, path)``，``idx`` 是被选中路径的索引，``path`` 为其节点序列。
    """
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(paths), p=probs)
    return idx, paths[idx]
