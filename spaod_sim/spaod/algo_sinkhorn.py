# Entropy-regularized path selection (soft assignment over K candidates)
import numpy as np

def softmin_costs(costs, temp=0.5):
    """
    Given list/np.array costs, return Gibbs distribution over candidates.
    """
    c = np.array(costs, dtype=float)
    c = c - c.min()  # numerical stability
    w = np.exp(-c / max(1e-9, temp))
    if w.sum() <= 0:  # fallback
        w = np.ones_like(w)
    p = w / w.sum()
    return p

def pick_path_by_probs(paths, probs, rng=None):
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(paths), p=probs)
    return idx, paths[idx]
