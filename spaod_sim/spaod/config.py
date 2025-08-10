"""Configuration helpers for the SPAO-D simulator.

该模块提供从 YAML 文件加载配置并进行基本合法性检查的函数。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import yaml

# 顶层必须存在的配置键
REQUIRED_KEYS: Iterable[str] = (
    "constellation",
    "links",
    "time",
    "algo",
)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    KeyError
        If any required top-level key is missing.
    """

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    return cfg


__all__ = ["load_config", "REQUIRED_KEYS"]

