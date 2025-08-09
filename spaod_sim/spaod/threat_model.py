"""
SPAO-D 仿真中的威胁模型定义。
包含：
1) 对手能力枚举（本地观察者、全局观察者、节点被攻陷）。
2) 信任级别枚举（可信、半可信、不可信），用于 UE、BTS-sat、core-sat、OGS 等实体。
3) 入侵场景生成器，可模拟短期大规模或长期小规模攻击。
"""
from enum import Enum
import numpy as np

class TrustLevel(Enum):
    TRUSTED = 1
    SEMI_TRUSTED = 2
    UNTRUSTED = 3

class AdversaryType(Enum):
    LOCAL = 1
    GLOBAL = 2
    NODE_COMPROMISE = 3

class ThreatModel:
    def __init__(self, trust_map, adv_type:AdversaryType, compromised_nodes=None):
        """初始化威胁模型。

        trust_map: ``{实体类型:str -> TrustLevel}``
        adv_type: ``AdversaryType``
        compromised_nodes: 节点 ID 列表（如 ['S12','S245']）
        """
        self.trust_map = trust_map
        self.adv_type = adv_type
        self.compromised_nodes = set(compromised_nodes or [])

    def is_link_observed(self, u, v):
        """若假设对手能观察链路 ``(u,v)`` 则返回 ``True``。"""
        if self.adv_type == AdversaryType.GLOBAL:
            return True
        if self.adv_type == AdversaryType.LOCAL:
            return u.startswith("G") or v.startswith("G")
        if self.adv_type == AdversaryType.NODE_COMPROMISE:
            return u in self.compromised_nodes or v in self.compromised_nodes
        return False

    @staticmethod
    def generate_intrusion_scenario(node_ids, mode:str, fraction_or_list):
        """生成入侵场景。

        mode='short_term' -> ``fraction_or_list`` 表示被攻陷节点的比例 (float)
        mode='long_term'  -> ``fraction_or_list`` 为明确的被攻陷节点 ID 列表
        """
        if mode == 'short_term':
            n = int(len(node_ids) * fraction_or_list)
            return np.random.choice(node_ids, n, replace=False).tolist()
        elif mode == 'long_term':
            return list(fraction_or_list)
        else:
            raise ValueError("Unknown mode")
