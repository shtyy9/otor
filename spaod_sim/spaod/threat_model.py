"""
Threat model definition for SPAO-D simulation.
Includes:
1) Adversary capability enum (local observer, global observer, compromised nodes).
2) Trust level enum (trusted, semi-trusted, untrusted) for UE, BTS-sat, core-sat, OGS.
3) Intrusion scenario generator for short-term large-scale and long-term small-scale attacks.
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
        """
        trust_map: dict{entity_type:str -> TrustLevel}
        adv_type: AdversaryType
        compromised_nodes: list of node IDs (e.g., ['S12','S245'])
        """
        self.trust_map = trust_map
        self.adv_type = adv_type
        self.compromised_nodes = set(compromised_nodes or [])

    def is_link_observed(self, u, v):
        if self.adv_type == AdversaryType.GLOBAL:
            return True
        if self.adv_type == AdversaryType.LOCAL:
            return u.startswith("G") or v.startswith("G")
        if self.adv_type == AdversaryType.NODE_COMPROMISE:
            return u in self.compromised_nodes or v in self.compromised_nodes
        return False

    @staticmethod
    def generate_intrusion_scenario(node_ids, mode:str, fraction_or_list):
        """
        mode='short_term' -> fraction_or_list = fraction of compromised nodes (float)
        mode='long_term'  -> fraction_or_list = explicit list of compromised node IDs
        """
        if mode == 'short_term':
            n = int(len(node_ids) * fraction_or_list)
            return np.random.choice(node_ids, n, replace=False).tolist()
        elif mode == 'long_term':
            return list(fraction_or_list)
        else:
            raise ValueError("Unknown mode")
