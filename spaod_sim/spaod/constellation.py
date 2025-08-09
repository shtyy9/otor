"""SPAO-D 使用的简化 Walker-Delta 星座模型。

本模块提供若干基础几何工具以及一个轻量级类，用于传播类似 Starlink 的卫星位置。
该模型并非追求严格的天体力学精度，而是力求在网络层仿真和可见性检查上足够使用。
"""
import numpy as np
from dataclasses import dataclass

Re = 6371000.0  # Earth radius [m]
mu = 3.986004418e14  # Earth GM [m^3/s^2]
DEG = np.pi/180.0

def geo_to_ecef(lat_deg: float, lon_deg: float, alt_m: float=0.0):
    """将大地坐标转换为 ECEF 笛卡尔坐标。"""
    lat = lat_deg*DEG; lon = lon_deg*DEG
    r = Re + alt_m
    x = r*np.cos(lat)*np.cos(lon)
    y = r*np.cos(lat)*np.sin(lon)
    z = r*np.sin(lat)
    return np.array([x,y,z])

def elevation_deg(sat_ecef: np.ndarray, gs_ecef: np.ndarray):
    """计算地面站指向卫星的仰角，单位为度。"""
    # Elevation = angle between (sat - gs) and local horizon at gs.
    vec = sat_ecef - gs_ecef
    rgs = np.linalg.norm(gs_ecef)
    z_hat = gs_ecef / (rgs + 1e-12)
    # Local zenith is +z_hat; elevation = 90° - angle to tangent plane
    cos_z = np.dot(vec/np.linalg.norm(vec), z_hat)
    # Zenith angle -> elevation
    el = np.arcsin(np.clip(cos_z, -1.0, 1.0)) / DEG
    return el

@dataclass
class WalkerDeltaCfg:
    """Walker‑Delta 星座的配置参数。"""
    planes: int = 24
    sats_per_plane: int = 66
    inclination_deg: float = 53.0
    altitude_m: float = 550000.0
    phase_factor: int = 1  # f in Walker-delta

class WalkerDelta:
    """Walker‑Delta 卫星的轻量级轨道传播器。"""
    def __init__(self, cfg: WalkerDeltaCfg):
        self.cfg = cfg
        self.N = cfg.planes * cfg.sats_per_plane
        self.a = Re + cfg.altitude_m
        self.omega = np.sqrt(mu / (self.a**3))  # rad/s
        self.inc = cfg.inclination_deg * DEG
        self._catalog = self._build_catalog()

    def _build_catalog(self):
        P, S, f = self.cfg.planes, self.cfg.sats_per_plane, self.cfg.phase_factor
        cat = []
        for p in range(P):
            raan = 2*np.pi * p / P
            for s in range(S):
                # Walker-delta true anomaly offset: 2π*(s/S + f*p/P)
                M0 = 2*np.pi*(s/S + (f*p)/P)  # mean anomaly at t=0 (circular -> true anomaly)
                cat.append((p, s, raan, M0))
        return np.array(cat, dtype=float)  # shape [N,4]

    def eci_position(self, t_s: float):
        """返回在 ``t_s`` 秒时刻所有卫星的 ECI 坐标。"""
        # Circular orbit in ECI, per-satellite RAAN & true anomaly = M0 + omega*t
        P, S = self.cfg.planes, self.cfg.sats_per_plane
        N = self.N
        a = self.a
        inc = self.inc
        # Unpack
        raan = self._catalog[:,2]
        nu   = self._catalog[:,3] + self.omega*t_s  # true anomaly at time t
        # Perifocal to ECI (circular): r_pf = [a*cos nu, a*sin nu, 0]
        cosO, sinO = np.cos(raan), np.sin(raan)
        cosi, sini = np.cos(inc), np.sin(inc)
        cosu, sinu = np.cos(nu), np.sin(nu)
        x_pf = a*cosu; y_pf = a*sinu; z_pf = np.zeros_like(x_pf)
        # Rotation: Rz(O) * Rx(i)
        x = (cosO)*x_pf + (-sinO*cosi)*y_pf + ( sinO*sini)*z_pf
        y = (sinO)*x_pf + ( cosO*cosi)*y_pf + (-cosO*sini)*z_pf
        z = (0.0  )*x_pf + (      sini)*y_pf + (      cosi)*z_pf
        return np.stack([x,y,z], axis=1)  # [N,3]

    def batch_eci(self, t_array_s: np.ndarray):
        """对多个时刻进行向量化调用 :meth:`eci_position`。"""
        return [self.eci_position(float(t)) for t in t_array_s]

def visible_to_ground(eci_sat: np.ndarray, gs_lat: float, gs_lon: float, gs_alt_m: float, elev_min_deg: float):
    """计算相对于某地面站的所有卫星的可见性掩码及仰角。"""
    gs = geo_to_ecef(gs_lat, gs_lon, gs_alt_m)
    el = np.array([elevation_deg(eci_sat[i], gs) for i in range(eci_sat.shape[0])])
    return el >= elev_min_deg, el

# 供其他模块调用的便捷工厂函数
def build_walker_constellation(planes=24, sats_per_plane=66, inclination_deg=53.0, altitude_m=550000.0, phase_factor=1):
    """返回一个 :class:`WalkerDelta` 实例的便捷工厂函数。"""
    cfg = WalkerDeltaCfg(planes, sats_per_plane, inclination_deg, altitude_m, phase_factor)
    return WalkerDelta(cfg)

# Quick self-test (runs only when called directly)
if __name__ == "__main__":
    const = build_walker_constellation()
    t = np.arange(0, 600, 1.0)
    eci0 = const.eci_position(0.0)
    print("Sat count:", eci0.shape[0], " first sat norm [km]:", np.linalg.norm(eci0[0])/1000.0)
