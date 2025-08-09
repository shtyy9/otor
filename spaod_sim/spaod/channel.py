"""Communication channel utilities for the SPAO-D simulator.

该模块提供了基础的自由空间路径损耗 (FSPL) 以及信噪比 (SNR) 估算函数，
方便后续在更复杂的信道模型上进行扩展。
"""

from __future__ import annotations

import math


def fspl(distance_km: float, freq_ghz: float) -> float:
    """Compute free-space path loss in decibels.

    Parameters
    ----------
    distance_km:
        Distance between the transmitter and receiver in kilometres.
    freq_ghz:
        Carrier frequency in GHz.

    Returns
    -------
    float
        The free-space path loss (dB).

    Notes
    -----
    采用常见的 FSPL 公式 ``L = 20*log10(d_km) + 20*log10(f_GHz) + 92.45``。
    """

    if distance_km <= 0 or freq_ghz <= 0:
        raise ValueError("distance_km and freq_ghz must be positive")
    return 20.0 * math.log10(distance_km) + 20.0 * math.log10(freq_ghz) + 92.45


def snr(
    tx_power_dbm: float,
    path_loss_db: float,
    noise_figure_db: float = 0.0,
    bandwidth_mhz: float = 20.0,
) -> float:
    """Estimate the signal-to-noise ratio in dB.

    Parameters
    ----------
    tx_power_dbm:
        Transmit power in dBm.
    path_loss_db:
        Path loss in dB, e.g. the result of :func:`fspl`.
    noise_figure_db:
        Receiver noise figure in dB.  Defaults to 0.
    bandwidth_mhz:
        Channel bandwidth in MHz.  Defaults to 20 MHz.

    Returns
    -------
    float
        Estimated SNR in dB.

    Notes
    -----
    这里使用热噪声模型 ``N = -174 + 10*log10(B_Hz) + NF``。
    """

    if bandwidth_mhz <= 0:
        raise ValueError("bandwidth_mhz must be positive")

    noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_mhz * 1e6) + noise_figure_db
    return tx_power_dbm - path_loss_db - noise_dbm


__all__ = ["fspl", "snr"]

