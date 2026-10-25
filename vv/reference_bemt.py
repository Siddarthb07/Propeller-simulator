"""
Textbook-style BEMT reference with axial induction iteration.

Shared inductive kernel for shipping `full_bemt` and harness identity check.
Supports optional BladeGeometry and freestream_axial (mutual-induction inflow).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def cl_cd(aoa_deg: float):
    aoa = np.radians(aoa_deg)
    cl = 2.0 * np.pi * np.sin(aoa) * np.cos(aoa)
    cd = 0.008 + 0.03 * (np.sin(aoa) ** 2)
    return max(0.0, float(cl)), max(0.001, float(cd))


def prandtl_tip_loss(B: int, r_R: float, phi: float) -> float:
    phi = max(abs(phi), 1e-6)
    f = (B / 2.0) * (1.0 - r_R) / (r_R * np.sin(phi))
    F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), -1.0, 1.0))
    return float(np.clip(F, 0.01, 1.0))


def inductive_bemt_forces(
    rpm: float,
    pitch_deg: float,
    radius: float = 0.5,
    blades: int = 3,
    chord: float = 0.1,
    rho: float = 1.225,
    advance_ratio: float = 0.0,
    n_stations: int = 25,
    max_iter: int = 40,
    tol: float = 1e-5,
    geometry=None,
    freestream_axial: float = 0.0,
):
    omega = 2.0 * np.pi * rpm / 60.0
    n = rpm / 60.0
    D = 2.0 * radius
    V0 = advance_ratio * n * D + float(freestream_axial)

    if geometry is not None:
        r = np.asarray(geometry.stations_m(), dtype=float)
        chord_arr = np.asarray(geometry.chord, dtype=float)
        twist_arr = np.asarray(geometry.twist_deg, dtype=float) + pitch_deg
        blades = int(geometry.blades)
        radius = float(geometry.radius)
    else:
        r = np.linspace(0.2, 0.99, n_stations) * radius
        chord_arr = np.full_like(r, chord)
        twist_arr = np.full_like(r, pitch_deg)

    dr = r[1] - r[0] if len(r) > 1 else radius * 0.04
    thrust = 0.0
    torque = 0.0
    a_vals = []

    for i, ri in enumerate(r):
        r_R = ri / max(radius, 1e-9)
        a = 0.1
        phi = 0.1
        cl = 0.0
        for _ in range(max_iter):
            if abs(V0) <= 1e-9:
                w = a * omega * radius
                phi = np.arctan2(w, omega * ri + 1e-12)
            else:
                phi = np.arctan2(V0 * (1.0 + a), omega * ri + 1e-12)
            aoa = float(twist_arr[i]) - np.degrees(phi)
            cl, cd = cl_cd(aoa)
            F = prandtl_tip_loss(blades, r_R, phi)
            sigma = blades * chord_arr[i] / (2.0 * np.pi * ri + 1e-12)
            sphi = max(abs(np.sin(phi)), 1e-6)
            cphi = np.cos(phi)
            rhs = (sigma * cl * cphi) / (4.0 * F * sphi * sphi + 1e-12)
            a_new = rhs / (1.0 + rhs) if abs(V0) > 1e-9 else min(0.5, 0.5 * rhs)
            if abs(a_new - a) < tol:
                a = a_new
                break
            a = 0.5 * a + 0.5 * a_new
        a_vals.append(a)
        F = prandtl_tip_loss(blades, r_R, phi)
        if abs(V0) <= 1e-9:
            V_rel = np.hypot(omega * ri, a * omega * radius)
        else:
            V_rel = np.hypot(omega * ri, V0 * (1.0 + a))
        aoa = float(twist_arr[i]) - np.degrees(phi)
        cl, cd = cl_cd(aoa)
        dL = 0.5 * rho * V_rel**2 * chord_arr[i] * dr * cl
        dD = 0.5 * rho * V_rel**2 * chord_arr[i] * dr * cd
        dT = blades * F * (dL * np.cos(phi) - dD * np.sin(phi))
        dQ = blades * F * ri * (dD * np.cos(phi) + dL * np.sin(phi))
        thrust += dT
        torque += dQ

    power = torque * omega
    v_i_mean = float(np.mean(a_vals) * omega * radius) if abs(V0) <= 1e-9 else float(
        np.mean(a_vals) * abs(V0)
    )
    return {
        "thrust": float(thrust),
        "power": float(power),
        "torque": float(torque),
        "v_i_mean": v_i_mean,
    }


def momentum_ideal_power(thrust: float, radius: float, rho: float = 1.225) -> float:
    area = np.pi * radius**2
    if thrust <= 0 or area <= 0:
        return 0.0
    return float(thrust * np.sqrt(thrust / (2.0 * rho * area)))
