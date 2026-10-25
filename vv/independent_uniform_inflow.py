"""
Independent static BEMT cross-check — intentionally NOT the a-factor kernel
in vv/reference_bemt.py.

Algorithm: uniform induced velocity v_i with momentum closure
    T = 2 ρ A v_i²
and blade-element loads using φ = arctan(v_i / (Ω r)).

This is a different code path (uniform inflow, not per-station a-iteration)
so agreement with full_bemt is a genuine physics cross-check, not an identity.
"""

from __future__ import annotations

import numpy as np


def _cl_cd(aoa_deg: float):
    aoa = np.radians(aoa_deg)
    cl = 2.0 * np.pi * np.sin(aoa) * np.cos(aoa)
    cd = 0.008 + 0.03 * (np.sin(aoa) ** 2)
    return max(0.0, float(cl)), max(0.001, float(cd))


def uniform_inflow_static_forces(
    rpm: float,
    pitch_deg: float,
    radius: float,
    blades: int,
    chord: float,
    rho: float = 1.225,
    n_stations: int = 25,
    max_iter: int = 60,
    tol: float = 1e-6,
):
    omega = 2.0 * np.pi * rpm / 60.0
    area = np.pi * radius**2
    r = np.linspace(0.2, 0.99, n_stations) * radius
    dr = r[1] - r[0]

    # Seed induced velocity from a modest guess
    v_i = 0.05 * omega * radius
    thrust = 0.0
    torque = 0.0

    for _ in range(max_iter):
        thrust = 0.0
        torque = 0.0
        for ri in r:
            phi = np.arctan2(v_i, omega * ri)
            aoa = pitch_deg - np.degrees(phi)
            cl, cd = _cl_cd(aoa)
            V_rel = np.hypot(omega * ri, v_i)
            dL = 0.5 * rho * V_rel**2 * chord * dr * cl
            dD = 0.5 * rho * V_rel**2 * chord * dr * cd
            # No Prandtl tip loss here — keeps the path distinct from reference_bemt
            dT = blades * (dL * np.cos(phi) - dD * np.sin(phi))
            dQ = blades * ri * (dD * np.cos(phi) + dL * np.sin(phi))
            thrust += dT
            torque += dQ

        # Momentum closure for static actuator disk
        v_i_new = np.sqrt(max(thrust, 0.0) / (2.0 * rho * area + 1e-12))
        if abs(v_i_new - v_i) < tol * max(1.0, abs(v_i_new)):
            v_i = v_i_new
            break
        v_i = 0.5 * v_i + 0.5 * v_i_new

    power = torque * omega
    p_ideal = thrust * v_i  # = T * sqrt(T/(2ρA)) when converged
    return {
        "thrust": float(thrust),
        "power": float(power),
        "v_i": float(v_i),
        "p_ideal": float(p_ideal),
        "fom": float(p_ideal / power) if power > 1e-12 else float("nan"),
    }
