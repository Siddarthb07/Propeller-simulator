"""
Headless BEMT-style propeller physics (no GUI).

- simple_bemt: no iterative induced velocity
- full_bemt: axial induction + Prandtl tip loss
Optional BladeGeometry for variable chord/twist.
Optional axial freestream override for mutual-induction coupling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from geometry_blade import BladeGeometry
except ImportError:  # pragma: no cover
    BladeGeometry = None  # type: ignore


class PropellerPhysics:
    def __init__(
        self,
        radius: float = 0.5,
        blades: int = 3,
        chord: float = 0.1,
        rho: float = 1.225,
        model: str = "simple_bemt",
        airfoil_name: str = "NACA2412",
        geometry: Optional["BladeGeometry"] = None,
        freestream_axial: float = 0.0,
    ):
        self.radius = radius
        self.blades = blades
        self.chord = chord
        self.rho = rho
        self.model = model
        self.airfoil_name = airfoil_name
        self.geometry = geometry
        self.freestream_axial = float(freestream_axial)  # m/s along rotor axis

        if geometry is not None:
            self.radius = geometry.radius
            self.blades = geometry.blades

    def cl_cd_from_aoa(self, aoa_deg: float):
        aoa = np.radians(aoa_deg)
        cl = 2.0 * np.pi * np.sin(aoa) * np.cos(aoa)
        cd = 0.008 + 0.03 * (np.sin(aoa) ** 2)
        return max(0.0, cl), max(0.001, cd)

    def tip_mach(self, rpm: float, speed_of_sound: float = 343.0):
        omega = 2.0 * np.pi * rpm / 60.0
        tip_speed = omega * self.radius
        return tip_speed / speed_of_sound, tip_speed

    def bem_forces(self, rpm: float, pitch_deg: float = 0.0, advance_ratio: float = 0.0):
        """
        pitch_deg is collective add-on when geometry provides twist schedule;
        for constant-chord legacy use, pitch_deg is the blade pitch.
        """
        if self.model == "full_bemt":
            return self._bem_full(rpm, pitch_deg, advance_ratio)
        return self._bem_simple(rpm, pitch_deg, advance_ratio)

    def _station_arrays(self, pitch_deg: float):
        if self.geometry is not None:
            r = self.geometry.stations_m()
            chord = self.geometry.chord
            twist = self.geometry.twist_deg + pitch_deg
        else:
            r = np.linspace(0.2, 1.0, 25) * self.radius
            chord = np.full_like(r, self.chord)
            twist = np.full_like(r, pitch_deg)
        return r, chord, twist

    def _bem_simple(self, rpm: float, pitch_deg: float, advance_ratio: float):
        omega = 2.0 * np.pi * rpm / 60.0
        r, chord, twist = self._station_arrays(pitch_deg)
        # advance from J plus external axial freestream
        n = rpm / 60.0
        D = 2.0 * self.radius
        V_j = advance_ratio * n * D
        V0 = V_j + self.freestream_axial

        thrust = 0.0
        torque = 0.0
        dr = r[1] - r[0] if len(r) > 1 else self.radius * 0.04
        cl_list, cd_list = [], []

        for i, ri in enumerate(r):
            phi = np.arctan2(V0, omega * ri + 1e-12)
            aoa_deg = float(twist[i]) - np.degrees(phi)
            cl, cd = self.cl_cd_from_aoa(aoa_deg)
            cl_list.append(cl)
            cd_list.append(cd)
            V_rel = np.hypot(omega * ri, V0)
            dL = 0.5 * self.rho * V_rel**2 * chord[i] * dr * cl
            dD = 0.5 * self.rho * V_rel**2 * chord[i] * dr * cd
            dT = self.blades * (dL * np.cos(phi) - dD * np.sin(phi))
            dQ = self.blades * ri * (dD * np.cos(phi) + dL * np.sin(phi))
            thrust += dT
            torque += dQ

        power = torque * omega
        efficiency = (thrust * 9.81) / power if power > 1e-6 else 0.0
        disk_area = np.pi * self.radius**2
        tip_mach, tip_speed = self.tip_mach(rpm)
        return {
            "thrust": float(thrust),
            "power": float(power),
            "torque": float(torque),
            "efficiency": efficiency,
            "cl_avg": float(np.mean(cl_list)),
            "cd_avg": float(np.mean(cd_list)),
            "disk_loading": thrust / disk_area if disk_area > 0 else 0.0,
            "power_loading": thrust / power if power > 1e-6 else 0.0,
            "tip_speed": tip_speed,
            "tip_mach": tip_mach,
            "induced_axial_est": 0.0,
        }

    def _bem_full(self, rpm: float, pitch_deg: float, advance_ratio: float):
        import importlib.util
        from pathlib import Path

        ref_path = Path(__file__).resolve().parent / "vv" / "reference_bemt.py"
        spec = importlib.util.spec_from_file_location(
            "propeller_reference_bemt", ref_path
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        inductive_bemt_forces = mod.inductive_bemt_forces

        geom = self.geometry
        forces = inductive_bemt_forces(
            rpm,
            pitch_deg,
            radius=self.radius,
            blades=self.blades,
            chord=self.chord if geom is None else float(np.mean(geom.chord)),
            rho=self.rho,
            advance_ratio=advance_ratio,
            geometry=geom,
            freestream_axial=self.freestream_axial,
        )
        tip_mach, tip_speed = self.tip_mach(rpm)
        thrust = forces["thrust"]
        power = forces["power"]
        disk_area = np.pi * self.radius**2
        return {
            "thrust": float(thrust),
            "power": float(power),
            "torque": float(forces.get("torque", 0.0)),
            "efficiency": (thrust * 9.81) / power if power > 1e-6 else 0.0,
            "cl_avg": float("nan"),
            "cd_avg": float("nan"),
            "disk_loading": thrust / disk_area if disk_area > 0 else 0.0,
            "power_loading": thrust / power if power > 1e-6 else 0.0,
            "tip_speed": tip_speed,
            "tip_mach": tip_mach,
            "induced_axial_est": float(forces.get("v_i_mean", 0.0)),
        }
