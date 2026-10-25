"""
Radial blade geometry schedules for BEMT.

Supports constant or variable chord/twist along r/R.
Morph parameter s in [0,1] interpolates between keyframes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BladeGeometry:
    radius: float
    blades: int
    r_R: np.ndarray
    chord: np.ndarray
    twist_deg: np.ndarray
    root_cutout: float = 0.2
    name: str = "custom"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.r_R = np.asarray(self.r_R, dtype=float)
        self.chord = np.asarray(self.chord, dtype=float)
        self.twist_deg = np.asarray(self.twist_deg, dtype=float)
        if not (len(self.r_R) == len(self.chord) == len(self.twist_deg)):
            raise ValueError("r_R, chord, twist_deg must share length")

    @classmethod
    def constant(
        cls,
        radius: float = 0.15,
        blades: int = 2,
        chord: float = 0.025,
        twist_deg: float = 12.0,
        n_stations: int = 25,
        root_cutout: float = 0.2,
        name: str = "constant",
    ) -> "BladeGeometry":
        r_R = np.linspace(root_cutout, 0.99, n_stations)
        return cls(
            radius=radius,
            blades=blades,
            r_R=r_R,
            chord=np.full_like(r_R, chord),
            twist_deg=np.full_like(r_R, twist_deg),
            root_cutout=root_cutout,
            name=name,
        )

    @classmethod
    def tapered(
        cls,
        radius: float = 0.15,
        blades: int = 2,
        chord_root: float = 0.035,
        chord_tip: float = 0.015,
        twist_root: float = 18.0,
        twist_tip: float = 8.0,
        n_stations: int = 25,
        root_cutout: float = 0.2,
        name: str = "tapered",
    ) -> "BladeGeometry":
        r_R = np.linspace(root_cutout, 0.99, n_stations)
        t = (r_R - root_cutout) / max(0.99 - root_cutout, 1e-9)
        chord = chord_root * (1 - t) + chord_tip * t
        twist = twist_root * (1 - t) + twist_tip * t
        return cls(radius, blades, r_R, chord, twist, root_cutout, name)

    @classmethod
    def elliptical_chord(
        cls,
        radius: float = 0.15,
        blades: int = 2,
        chord_max: float = 0.03,
        twist_root: float = 16.0,
        twist_tip: float = 6.0,
        n_stations: int = 25,
        root_cutout: float = 0.2,
        name: str = "elliptical",
    ) -> "BladeGeometry":
        r_R = np.linspace(root_cutout, 0.99, n_stations)
        # half-ellipse from root_cutout→tip
        xi = (r_R - root_cutout) / max(0.99 - root_cutout, 1e-9)
        chord = chord_max * np.sqrt(np.clip(1.0 - (2 * xi - 1.0) ** 2, 0.0, 1.0))
        chord = np.maximum(chord, 0.005)
        twist = twist_root * (1 - xi) + twist_tip * xi
        return cls(radius, blades, r_R, chord, twist, root_cutout, name)

    def stations_m(self) -> np.ndarray:
        return self.r_R * self.radius

    def hash_id(self) -> str:
        payload = {
            "name": self.name,
            "R": self.radius,
            "B": self.blades,
            "r_R": self.r_R.tolist(),
            "chord": self.chord.tolist(),
            "twist": self.twist_deg.tolist(),
            "cut": self.root_cutout,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def interpolate_s(self, other: "BladeGeometry", s: float) -> "BladeGeometry":
        """Linear morph between self (s=0) and other (s=1)."""
        s = float(np.clip(s, 0.0, 1.0))
        if not np.allclose(self.r_R, other.r_R):
            # resample other onto self stations
            chord_o = np.interp(self.r_R, other.r_R, other.chord)
            twist_o = np.interp(self.r_R, other.r_R, other.twist_deg)
        else:
            chord_o = other.chord
            twist_o = other.twist_deg
        return BladeGeometry(
            radius=self.radius * (1 - s) + other.radius * s,
            blades=self.blades,
            r_R=self.r_R.copy(),
            chord=self.chord * (1 - s) + chord_o * s,
            twist_deg=self.twist_deg * (1 - s) + twist_o * s,
            root_cutout=self.root_cutout,
            name=f"morph({self.name}->{other.name},s={s:.3f})",
            meta={"s": s, "from": self.hash_id(), "to": other.hash_id()},
        )


def apply_collective(geom: BladeGeometry, collective_deg: float) -> BladeGeometry:
    g = BladeGeometry(
        radius=geom.radius,
        blades=geom.blades,
        r_R=geom.r_R.copy(),
        chord=geom.chord.copy(),
        twist_deg=geom.twist_deg + collective_deg,
        root_cutout=geom.root_cutout,
        name=geom.name,
        meta=dict(geom.meta),
    )
    g.meta["collective_deg"] = collective_deg
    return g
