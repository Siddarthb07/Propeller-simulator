"""
V&V harness for Propeller-simulator simple_bemt.

References:
1) Textbook inductive BEMT (vv/reference_bemt.py) — same Cl/Cd, with axial induction
2) Actuator-disk ideal power (momentum theory) for FoM at matched thrust
3) Quadratic RPM scaling self-check (T ∝ n² at fixed pitch)

Usage:
  python -m vv.bench_bemt
  python vv/bench_bemt.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from propeller_physics import PropellerPhysics  # noqa: E402
from vv.independent_uniform_inflow import uniform_inflow_static_forces  # noqa: E402
from vv.reference_bemt import inductive_bemt_forces, momentum_ideal_power  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# UAV-scale envelope (aligned with Drone-Vortex-Ring-Simulation defaults)
RADIUS = 0.15
BLADES = 2
CHORD = 0.025
RHO = 1.225
PITCH = 12.0
RPMS = list(range(2000, 8001, 500))


def pct_err(pred, ref):
    if abs(ref) < 1e-12:
        return float("nan")
    return 100.0 * (pred - ref) / ref


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run():
    ship = PropellerPhysics(
        radius=RADIUS, blades=BLADES, chord=CHORD, rho=RHO, model="simple_bemt"
    )
    full = PropellerPhysics(
        radius=RADIUS, blades=BLADES, chord=CHORD, rho=RHO, model="full_bemt"
    )

    rows = []
    for rpm in RPMS:
        s = ship.bem_forces(rpm, PITCH, advance_ratio=0.0)
        f = full.bem_forces(rpm, PITCH, advance_ratio=0.0)
        ref = inductive_bemt_forces(
            rpm, PITCH, radius=RADIUS, blades=BLADES, chord=CHORD, rho=RHO, advance_ratio=0.0
        )
        uni = uniform_inflow_static_forces(
            rpm, PITCH, RADIUS, BLADES, CHORD, rho=RHO
        )
        p_ideal_at_ref_T = momentum_ideal_power(ref["thrust"], RADIUS, RHO)
        p_ideal_at_full_T = momentum_ideal_power(f["thrust"], RADIUS, RHO)
        p_ideal_at_ship_T = momentum_ideal_power(s["thrust"], RADIUS, RHO)
        fom_ref = (
            p_ideal_at_ref_T / ref["power"] if ref["power"] > 1e-9 else float("nan")
        )
        fom_full = (
            p_ideal_at_full_T / f["power"] if f["power"] > 1e-9 else float("nan")
        )
        # Static thrust coeff CT = T / (ρ n² D⁴), n in rev/s, D=2R
        n = rpm / 60.0
        D = 2.0 * RADIUS
        denom = RHO * (n**2) * (D**4)
        ct_simple = s["thrust"] / denom if denom > 0 else float("nan")
        ct_full = f["thrust"] / denom if denom > 0 else float("nan")
        ct_ref = ref["thrust"] / denom if denom > 0 else float("nan")
        ct_uni = uni["thrust"] / denom if denom > 0 else float("nan")
        rows.append(
            {
                "rpm": rpm,
                "pitch_deg": PITCH,
                "simple_thrust_N": s["thrust"],
                "simple_power_W": s["power"],
                "full_thrust_N": f["thrust"],
                "full_power_W": f["power"],
                "ref_thrust_N": ref["thrust"],
                "ref_power_W": ref["power"],
                "uniform_inflow_thrust_N": uni["thrust"],
                "uniform_inflow_power_W": uni["power"],
                "thrust_ratio_simple_over_ref": s["thrust"] / ref["thrust"]
                if abs(ref["thrust"]) > 1e-12
                else float("nan"),
                "thrust_ratio_full_over_ref": f["thrust"] / ref["thrust"]
                if abs(ref["thrust"]) > 1e-12
                else float("nan"),
                "thrust_ratio_full_over_uniform": f["thrust"] / uni["thrust"]
                if abs(uni["thrust"]) > 1e-12
                else float("nan"),
                "thrust_pct_err_simple_vs_ref": pct_err(s["thrust"], ref["thrust"]),
                "thrust_pct_err_full_vs_ref": pct_err(f["thrust"], ref["thrust"]),
                "thrust_pct_err_full_vs_uniform": pct_err(f["thrust"], uni["thrust"]),
                "power_pct_err_simple_vs_ref": pct_err(s["power"], ref["power"]),
                "power_pct_err_full_vs_ref": pct_err(f["power"], ref["power"]),
                "CT_simple": ct_simple,
                "CT_full": ct_full,
                "CT_ref": ct_ref,
                "CT_uniform": ct_uni,
                "momentum_ideal_power_at_ref_T_W": p_ideal_at_ref_T,
                "momentum_ideal_power_at_full_T_W": p_ideal_at_full_T,
                "momentum_ideal_power_at_ship_T_W": p_ideal_at_ship_T,
                "hover_fom_ref": fom_ref,
                "hover_fom_full": fom_full,
                "tip_mach": s["tip_mach"],
            }
        )

    rpm_arr = np.array([r["rpm"] for r in rows], dtype=float)
    t_simple = np.array([r["simple_thrust_N"] for r in rows], dtype=float)
    t_full = np.array([r["full_thrust_N"] for r in rows], dtype=float)
    t_ref = np.array([r["ref_thrust_N"] for r in rows], dtype=float)
    k = float(np.sum(t_simple * rpm_arr**2) / np.sum(rpm_arr**4))
    t_quad = k * rpm_arr**2

    t_uni = np.array([r["uniform_inflow_thrust_N"] for r in rows], dtype=float)
    # Literature-band anchor from UIUC-style small-prop static CT0 ranges
    # (Brandt & Selig 2011; Deters et al. 2017): commercial MR props often
    # ~0.05–0.15, but poorer / low-pitch / low-Re cases can sit near ~0.02.
    # Educational untwisted blades are expected near the low end.
    CT_LIT_LO, CT_LIT_HI = 0.015, 0.15
    ct_ref_mean = float(np.nanmean([r["CT_ref"] for r in rows]))
    ct_full_mean = float(np.nanmean([r["CT_full"] for r in rows]))
    ct_simple_mean = float(np.nanmean([r["CT_simple"] for r in rows]))
    ct_uni_mean = float(np.nanmean([r["CT_uniform"] for r in rows]))
    ratio_vals = [r["thrust_ratio_simple_over_ref"] for r in rows]
    ratio_span = float(np.nanmax(ratio_vals) - np.nanmin(ratio_vals))

    csv_path = OUT / "bemt_error.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "geometry": {
            "radius_m": RADIUS,
            "blades": BLADES,
            "chord_m": CHORD,
            "rho": RHO,
            "pitch_deg": PITCH,
            "advance_ratio": 0.0,
        },
        "n_points": len(rows),
        "endpoint_thrust_N": {
            "rpm_min": int(rows[0]["rpm"]),
            "simple": rows[0]["simple_thrust_N"],
            "full": rows[0]["full_thrust_N"],
            "ref": rows[0]["ref_thrust_N"],
            "uniform": rows[0]["uniform_inflow_thrust_N"],
            "rpm_max": int(rows[-1]["rpm"]),
            "simple_max": rows[-1]["simple_thrust_N"],
            "full_max": rows[-1]["full_thrust_N"],
            "ref_max": rows[-1]["ref_thrust_N"],
            "uniform_max": rows[-1]["uniform_inflow_thrust_N"],
        },
        "endpoint_power_W": {
            "simple_min": rows[0]["simple_power_W"],
            "full_min": rows[0]["full_power_W"],
            "uniform_min": rows[0]["uniform_inflow_power_W"],
            "simple_max": rows[-1]["simple_power_W"],
            "full_max": rows[-1]["full_power_W"],
            "uniform_max": rows[-1]["uniform_inflow_power_W"],
        },
        "simple_vs_ref": {
            "thrust_rmse_N": rmse(t_simple, t_ref),
            "thrust_mean_abs_pct_err": float(
                np.nanmean([abs(r["thrust_pct_err_simple_vs_ref"]) for r in rows])
            ),
            "mean_thrust_ratio": float(np.nanmean(ratio_vals)),
            "thrust_ratio_span_across_rpm": ratio_span,
            "power_rmse_W": rmse(
                [r["simple_power_W"] for r in rows], [r["ref_power_W"] for r in rows]
            ),
            "power_mean_abs_pct_err": float(
                np.nanmean([abs(r["power_pct_err_simple_vs_ref"]) for r in rows])
            ),
        },
        "full_vs_ref_identity": {
            "thrust_rmse_N": rmse(t_full, t_ref),
            "mean_thrust_ratio": float(
                np.nanmean([r["thrust_ratio_full_over_ref"] for r in rows])
            ),
            "note": "Shared kernel — integration/identity check only",
        },
        "full_vs_independent_uniform_inflow": {
            "thrust_rmse_N": rmse(t_full, t_uni),
            "thrust_mean_abs_pct_err": float(
                np.nanmean([abs(r["thrust_pct_err_full_vs_uniform"]) for r in rows])
            ),
            "mean_thrust_ratio": float(
                np.nanmean([r["thrust_ratio_full_over_uniform"] for r in rows])
            ),
            "note": "Independent code path (uniform v_i + momentum closure; no tip loss)",
        },
        "structural_T_propto_n2_check": {
            "quadratic_k_T_per_rpm2": k,
            "thrust_rmse_vs_quadratic_fit_N": rmse(t_simple, t_quad),
            "note": "Self-consistency of simple_bemt form, not external validation",
        },
        "CT_static_mean": {
            "simple": ct_simple_mean,
            "full": ct_full_mean,
            "ref": ct_ref_mean,
            "uniform_inflow": ct_uni_mean,
            "literature_band_lo": CT_LIT_LO,
            "literature_band_hi": CT_LIT_HI,
            "literature_band_note": "UIUC-style small-prop static CT0 span incl. poor/low-Re cases (~0.015–0.15); not a matched catalog prop",
            "ref_inside_band": CT_LIT_LO <= ct_ref_mean <= CT_LIT_HI,
            "full_inside_band": CT_LIT_LO <= ct_full_mean <= CT_LIT_HI,
            "uniform_inside_band": CT_LIT_LO <= ct_uni_mean <= CT_LIT_HI,
            "simple_inside_band": CT_LIT_LO <= ct_simple_mean <= CT_LIT_HI,
        },
        "mean_hover_fom_ref": float(np.nanmean([r["hover_fom_ref"] for r in rows])),
        "mean_hover_fom_full": float(np.nanmean([r["hover_fom_full"] for r in rows])),
        "assumptions": [
            "simple_bemt: no iterative induced velocity (baseline / teaching path)",
            "full_bemt: axial induction + Prandtl tip loss; no a', hub loss, or Re polars",
            "full vs harness ref is an identity/wiring check (shared kernel)",
            "full vs uniform-inflow is an independent physics cross-check (different algorithm)",
            "UAV-scale static envelope (J=0); not flight-test validated",
            "CT band 0.015–0.15 spans UIUC-reported small-prop static CT0 including poor performers",
            "GUI 'efficiency' metric is not FoM or propulsive efficiency",
        ],
    }
    (OUT / "bemt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(rpm_arr, t_simple, "o-", label="simple_bemt")
    ax[0].plot(rpm_arr, t_full, "^-", label="full_bemt")
    ax[0].plot(rpm_arr, t_uni, "d-", label="indep. uniform inflow")
    ax[0].plot(rpm_arr, t_ref, "s--", label="harness ref (identity)")
    ax[0].set_xlabel("RPM")
    ax[0].set_ylabel("Thrust [N]")
    ax[0].set_title("Static thrust vs RPM")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(
        rpm_arr,
        [r["thrust_pct_err_simple_vs_ref"] for r in rows],
        "o-",
        label="simple vs ref",
    )
    ax[1].plot(
        rpm_arr,
        [r["thrust_pct_err_full_vs_uniform"] for r in rows],
        "d-",
        label="full vs uniform",
    )
    ax[1].plot(
        rpm_arr,
        [r["thrust_pct_err_full_vs_ref"] for r in rows],
        "^-",
        label="full vs ref (identity)",
    )
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xlabel("RPM")
    ax[1].set_ylabel("% error vs inductive ref")
    ax[1].set_title("Thrust error")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_thrust_vv.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rpm_arr, [r["simple_power_W"] for r in rows], "o-", label="simple_bemt")
    ax.plot(rpm_arr, [r["full_power_W"] for r in rows], "^-", label="full_bemt")
    ax.plot(rpm_arr, [r["ref_power_W"] for r in rows], "s--", label="harness ref")
    ax.plot(
        rpm_arr,
        [r["momentum_ideal_power_at_ref_T_W"] for r in rows],
        ":",
        label="momentum ideal @ ref T",
    )
    ax.set_xlabel("RPM")
    ax.set_ylabel("Power [W]")
    ax.set_title("Power comparison (static)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_power_vv.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    return summary


if __name__ == "__main__":
    run()
