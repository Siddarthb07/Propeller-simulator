# Propeller-simulator — modeling assumptions (AIAA honesty lock)

## Shipping model (`simple_bemt`)

- Blade-element integration with analytical thin-airfoil Cl/Cd.
- **No** iterative axial/tangential induction (`a`, `a'`).
- **No** Prandtl tip/hub loss in the shipping path.
- Advance ratio enters only through geometric inflow angle; static V&V uses `J = 0`.
- Optional `BladeGeometry` (variable chord/twist); AIAA lock still uses uniform chord.
- Optional `freestream_axial` for mutual-induction coupling (DroneAero OS).
- No Re-dependent polar database (hook is a stub).

## `full_bemt`

- Implements axial induction iteration + Prandtl tip loss (shares kernel with `vv/reference_bemt.py`).
- Still incomplete vs production rotor codes: **no** tangential induction \(a'\), **no** hub loss, **no** Re-dependent polar database.
- Default CLI/GUI may still use `simple_bemt` for teaching; V&V compares both.

## Efficiency metric in the GUI

- The logged `efficiency = (thrust * 9.81) / power` is **not** propulsive efficiency and **not** hover figure of merit.
- Paper / V&V tables use momentum-theory ideal power and inductive-BEMT reference instead.

## V&V reference used for AIAA

- Inductive BEMT + Prandtl tip loss in `vv/reference_bemt.py` (same Cl/Cd family); `full_bemt` wires this kernel (identity check).
- Ablation contrast is `simple_bemt` vs inductive path.
- Actuator-disk ideal hover power for FoM context on the inductive solution.
- Weak external anchor: static \(C_T\) vs a published-range band (~0.05–0.15), not a matched catalog propeller.
- Not flight-test validated (yet).
