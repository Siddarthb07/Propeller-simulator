# Propeller-simulator

Python project for simulating propeller performance across RPM, pitch, and blade count, with both interactive GUI and command-line execution.

## What The Simulator Includes

### Physics Core (`main.py`)

- `PropellerPhysics` class with BEMT-style force estimation
- Lift/drag approximation via angle-of-attack model
- Tip-speed and tip-Mach computation
- Two model modes:
  - `simple_bemt` (no induction; teaching / baseline)
  - `full_bemt` (axial induction + Prandtl tip loss; still no \(a'\)/hub/Re polars)

### Dynamic State

- `SimulationState` models RPM and pitch ramping toward targets
- Allows transient behavior instead of instant parameter jumps

### Visualization

- Matplotlib 2D top view + force vectors
- Matplotlib 3D propeller view
- Rolling history panel for thrust/power/efficiency
- Animated run loop via `FuncAnimation`

### GUI Controls (Tkinter)

- RPM slider/input
- Pitch slider/input
- Blade count selector
- Start/Pause animation
- Snapshot logging

### Logging

- CSV log output in `propeller_logs/`
- Fields include thrust, power, efficiency, disk loading, power loading, tip speed/mach, Cl/Cd

## Analysis Script (`analyze_propeller.py`)

- Loads one or more simulation CSV files
- Produces:
  - summary stats CSV
  - cleaned combined CSV
  - multiple figures (time series, histograms, correlation, scatter)
- Optionally generates DOCX paper if `python-docx` is installed
- Packs outputs into `propeller_analysis_output/propeller_analysis_package.zip`

## Dependencies

From `requirements.txt`:

- `numpy`
- `matplotlib`
- `pandas`
- `scipy`

Optional for DOCX output in analysis script:

- `python-docx`

## Run

### GUI mode (default)

```bash
python main.py
```

### CLI mode

```bash
python main.py --rpm 3000 --pitch 18 --blades 4 --model simple_bemt
```

## Analysis Workflow

1. Run simulation and generate logs in `propeller_logs/`
2. Update the `files` list in `analyze_propeller.py` to point to desired CSV(s)
3. Run:

```bash
python analyze_propeller.py
```

4. Collect outputs from `propeller_analysis_output/`

## V&V / AIAA artifacts

```bash
python -m vv.bench_bemt
# or regenerate both repos + copy into paper folder:
python scripts/regen_aiaa_artifacts.py
```

Outputs land in `vv/out/` (CSV error tables, summary JSON, plots). Assumptions: `docs/ASSUMPTIONS.md`.

## Known Limitations

- `full_bemt` has axial induction + tip loss but not \(a'\), hub loss, or Re polars
- Airfoil polar database integration is placeholder-only
- `simple_bemt` omits induction and overpredicts static thrust (see V&V)
- GUI `efficiency` is not hover FoM or propulsive efficiency; use V&V FoM instead

