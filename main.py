import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd

import tkinter as tk
from tkinter import ttk


# Physics: BEMT framework


class PropellerPhysics:
    """
    Physics engine with switchable models:
    - model="simple_bemt": current working simplified BEMT
    - model="full_bemt": placeholder for iterative induced-velocity BEMT. [web:75][web:81]
    """

    def __init__(self, radius=0.5, blades=3, chord=0.1, rho=1.225,
                 model="full_bemt", airfoil_name="NACA2412"):
        self.radius = radius
        self.blades = blades
        self.chord = chord
        self.rho = rho  # kg/m^3
        self.model = model
        self.airfoil_name = airfoil_name

    # --- basic Cl/Cd model ---

    def cl_cd_from_aoa(self, aoa_deg: float):
        aoa = np.radians(aoa_deg)
        cl = 2.0 * np.pi * np.sin(aoa) * np.cos(aoa)
        cd = 0.008 + 0.03 * (np.sin(aoa) ** 2)
        return max(0.0, cl), max(0.001, cd)

    # --- real polar hook (future integration of UIUC database) ---

    def cl_cd_from_database(self, aoa_deg: float, reynolds: float):
        """
        Placeholder: in future, load CSV polars for self.airfoil_name and
        interpolate Cl/Cd vs AoA and Re. [web:87]
        Currently falls back to analytical thin-airfoil model.
        """
        return self.cl_cd_from_aoa(aoa_deg)

    # --- compressibility helper ---

    def tip_mach(self, rpm: float, speed_of_sound: float = 343.0):
        omega = 2.0 * np.pi * rpm / 60.0
        tip_speed = omega * self.radius
        return tip_speed / speed_of_sound, tip_speed

    # --- public interface ---

    def bem_forces(self, rpm: float, pitch_deg: float, advance_ratio: float = 0.0):
        if self.model == "full_bemt":
            return self._bem_full(rpm, pitch_deg, advance_ratio)
        else:
            return self._bem_simple(rpm, pitch_deg, advance_ratio)

    # --- simplified BEMT (working) ---

    def _bem_simple(self, rpm: float, pitch_deg: float, advance_ratio: float):
        omega = 2.0 * np.pi * rpm / 60.0
        r = np.linspace(0.2, 1.0, 25) * self.radius

        thrust = 0.0
        torque = 0.0
        dr = r[1] - r[0]

        cl_list, cd_list = [], []

        for ri in r:
            phi = np.arctan2(advance_ratio * self.radius, ri)
            aoa_deg = pitch_deg - np.degrees(phi)
            cl, cd = self.cl_cd_from_aoa(aoa_deg)
            cl_list.append(cl)
            cd_list.append(cd)

            V_rel = omega * ri
            dL = 0.5 * self.rho * V_rel ** 2 * self.chord * dr * cl
            dD = 0.5 * self.rho * V_rel ** 2 * self.chord * dr * cd

            dT = self.blades * (dL * np.cos(phi) - dD * np.sin(phi))
            dQ = self.blades * ri * (dD * np.cos(phi) + dL * np.sin(phi))

            thrust += dT
            torque += dQ

        power = torque * omega
        efficiency = (thrust * 9.81) / power if power > 1e-6 else 0.0

        disk_area = np.pi * self.radius ** 2
        disk_loading = thrust / disk_area if disk_area > 0 else 0.0
        power_loading = thrust / power if power > 1e-6 else 0.0
        tip_mach, tip_speed = self.tip_mach(rpm)

        return {
            "thrust": thrust,
            "power": power,
            "efficiency": efficiency,
            "cl_avg": float(np.mean(cl_list)),
            "cd_avg": float(np.mean(cd_list)),
            "disk_loading": disk_loading,
            "power_loading": power_loading,
            "tip_speed": tip_speed,
            "tip_mach": tip_mach,
        }

    # --- full BEMT placeholder (research extension) ---

    def _bem_full(self, rpm: float, pitch_deg: float, advance_ratio: float):
        """
        TODO:
        - Iterate axial/tangential induction (a, a') using Glauert's method
        - Apply Prandtl tip & hub loss factors
        - Use cl_cd_from_database for Re-dependent polars
        - Include radial inflow and helical wake effects. [web:75][web:81]
        For now, just call simple model so simulator still runs.
        """
        return self._bem_simple(rpm, pitch_deg, advance_ratio)


# Transient state (rpm ramp)


class SimulationState:
    """
    Stores current vs target RPM and pitch to simulate ramp-up and lag.
    """

    def __init__(self, rpm0=0.0, pitch0=0.0):
        self.rpm = rpm0
        self.pitch = pitch0
        self.target_rpm = rpm0
        self.target_pitch = pitch0

    def update(self, dt: float, rpm_ramp: float = 800.0, pitch_rate: float = 10.0):
        # ramp RPM (rpm/s)
        if self.rpm < self.target_rpm:
            self.rpm = min(self.target_rpm, self.rpm + rpm_ramp * dt)
        else:
            self.rpm = max(self.target_rpm, self.rpm - rpm_ramp * dt)
        # ramp pitch (deg/s)
        if self.pitch < self.target_pitch:
            self.pitch = min(self.target_pitch, self.pitch + pitch_rate * dt)
        else:
            self.pitch = max(self.target_pitch, self.pitch - pitch_rate * dt)


# Data logging & reporting


class DataLogger:
    def __init__(self, log_dir="propeller_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = None

    def start_logging(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"simulation_{ts}.csv"
        header_df = pd.DataFrame(
            columns=[
                "timestamp",
                "rpm",
                "pitch_deg",
                "blades",
                "thrust_N",
                "power_W",
                "efficiency",
                "disk_loading_N_m2",
                "power_loading_N_W",
                "tip_speed_m_s",
                "tip_mach",
                "cl",
                "cd",
            ]
        )
        header_df.to_csv(self.log_file, index=False)

    def log_data(self, rpm, pitch, blades, forces):
        if self.log_file is None:
            self.start_logging()
        row = {
            "timestamp": datetime.now().isoformat(),
            "rpm": rpm,
            "pitch_deg": pitch,
            "blades": blades,
            "thrust_N": forces["thrust"],
            "power_W": forces["power"],
            "efficiency": forces["efficiency"],
            "disk_loading_N_m2": forces["disk_loading"],
            "power_loading_N_W": forces["power_loading"],
            "tip_speed_m_s": forces["tip_speed"],
            "tip_mach": forces["tip_mach"],
            "cl": forces["cl_avg"],
            "cd": forces["cd_avg"],
        }
        pd.DataFrame([row]).to_csv(self.log_file, mode="a", header=False, index=False)

class ReportGenerator:
    """
    Stub for future automatic PDF/HTML report generation. [web:69]
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file

    def generate_report(self, out_path: Path):
        # TODO: load CSV, make plots, and write PDF via ReportLab / LaTeX.
        print(f"[ReportGenerator] Would create report from {self.log_file} -> {out_path}")


# Visualization (Matplotlib)


class PropellerViz:
    def __init__(self):
        plt.style.use("dark_background")

        self.fig = plt.figure(figsize=(13, 5))
        self.ax2d = self.fig.add_subplot(1, 3, 1)
        self.ax3d = self.fig.add_subplot(1, 3, 2, projection="3d")
        self.ax_hist = self.fig.add_subplot(1, 3, 3)
        self.fig.suptitle("Aerodynamic Propeller Simulator", fontsize=14, fontweight="bold")

        self._setup_2d()
        self._setup_3d()
        self._setup_hist()

        self.time_data = {"thrust": [], "power": [], "eff": []}
        self.max_points = 200  # sliding window

    def _setup_2d(self):
        self.ax2d.clear()
        self.ax2d.set_xlim(-1.2, 1.2)
        self.ax2d.set_ylim(-1.2, 1.2)
        self.ax2d.set_aspect("equal")
        self.ax2d.grid(True, alpha=0.2)
        self.ax2d.set_title("Top View & Force Vectors")

    def _setup_3d(self):
        self.ax3d.clear()
        self.ax3d.set_xlim(-0.6, 0.6)
        self.ax3d.set_ylim(-0.6, 0.6)
        self.ax3d.set_zlim(-0.1, 0.6)
        self.ax3d.set_title("3D Propeller View")
        self.ax3d.set_xlabel("x")
        self.ax3d.set_ylabel("y")
        self.ax3d.set_zlabel("z")

    def _setup_hist(self):
        self.ax_hist.clear()
        self.ax_hist.grid(True, alpha=0.2)
        self.ax_hist.set_title("Performance (recent)")
        self.ax_hist.set_xlabel("Step")

    def update_blade_geometry(self, pitch_deg, blades=3, radius=0.5):
        self._setup_2d()
        theta = np.linspace(-0.35, 0.35, 40)
        for i in range(blades):
            blade_angle = i * 2 * np.pi / blades
            x = radius * np.cos(theta + blade_angle)
            y = radius * np.sin(theta + blade_angle) * np.cos(np.radians(pitch_deg))
            self.ax2d.fill(x, y, color="#5dade2", alpha=0.9, edgecolor="#2e86c1")
        self.ax2d.add_artist(plt.Circle((0, 0), 0.03, color="#ecf0f1"))

    def _draw_3d_prop(self, physics: PropellerPhysics, base_angle, pitch_deg):
        self._setup_3d()
        r = np.linspace(0.0, physics.radius, 30)
        z_profile = np.linspace(
            0.0, 0.25 * np.sin(np.radians(pitch_deg)), len(r)
        )
        for i in range(physics.blades):
            az = i * 2 * np.pi / physics.blades + base_angle
            x = r * np.cos(az)
            y = r * np.sin(az)
            self.ax3d.plot(x, y, z_profile, color="#5dade2", lw=3)
        self.ax3d.scatter([0], [0], [0], color="#ecf0f1", s=20)

    def animate(self, frame, state: SimulationState, blades_var,
                physics: PropellerPhysics, logger: DataLogger, dt=0.08):
        # update transient state
        state.update(dt)
        rpm = float(state.rpm)
        pitch = float(state.pitch)
        blades = int(blades_var.get())
        physics.blades = blades

        forces = physics.bem_forces(rpm, pitch)

        # --- 2D view ---
        self._setup_2d()
        omega = 2 * np.pi * rpm / 60.0 / 10.0
        base_angle = frame * omega
        t_blade = np.linspace(-0.35, 0.35, 40)

        for i in range(blades):
            a = i * 2 * np.pi / blades + base_angle
            x = physics.radius * np.cos(t_blade + a)
            y = physics.radius * np.sin(t_blade + a) * np.cos(np.radians(pitch))
            self.ax2d.fill(x, y, color="#5dade2", alpha=0.9, edgecolor="#2e86c1")

        self.ax2d.add_artist(plt.Circle((0, 0), 0.03, color="#ecf0f1"))
        self.ax2d.arrow(0, 0, 0, 0.4, head_width=0.04, head_length=0.04,
                        fc="lime", ec="lime", lw=2)
        self.ax2d.arrow(0, 0, 0.4, 0, head_width=0.04, head_length=0.04,
                        fc="orange", ec="orange", lw=2)

        tw_ratio = forces["thrust"] / 9.81 if 9.81 > 0 else 0.0

        info_text = (
            f"RPM: {rpm:.0f} (target {state.target_rpm:.0f})\n"
            f"Pitch: {pitch:.0f}°  Blades: {blades}\n"
            f"T: {forces['thrust']:.0f} N  P: {forces['power']:.0f} W\n"
            f"η: {forces['efficiency']:.1%}\n"
            f"Disk load: {forces['disk_loading']:.0f} N/m²\n"
            f"P load: {forces['power_loading']:.3f} N/W\n"
            f"Tip speed: {forces['tip_speed']:.1f} m/s\n"
            f"T/W (1 kg): {tw_ratio:.1f}"
        )
        if forces["tip_mach"] > 0.8:
            info_text += f"\nWARNING: Tip Mach={forces['tip_mach']:.2f} > 0.8"

        self.ax2d.text(
            -1.15,
            1.02,
            info_text,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#1f2833", alpha=0.85),
        )

        # --- 3D view ---
        self._draw_3d_prop(physics, base_angle, pitch)

        # --- history ---
        self.time_data["thrust"].append(forces["thrust"])
        self.time_data["power"].append(forces["power"])
        self.time_data["eff"].append(forces["efficiency"])
        for k in self.time_data:
            if len(self.time_data[k]) > self.max_points:
                self.time_data[k] = self.time_data[k][-self.max_points:]

        self._setup_hist()
        t = np.arange(len(self.time_data["thrust"]))
        self.ax_hist.plot(t, np.array(self.time_data["thrust"]), "g-", label="Thrust (N)")
        self.ax_hist.plot(t, np.array(self.time_data["power"]) / 10.0, "r-", label="Power (W/10)")
        self.ax_hist.plot(t, np.array(self.time_data["eff"]) * 100.0, "c-", label="η (%)")
        self.ax_hist.legend(loc="upper right", fontsize=8)

        # log for research
        logger.log_data(rpm, pitch, blades, forces)


# GUI (Tkinter)


class PropellerSimulator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Aerodynamic Propeller Simulator")
        self.root.configure(bg="#2b2b2b")

        self.viz = PropellerViz()
        self.physics = PropellerPhysics()
        self.logger = DataLogger()
        self.state = SimulationState(rpm0=2000.0, pitch0=15.0)
        self.anim = None

        self.rpm_var = tk.IntVar(value=2000)
        self.pitch_var = tk.IntVar(value=15)
        self.blades_var = tk.IntVar(value=3)

        self.setup_gui()
        self.update_viz()
        plt.ion()

    def setup_gui(self):
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="x")

        # RPM
        ttk.Label(frame, text="RPM (0–5000):").grid(row=0, column=0, sticky="w")
        rpm_scale = ttk.Scale(
            frame,
            from_=0,
            to=5000,
            orient="horizontal",
            command=self._on_rpm_slider,
        )
        rpm_scale.set(self.rpm_var.get())
        rpm_scale.grid(row=0, column=1, sticky="ew", padx=5)
        self.rpm_entry = ttk.Entry(frame, width=7, textvariable=self.rpm_var)
        self.rpm_entry.grid(row=0, column=2)
        self.rpm_entry.bind("<Return>", lambda e: self._sync_from_entry())

        # Pitch
        ttk.Label(frame, text="Pitch (0–60°):").grid(row=1, column=0, sticky="w")
        pitch_scale = ttk.Scale(
            frame,
            from_=0,
            to=60,
            orient="horizontal",
            command=self._on_pitch_slider,
        )
        pitch_scale.set(self.pitch_var.get())
        pitch_scale.grid(row=1, column=1, sticky="ew", padx=5)
        self.pitch_entry = ttk.Entry(frame, width=7, textvariable=self.pitch_var)
        self.pitch_entry.grid(row=1, column=2)
        self.pitch_entry.bind("<Return>", lambda e: self._sync_from_entry())

        # Blades
        ttk.Label(frame, text="Blades:").grid(row=2, column=0, sticky="w")
        self.blades_spin = ttk.Spinbox(
            frame, from_=1, to=6, width=5, textvariable=self.blades_var,
            command=self.update_viz
        )
        self.blades_spin.grid(row=2, column=1, sticky="w")

        # Buttons
        ttk.Button(frame, text="Start / Resume",
                   command=self.start_animation).grid(row=3, column=0, pady=6)
        ttk.Button(frame, text="Pause",
                   command=self.stop_animation).grid(row=3, column=1, pady=6)
        ttk.Button(frame, text="Log Snapshot",
                   command=self.export_single_step).grid(row=3, column=2, pady=6)

        frame.columnconfigure(1, weight=1)

    def _on_rpm_slider(self, val):
        self.rpm_var.set(int(float(val)))
        self.state.target_rpm = float(self.rpm_var.get())
        self.update_viz()

    def _on_pitch_slider(self, val):
        self.pitch_var.set(int(float(val)))
        self.state.target_pitch = float(self.pitch_var.get())
        self.update_viz()

    def _sync_from_entry(self):
        self.rpm_var.set(max(0, min(5000, int(self.rpm_var.get()))))
        self.pitch_var.set(max(0, min(60, int(self.pitch_var.get()))))
        self.blades_var.set(max(1, min(6, int(self.blades_var.get()))))
        self.state.target_rpm = float(self.rpm_var.get())
        self.state.target_pitch = float(self.pitch_var.get())
        self.update_viz()

    def update_viz(self):
        self.physics.blades = int(self.blades_var.get())
        self.viz.update_blade_geometry(
            int(self.pitch_var.get()), int(self.blades_var.get())
        )

    def start_animation(self):
        if self.anim is None:
            self.logger.start_logging()
            self.anim = FuncAnimation(
                self.viz.fig,
                self.viz.animate,
                fargs=(self.state, self.blades_var, self.physics, self.logger),
                interval=80,
                blit=False,
                cache_frame_data=False,
            )
            plt.draw()
            plt.pause(0.01)

    def stop_animation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def export_single_step(self):
        forces = self.physics.bem_forces(
            float(self.state.rpm), float(self.state.pitch)
        )
        self.logger.log_data(
            float(self.state.rpm),
            float(self.state.pitch),
            int(self.blades_var.get()),
            forces,
        )
        print("Snapshot logged to", self.logger.log_file)

    def run(self):
        self.root.mainloop()
        plt.ioff()
        plt.show()


# CLI mode


def cli_mode():
    parser = argparse.ArgumentParser(description="Propeller Simulator CLI")
    parser.add_argument("--rpm", type=float, default=2000)
    parser.add_argument("--pitch", type=float, default=15)
    parser.add_argument("--blades", type=int, default=3)
    parser.add_argument("--model", type=str, default="simple_bemt",
                        choices=["simple_bemt", "full_bemt"])
    args = parser.parse_args()

    physics = PropellerPhysics(blades=args.blades, model=args.model)
    forces = physics.bem_forces(args.rpm, args.pitch)

    print(f"RPM {args.rpm:.0f}, pitch {args.pitch:.1f}°, blades {args.blades}")
    print(f"Thrust: {forces['thrust']:.1f} N")
    print(f"Power:  {forces['power']:.0f} W")
    print(f"η:      {forces['efficiency']:.1%}")
    print(f"Disk loading:  {forces['disk_loading']:.1f} N/m²")
    print(f"Power loading: {forces['power_loading']:.3f} N/W")
    print(f"Tip speed:     {forces['tip_speed']:.1f} m/s, Mach {forces['tip_mach']:.2f}")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        app = PropellerSimulator()
        app.run()
