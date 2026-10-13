import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BG_CANVAS = "#0e1015"
BG_AXES = "#14171f"
GRID_COLOR = "#282c37"
LEGEND_BG = "#1e222d"

CORNERS = {350: "Variante Rettifilo", 1650: "Curva Grande", 2200: "Variante Roggia",
           2750: "Lesmo 1", 3150: "Lesmo 2", 4400: "Variante Ascari", 5350: "Parabolica"}


def apply_style(ax):
    ax.set_facecolor(BG_AXES)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.7)


def make_figure(n_panels, title, height_ratios=None, height_per_panel=4.5):
    """Creates a figure with n stacked panels sharing the x axis."""
    fig, axs = plt.subplots(n_panels, 1, figsize=(16, height_per_panel * n_panels + 1), sharex=True,
                            gridspec_kw={"height_ratios": height_ratios or [1] * n_panels})
    fig.patch.set_facecolor(BG_CANVAS)
    axs = np.atleast_1d(axs)
    for ax in axs:
        apply_style(ax)
    axs[0].set_title(title, color="white", fontsize=13, weight="bold")
    axs[-1].set_xlabel("Track Distance (m)", color="white", fontsize=11)
    return fig, axs


def add_corner_markers(axs, min_len, label_ax=None):
    """Vertical corner lines on every panel, names only on one panel (placed in axes coords)."""
    label_ax = label_ax if label_ax is not None else axs[0]
    for dist_m, name in CORNERS.items():
        if dist_m < min_len:
            for ax in axs:
                ax.axvline(dist_m, color="#4f5666", linestyle=":", alpha=0.5)
            label_ax.text(dist_m + 15, 0.04, name, color="#8b949e", fontsize=8, rotation=90,
                          transform=label_ax.get_xaxis_transform(), va="bottom")


def legend(ax, loc="upper right", ncol=3, handles=None, labels=None):
    if handles is not None:
        ax.legend(handles, labels, loc=loc, facecolor=LEGEND_BG, edgecolor="none", ncol=ncol)
    else:
        ax.legend(loc=loc, facecolor=LEGEND_BG, edgecolor="none", ncol=ncol)


def save(fig, output_dir, filename_clean, suffix):
    fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.07, hspace=0.12)
    out = os.path.join(output_dir, f"{filename_clean}_{suffix}.png")
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PYTHON] -> {os.path.basename(out)}")


def main():
    if len(sys.argv) < 3:
        print("[PYTHON] Error: Missing file arguments. Usage: plot_dynamics.py <path_real> <path_sim>", file=sys.stderr)
        sys.exit(1)

    path_real = sys.argv[1]
    path_sim = sys.argv[2]

    # --- Year/GP/session name ---
    filename_clean = os.path.splitext(os.path.basename(path_real))[0]
    parts = filename_clean.split('_')
    gp_name = parts[1].capitalize() if len(parts) > 1 else "Track"

    output_dir = os.path.dirname(path_real) if os.path.dirname(path_real) else "../data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_real = pd.read_csv(path_real)
    df_sim = pd.read_csv(path_sim)

    min_len = min(len(df_real), len(df_sim))
    df_real = df_real.iloc[:min_len].copy()
    df_sim = df_sim.iloc[:min_len].copy()
    distance = np.arange(min_len)

    # Reconstruct Real-World Forces
    real_mass_kg = 804.0
    v_real_ms = df_real["Real Speed"] / 3.6
    dv_ds_real = np.gradient(v_real_ms, 1.0)
    f_long_real_kn = (real_mass_kg * v_real_ms * dv_ds_real) / 1000.0

    if "Radius" in df_real.columns:
        radius_m = np.maximum(np.abs(df_real["Radius"].to_numpy()), 25.0)
        f_lat_real_kn = np.where(radius_m < 10000, (real_mass_kg * (v_real_ms ** 2) / radius_m) / 1000.0, 0.0)
    else:
        f_lat_real_kn = np.zeros(min_len)

    plt.style.use("dark_background")
    print("[PYTHON] Plotting dynamics (split into focused graphs)...")

    # =========================================================================
    # PLOT 1: GRIP & LONGITUDINAL FORCES
    # =========================================================================
    fig, axs = make_figure(2, f"{gp_name} GRIP & FORCES | KAMM CIRCLE", height_ratios=[1.2, 1.2])

    axs[0].plot(distance, df_sim["Max_Grip_N"] / 1000.0, label="Max Grip Radius", color="#00e5ff", linewidth=1.5)
    axs[0].plot(distance, df_sim["Long_Grip_N"] / 1000.0, label="Available Long Grip", color="#ff9100", linestyle="--", linewidth=1.3)
    axs[0].plot(distance, df_sim["Lateral_Force_N"] / 1000.0, label="Sim Lateral Force", color="#e040fb", linewidth=1.3)
    axs[0].plot(distance, f_lat_real_kn, label="Real Lateral Force", color="#00d2be", alpha=0.6, linewidth=1.1)
    axs[0].set_ylabel("Grip / Lat\nForces (kN)")
    legend(axs[0], ncol=4)

    axs[1].plot(distance, f_long_real_kn, label="Real Net Long Force", color="#00d2be", alpha=0.55, linewidth=1.2)
    axs[1].plot(distance, df_sim["Applied_Long_Force_N"] / 1000.0, label="Sim Net Long Force", color="#ff5252", linewidth=1.4)
    axs[1].plot(distance, df_sim["Max_Traction_N"] / 1000.0, label="Rear Axle Max Traction", color="#ffd740", linewidth=1.1)
    axs[1].axhline(0, color="#6e7681", linestyle="--", linewidth=0.8)
    axs[1].set_ylabel("Longitudinal\nForces (kN)")
    axs[1].set_ylim(-48, 48)
    legend(axs[1], loc="lower right", ncol=3)

    add_corner_markers(axs, min_len)
    save(fig, output_dir, filename_clean, "grip")

    # =========================================================================
    # PLOT 2: SLIP ANGLE & CORNERING DRAG (PACEJKA)
    # =========================================================================
    fig, axs = make_figure(1, f"{gp_name} TYRE SLIP | PACEJKA MAGIC FORMULA", height_per_panel=5)
    ax_slip = axs[0]
    ax_drag = ax_slip.twinx()
    slip_degrees = df_sim["Slip_Angle"] * (180.0 / np.pi)

    line1 = ax_slip.plot(distance, slip_degrees, label="Slip Angle (Alpha)", color="#ffff00", linewidth=1.3)
    line2 = ax_drag.plot(distance, df_sim["Drag_Corner"] / 1000.0, label="Cornering Drag", color="#ff1744", linewidth=1.5, linestyle="--")
    ax_slip.set_ylabel("Slip Angle\n(Degrees)", color="#ffff00")
    ax_drag.set_ylabel("Cornering\nDrag (kN)", color="#ff1744")
    ax_slip.set_ylim(0, 8)

    lines = line1 + line2
    legend(ax_slip, ncol=2, handles=lines, labels=[l.get_label() for l in lines])

    add_corner_markers(axs, min_len)
    save(fig, output_dir, filename_clean, "slip")

    # =========================================================================
    # PLOT 3: AERODYNAMICS & ENGINE BRAKING
    # =========================================================================
    fig, axs = make_figure(1, f"{gp_name} AERODYNAMICS & ENGINE BRAKING", height_per_panel=5)
    axs[0].plot(distance, df_sim["Downforce_N"] / 1000.0, label="Aero Downforce", color="#2979ff", linewidth=1.5)
    axs[0].plot(distance, df_sim["Drag_Force_N"] / 1000.0, label="Aero Drag", color="#ff5252", linewidth=1.3)
    axs[0].plot(distance, df_sim["Engine_Braking_N"] / 1000.0, label="Engine Braking", color="#ab47bc", linestyle="--", linewidth=1.1)
    axs[0].set_ylabel("Aero & Eng\nBraking (kN)")
    legend(axs[0])

    add_corner_markers(axs, min_len)
    save(fig, output_dir, filename_clean, "aero")

    # =========================================================================
    # PLOT 4: TYRES (wheel loads + front temps + rear temps)
    # =========================================================================
    fig, axs = make_figure(3, f"{gp_name} TYRES | WHEEL LOADS & TEMPERATURES", height_ratios=[1.3, 1, 1])

    axs[0].plot(distance, df_sim["Load_FL_N"] / 1000.0, label="FL Load", color="#ffeb3b", linewidth=1.2)
    axs[0].plot(distance, df_sim["Load_FR_N"] / 1000.0, label="FR Load", color="#ff9800", linewidth=1.2, linestyle="--")
    axs[0].plot(distance, df_sim["Load_RL_N"] / 1000.0, label="RL Load", color="#00e676", linewidth=1.2)
    axs[0].plot(distance, df_sim["Load_RR_N"] / 1000.0, label="RR Load", color="#00b0ff", linewidth=1.2, linestyle="--")
    axs[0].set_ylabel("Wheel\nLoads (kN)")
    legend(axs[0], ncol=4)

    # shared y range so front and rear temps are directly comparable
    temp_cols = ["Tyre_Temp_FL_C", "Tyre_Temp_FR_C", "Tyre_Temp_RL_C", "Tyre_Temp_RR_C"]
    t_min = min(df_sim[c].min() for c in temp_cols) - 5.0
    t_max = max(max(df_sim[c].max() for c in temp_cols), 105.0) + 5.0

    axs[1].plot(distance, df_sim["Tyre_Temp_FL_C"], label="Temp FL", color="#ffeb3b", linewidth=1.3)
    axs[1].plot(distance, df_sim["Tyre_Temp_FR_C"], label="Temp FR", color="#ff9800", linewidth=1.3, linestyle="--")
    axs[1].axhline(105.0, color="#ffffff", linestyle=":", alpha=0.5, label="Optimal (105°C)")
    axs[1].set_ylabel("Front Tyre\nTemp (°C)")
    axs[1].set_ylim(t_min, t_max)
    legend(axs[1], ncol=3)

    axs[2].plot(distance, df_sim["Tyre_Temp_RL_C"], label="Temp RL", color="#00e676", linewidth=1.3)
    axs[2].plot(distance, df_sim["Tyre_Temp_RR_C"], label="Temp RR", color="#00b0ff", linewidth=1.3, linestyle="--")
    axs[2].axhline(105.0, color="#ffffff", linestyle=":", alpha=0.5, label="Optimal (105°C)")
    axs[2].set_ylabel("Rear Tyre\nTemp (°C)")
    axs[2].set_ylim(t_min, t_max)
    legend(axs[2], ncol=3)

    add_corner_markers(axs, min_len)
    save(fig, output_dir, filename_clean, "tyres")

    fig, axs = make_figure(1, f"{gp_name} TYRES | Wear", height_per_panel=5)

    axs[0].plot(distance, df_sim["Tyre_Wear_FL"], label="Front Left Wear", color="#ffeb3b", linewidth=1.2)
    axs[0].plot(distance, df_sim["Tyre_Wear_FR"], label="Front Right Wear", color="#EA1DED", linewidth=1.2)
    axs[0].plot(distance, df_sim["Tyre_Wear_RL"], label="Rear Left Wear", color="#df1218", linewidth=1.2)
    axs[0].plot(distance, df_sim["Tyre_Wear_RR"], label="Rear Right Wear", color="#3d07e0", linewidth=1.2)
    max_wear = df_sim["Tyre_Wear_FL"].max()
    axs[0].set_ylim(0, max_wear)
    legend(axs[0], ncol=4)
    add_corner_markers(axs, min_len)

    save(fig, output_dir, filename_clean, "wear")

    print(f"[PYTHON] Dynamics graphs generated successfully in '{output_dir}/' folder.")


if __name__ == "__main__":
    main()