import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print("[PYTHON] Error: Missing file arguments. Usage: plot_dynamics.py <path_real> <path_sim>", file=sys.stderr)
        sys.exit(1)

    path_real = sys.argv[1]
    path_sim = sys.argv[2]

    # --- Year/GP/session name ---
    filename_clean = os.path.splitext(os.path.basename(path_real))[0]
    parts = filename_clean.split('_')
    year = parts[0] if len(parts) > 1 else "_"
    gp_name = parts[1].lower() if len(parts) > 1 else "_"
    session_type = parts[2].lower() if len(parts) > 1 else "_"

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
    v_real_smooth = pd.Series(v_real_ms).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    dv_ds_real = np.gradient(v_real_smooth, 1.0)
    f_long_real_kn = (real_mass_kg * v_real_smooth * dv_ds_real) / 1000.0

    if "Radius" in df_real.columns:
        radius_m = np.maximum(np.abs(df_real["Radius"].to_numpy()), 25.0)
        f_lat_real_kn = np.where(radius_m < 5000.0, (real_mass_kg * (v_real_ms ** 2) / radius_m) / 1000.0, 0.0)
    else:
        f_lat_real_kn = np.zeros(min_len)

    # Configure Figure (Now with 6 subplots)
    plt.style.use("dark_background")
    fig, axs = plt.subplots(6, 1, figsize=(16, 18), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.4, 1.2, 1.2, 1.2, 1.2]})
    fig.patch.set_facecolor("#0e1015")

    for ax in axs:
        ax.set_facecolor("#14171f")
        ax.grid(True, color="#282c37", linestyle="--", linewidth=0.7, alpha=0.7)

    corners = {350: "Variante Rettifilo", 1650: "Curva Grande", 2200: "Variante Roggia", 2750: "Lesmo 1", 3150: "Lesmo 2", 4400: "Variante Ascari", 5350: "Parabolica"}
    for dist_m, name in corners.items():
        if dist_m < min_len:
            for ax in axs:
                ax.axvline(dist_m, color="#4f5666", linestyle=":", alpha=0.5)
            axs[0].text(dist_m + 15, 42, name, color="#8b949e", fontsize=8, rotation=90)

    # 1. GRIP LIMITS
    axs[0].plot(distance, df_sim["Max_Grip_N"] / 1000.0, label="Max Grip Radius", color="#00e5ff", linewidth=1.5)
    axs[0].plot(distance, df_sim["Long_Grip_N"] / 1000.0, label="Available Long Grip", color="#ff9100", linestyle="--", linewidth=1.3)
    axs[0].plot(distance, df_sim["Lateral_Force_N"] / 1000.0, label="Sim Lateral Force", color="#e040fb", linewidth=1.3)
    axs[0].plot(distance, f_lat_real_kn, label="Real Lateral Force", color="#00d2be", alpha=0.6, linewidth=1.1)
    axs[0].set_ylabel("Grip / Lat\nForces (kN)")
    axs[0].set_title("F1CarDynamics ANALYSIS | KAMM, PACEJKA & 4-WHEEL LOADS", color="white", fontsize=13, weight="bold")
    axs[0].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=4)

    # 2. LONGITUDINAL FORCES
    axs[1].plot(distance, f_long_real_kn, label="Real Net Long Force", color="#00d2be", alpha=0.55, linewidth=1.2)
    axs[1].plot(distance, df_sim["Applied_Long_Force_N"] / 1000.0, label="Sim Net Long Force", color="#ff5252", linewidth=1.4)
    axs[1].plot(distance, df_sim["Max_Traction_N"] / 1000.0, label="Rear Axle Max Traction", color="#ffd740", linestyle=":", linewidth=1.1)
    axs[1].axhline(0, color="#6e7681", linestyle="--", linewidth=0.8)
    axs[1].set_ylabel("Longitudinal\nForces (kN)")
    axs[1].set_ylim(-48, 28)
    axs[1].legend(loc="lower right", facecolor="#1e222d", edgecolor="none", ncol=3)

    # 3. PACEJKA MAGIC FORMULA (NEW)
    ax_slip = axs[2]
    ax_drag = ax_slip.twinx()
    slip_degrees = df_sim["Slip_Angle"] * (180.0 / np.pi)
    
    line1 = ax_slip.plot(distance, slip_degrees, label="Slip Angle (Alpha)", color="#ffff00", linewidth=1.3)
    line2 = ax_drag.plot(distance, df_sim["Drag_Corner"] / 1000.0, label="Cornering Drag", color="#ff1744", linewidth=1.5, linestyle="--")
    
    ax_slip.set_ylabel("Slip Angle\n(Degrees)", color="#ffff00")
    ax_drag.set_ylabel("Cornering\nDrag (kN)", color="#ff1744")
    ax_slip.set_ylim(0, 8) # F1 peak is usually around 5-6 degrees
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_slip.legend(lines, labels, loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=2)

    # 4. AERODYNAMICS
    axs[3].plot(distance, df_sim["Downforce_N"] / 1000.0, label="Aero Downforce", color="#2979ff", linewidth=1.5)
    axs[3].plot(distance, df_sim["Drag_Force_N"] / 1000.0, label="Aero Drag", color="#ff5252", linewidth=1.3)
    axs[3].plot(distance, df_sim["Engine_Braking_N"] / 1000.0, label="Engine Braking", color="#ab47bc", linestyle="--", linewidth=1.1)
    axs[3].set_ylabel("Aero & Eng\nBraking (kN)")
    axs[3].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=3)

    # 5. 4-WHEEL LOADS
    axs[4].plot(distance, df_sim["Load_FL_N"] / 1000.0, label="FL Load", color="#ffeb3b", linewidth=1.2)
    axs[4].plot(distance, df_sim["Load_FR_N"] / 1000.0, label="FR Load", color="#ff9800", linewidth=1.2, linestyle="--")
    axs[4].plot(distance, df_sim["Load_RL_N"] / 1000.0, label="RL Load", color="#00e676", linewidth=1.2)
    axs[4].plot(distance, df_sim["Load_RR_N"] / 1000.0, label="RR Load", color="#00b0ff", linewidth=1.2, linestyle="--")
    axs[4].set_ylabel("Wheel\nLoads (kN)")
    axs[4].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=4)

    # 6. TYRE TEMPERATURES
    axs[5].plot(distance, df_sim["Tyre_Temp_FL_C"], label="Temp FL", color="#ffeb3b", linewidth=1.2)
    axs[5].plot(distance, df_sim["Tyre_Temp_FR_C"], label="Temp FR", color="#ff9800", linewidth=1.2, linestyle="--")
    axs[5].plot(distance, df_sim["Tyre_Temp_RL_C"], label="Temp RL", color="#00e676", linewidth=1.2)
    axs[5].plot(distance, df_sim["Tyre_Temp_RR_C"], label="Temp RR", color="#00b0ff", linewidth=1.2, linestyle="--")
    axs[5].axhline(105.0, color="#ffffff", linestyle=":", alpha=0.5, label="Optimal (105°C)")
    axs[5].set_ylabel("Tyre\nTemp (°C)")
    axs[5].set_xlabel("Track Distance (m)", fontsize=11)
    axs[5].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=5)

    plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.04, hspace=0.15)
    plt.savefig(os.path.join(output_dir, f"{filename_clean}_dynamics.png"), dpi=300, facecolor=fig.get_facecolor())
    print(f"[PYTHON] Dynamics plot saved successfully.")

if __name__ == "__main__":
    main()