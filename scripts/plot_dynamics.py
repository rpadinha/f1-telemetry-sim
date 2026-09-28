import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("[PYTHON] Plotting dynamics of sim vs real (splitting into focused graphs)...")
    if len(sys.argv) < 3:
        print("[PYTHON] Error: Missing file arguments. Usage: plot_dynamics.py <path_real> <path_sim>", file=sys.stderr)
        sys.exit(1)

    path_real = sys.argv[1]
    path_sim = sys.argv[2]

    if not os.path.exists(path_real) or not os.path.exists(path_sim):
        print(f"[PYTHON] CSV files not found:\n -> {path_real}\n -> {path_sim}")
        sys.exit(1)

    # --- Year/GP/session name ---
    filename_clean = os.path.splitext(os.path.basename(path_real))[0]
    parts = filename_clean.split('_')
    year = parts[0] if len(parts) > 1 else "_"
    gp_name = parts[1].lower() if len(parts) > 1 else "_"
    session_type = parts[2].lower() if len(parts) > 1 else "_"

    output_dir = os.path.dirname(path_real) if os.path.dirname(path_real) else "../data"
    output_dir += "/" + year + "_" + gp_name + "_" + session_type

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_real = pd.read_csv(path_real)
    df_sim = pd.read_csv(path_sim)

    min_len = min(len(df_real), len(df_sim))
    df_real = df_real.iloc[:min_len].copy()
    df_sim = df_sim.iloc[:min_len].copy()
    distance = np.arange(min_len)

    # 2. Reconstruct Real-World Forces from Verstappen Telemetry (Mass ~ 804 kg with fuel)
    real_mass_kg = 804.0
    v_real_ms = df_real["Real Speed"] / 3.6

    # Longitudinal acceleration: a_x = v * (dv / ds) smoothed over a 5m window to reduce sensor noise
    v_real_smooth = pd.Series(v_real_ms).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    dv_ds_real = np.gradient(v_real_smooth, 1.0)
    a_long_real = v_real_smooth * dv_ds_real
    f_long_real_kn = (real_mass_kg * a_long_real) / 1000.0

    # Real centrifugal lateral force demand: F_y = m * v^2 / R
    if "Radius" in df_real.columns:
        radius_m = np.maximum(np.abs(df_real["Radius"].to_numpy()), 25.0)
        f_lat_real_kn = np.where(radius_m < 5000.0, (real_mass_kg * (v_real_ms ** 2) / radius_m) / 1000.0, 0.0)
    else:
        f_lat_real_kn = np.zeros(min_len)

    # 3. Configure Dark Theme Figure
    plt.style.use("dark_background")
    fig, axs = plt.subplots(
        5, 1, figsize=(16, 15), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1.3, 1.3, 1.3]}
    )
    fig.patch.set_facecolor("#0e1015")

    for ax in axs:
        ax.set_facecolor("#14171f")
        ax.grid(True, color="#282c37", linestyle="--", linewidth=0.7, alpha=0.7)

    # Monza Key Corner Markers
    corners = {
        350: "Variante Rettifilo",
        1650: "Curva Grande",
        2200: "Variante Roggia",
        2750: "Lesmo 1",
        3150: "Lesmo 2",
        4400: "Variante Ascari",
        5350: "Parabolica",
    }
    for dist_m, name in corners.items():
        if dist_m < min_len:
            for ax in axs:
                ax.axvline(dist_m, color="#4f5666", linestyle=":", alpha=0.5)
            axs[0].text(dist_m + 15, 42, name, color="#8b949e", fontsize=8, rotation=90)

    # --- SUBPLOT 1: KAMM FRICTION CIRCLE & GRIP LIMITS ---
    axs[0].plot(distance, df_sim["Max_Grip_N"] / 1000.0, label="Max Grip Radius (F_grip,max)", color="#00e5ff", linewidth=1.5)
    axs[0].plot(distance, df_sim["Long_Grip_N"] / 1000.0, label="Available Longitudinal Grip (F_long,grip)", color="#ff9100", linestyle="--", linewidth=1.3)
    axs[0].plot(distance, df_sim["Lateral_Force_N"] / 1000.0, label="Sim Lateral Force (F_lat)", color="#e040fb", linewidth=1.3)
    axs[0].plot(distance, f_lat_real_kn, label="Real Lateral Force (Verstappen)", color="#00d2be", alpha=0.6, linewidth=1.1)
    axs[0].set_ylabel("Grip & Lateral\nForces (kN)", fontsize=10)
    axs[0].set_title("F1CarDynamics ANALYSIS | KAMM CIRCLE, TRACTION, AERO & 4-WHEEL LOADS (MONZA)", color="white", fontsize=13, weight="bold")
    axs[0].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=2)

    # --- SUBPLOT 2: LONGITUDINAL FORCES (NET FORCE, TRACTION & BRAKING) ---
    axs[1].plot(distance, f_long_real_kn, label="Real Net Longitudinal Force (m * a_x)", color="#00d2be", alpha=0.55, linewidth=1.2)
    axs[1].plot(distance, df_sim["Applied_Long_Force_N"] / 1000.0, label="Sim Net Longitudinal Force", color="#ff5252", linewidth=1.4)
    axs[1].plot(distance, df_sim["Max_Traction_N"] / 1000.0, label="Rear Axle Max Traction Limit", color="#ffd740", linestyle=":", linewidth=1.1)
    axs[1].plot(distance, df_sim["Desired_Engine_N"] / 1000.0, label="Desired Engine Drive Force", color="#69f0ae", alpha=0.6, linewidth=1.0)
    axs[1].axhline(0, color="#6e7681", linestyle="--", linewidth=0.8)
    axs[1].set_ylabel("Longitudinal\nForces (kN)", fontsize=10)
    axs[1].set_ylim(-48, 28)
    axs[1].legend(loc="lower right", facecolor="#1e222d", edgecolor="none", ncol=2)

    # --- SUBPLOT 3: AERODYNAMICS & RETARDING FORCES ---
    axs[2].plot(distance, df_sim["Downforce_N"] / 1000.0, label="Aerodynamic Downforce", color="#2979ff", linewidth=1.5)
    axs[2].plot(distance, df_sim["Drag_Force_N"] / 1000.0, label="Aerodynamic Drag", color="#ff1744", linewidth=1.3)
    axs[2].plot(distance, df_sim["Engine_Braking_N"] / 1000.0, label="Engine Braking", color="#ab47bc", linestyle="--", linewidth=1.1)
    axs[2].set_ylabel("Aero & Engine\nBraking (kN)", fontsize=10)
    axs[2].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=3)

    # --- SUBPLOT 4: 4-WHEEL VERTICAL LOAD DISTRIBUTION ---
    axs[3].plot(distance, df_sim["Load_FL_N"] / 1000.0, label="Load FL (Front-Left)", color="#ffeb3b", linewidth=1.2)
    axs[3].plot(distance, df_sim["Load_FR_N"] / 1000.0, label="Load FR (Front-Right)", color="#ff9800", linewidth=1.2, linestyle="--")
    axs[3].plot(distance, df_sim["Load_RL_N"] / 1000.0, label="Load RL (Rear-Left)", color="#00e676", linewidth=1.2)
    axs[3].plot(distance, df_sim["Load_RR_N"] / 1000.0, label="Load RR (Rear-Right)", color="#00b0ff", linewidth=1.2, linestyle="--")
    axs[3].set_ylabel("Vertical Wheel\nLoads (kN)", fontsize=10)
    axs[3].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=4)

    # --- SUBPLOT 5: 4-WHEEL INDEPENDENT TYRE TEMPERATURES ---
    axs[4].plot(distance, df_sim["Tyre_Temp_FL_C"], label="Temp FL (°C)", color="#ffeb3b", linewidth=1.2)
    axs[4].plot(distance, df_sim["Tyre_Temp_FR_C"], label="Temp FR (°C)", color="#ff9800", linewidth=1.2, linestyle="--")
    axs[4].plot(distance, df_sim["Tyre_Temp_RL_C"], label="Temp RL (°C)", color="#00e676", linewidth=1.2)
    axs[4].plot(distance, df_sim["Tyre_Temp_RR_C"], label="Temp RR (°C)", color="#00b0ff", linewidth=1.2, linestyle="--")
    axs[4].axhline(105.0, color="#ffffff", linestyle=":", alpha=0.5, label="Optimal Window (105°C)")
    axs[4].set_ylabel("4-Wheel Tyre\nTemp (°C)", fontsize=10)
    axs[4].set_xlabel("Track Distance (m)", fontsize=11)
    axs[4].legend(loc="upper right", facecolor="#1e222d", edgecolor="none", ncol=5)

    plt.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05, hspace=0.14)

    plt.savefig(os.path.join(output_dir, f"{filename_clean}_dynamics.png"), dpi=300, facecolor=fig.get_facecolor())
    print(f"[PYTHON] Dynamics plot saved to: {os.path.join(output_dir, f"{filename_clean}_dynamics.png")}")

if __name__ == "__main__":
    main()