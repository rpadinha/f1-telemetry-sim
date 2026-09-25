import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Plotting comparisons of sim vs real (splitting into focused graphs)...")
    if len(sys.argv) < 3:
        print("Error: Missing file arguments. Usage: script.py <path_real> <path_sim>", file=sys.stderr)
        sys.exit(1)

    path_real = sys.argv[1]
    path_sim = sys.argv[2]

    if not os.path.exists(path_real) or not os.path.exists(path_sim):
        print(f"[PYTHON] CSV files not found:\n -> {path_real}\n -> {path_sim}")
        sys.exit(1)

    # --- GP name ---
    filename_clean = os.path.splitext(os.path.basename(path_real))[0]
    parts = filename_clean.split('_')
    gp_name = parts[1].upper() if len(parts) > 1 else "F1"

    df_real = pd.read_csv(path_real)
    df_sim = pd.read_csv(path_sim)

    min_len = min(len(df_real), len(df_sim))
    df_real = df_real.iloc[:min_len].copy()
    df_sim = df_sim.iloc[:min_len].copy()
    distance = np.arange(min_len)

    # Normalizar pedais se necessário
    if df_real['Throttle'].max() > 1.5:
        df_real['Throttle'] = df_real['Throttle'] / 100.0
    if df_real['Brake'].max() > 1.5:
        df_real['Brake'] = df_real['Brake'] / 100.0

    # Cálculo do Delta
    v_real_ms = np.maximum(df_real['Real Speed'] / 3.6, 2.0)
    real_cum_time = np.cumsum(1.0 / v_real_ms)
    delta_time = df_sim['Sim_Time_s'] - real_cum_time
    final_delta = delta_time.iloc[-1]

    # Configurações globais de estilo (Mantendo a tua identidade visual escura)
    plt.style.use('dark_background')
    bg_canvas = '#0e1015'
    bg_axes = '#14171f'
    grid_color = '#282c37'

    def apply_style(ax):
        ax.set_facecolor(bg_axes)
        ax.grid(True, color=grid_color, linestyle='--', linewidth=0.7, alpha=0.7)

    # Descobre a pasta onde guardar os gráficos (mesma pasta do script ou subpasta data)
    output_dir = os.path.dirname(path_real) if os.path.dirname(path_real) else "../data"

    # =========================================================================
    # GRÁFICO 1: PERFORMANCE & TIME DELTA (Speed + Delta)
    # =========================================================================
    fig1, axs1 = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.2]})
    fig1.patch.set_facecolor(bg_canvas)
    for ax in axs1: apply_style(ax)

    # Canal de Velocidade
    axs1[0].plot(distance, df_real['Real Speed'], label="Real Data", color="#00d2be", linewidth=1.6)
    axs1[0].plot(distance, df_sim['Sim_Speed'], label="CUDA Simulator", color="#ff8800", linewidth=1.4, linestyle="--")
    axs1[0].set_ylabel("Speed\n(km/h)", color="white", fontsize=10)
    axs1[0].set_ylim(40, 365)
    axs1[0].legend(loc="lower left", facecolor="#1e222d", edgecolor="none")
    axs1[0].set_title(f"{gp_name} PERFORMANCE ANALYSIS | Lap Time Delta: {final_delta:+.3f}s", color="white", fontsize=13, weight="bold")

    # why yhis
    if gp_name == "MONZA":
        corners = {900: "Variante Rettifilo", 1500: "Curva Grande", 2100: "Variante Roggia", 2500: "Lesmo 1", 2850: "Lesmo 2", 3900: "Variante Ascari", 5000: "Parabolica"}
        for dist, name in corners.items():
            if dist < min_len:
                axs1[0].axvline(dist, color="#4f5666", linestyle=":", alpha=0.6)
                axs1[0].text(dist + 20, 70, name, color="#8b949e", fontsize=8, rotation=90)

    # Canal de Delta
    axs1[1].plot(distance, delta_time, color="#ffffff", linewidth=1.5)
    axs1[1].axhline(0, color="#6e7681", linestyle="--", linewidth=0.8)
    axs1[1].fill_between(distance, delta_time, 0, where=(delta_time > 0), color="#ff3344", alpha=0.35, interpolate=True)
    axs1[1].fill_between(distance, delta_time, 0, where=(delta_time <= 0), color="#00e676", alpha=0.35, interpolate=True)
    axs1[1].set_ylabel("Delta t\n(s)", color="white", fontsize=10)
    axs1[1].set_xlabel("Track Distance (m)", color="white", fontsize=11)

    plt.subplots_adjust(left=0.07, right=0.94, top=0.92, bottom=0.08, hspace=0.1)
    fig1.savefig(os.path.join(output_dir, f"{filename_clean}_performance.png"), dpi=300, facecolor=fig1.get_facecolor())

    # =========================================================================
    # PLOT 2: DRIVER INPUTS & POWERTRAIN (Pedals + RPM/Gear)
    # =========================================================================
    fig2, axs2 = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5]})
    fig2.patch.set_facecolor(bg_canvas)
    for ax in axs2: apply_style(ax)

    # Pedals
    axs2[0].plot(distance, df_real['Throttle'], label="Real Throttle", color="#00e676", alpha=0.5, linewidth=1.2)
    axs2[0].plot(distance, df_sim['Sim_Throttle'], label="Sim Throttle", color="#76ff03", linewidth=1.5, linestyle=":")
    axs2[0].plot(distance, -df_real['Brake'], label="Real Brake", color="#ff1744", alpha=0.5, linewidth=1.2)
    axs2[0].plot(distance, -df_sim['Sim_Brake'], label="Sim Brake", color="#ff5252", linewidth=1.5, linestyle=":")
    axs2[0].axhline(0, color="#4f5666", linewidth=0.8)
    axs2[0].set_ylabel("Pedals\n[-Brk / +Thr]", color="white", fontsize=10)
    axs2[0].set_yticks([-1, 0, 1])
    axs2[0].set_yticklabels(["1.0 (Brake)", "0", "1.0 (Gas)"])
    axs2[0].legend(loc="lower left", facecolor="#1e222d", edgecolor="none", ncol=4)
    axs2[0].set_title(f"{gp_name} DRIVER INPUTS & TELEMETRY", color="white", fontsize=13, weight="bold")

    # RPM and Gears
    ax_rpm = axs2[1]
    ax_gear = ax_rpm.twinx()
    
    line_rpm_real = ax_rpm.plot(distance, df_real['RPM'], label="Real RPM", color="#00bcd4", alpha=0.5, linewidth=1.0)
    line_rpm_sim  = ax_rpm.plot(distance, df_sim['Sim_RPM'], label="Sim RPM", color="#ffc107", linewidth=1.3)
    line_gear_real = ax_gear.plot(distance, df_real['nGear'], label="Real Gear", color="#9c27b0", alpha=0.6, linewidth=1.2, drawstyle='steps-post')
    line_gear_sim  = ax_gear.plot(distance, df_sim['Sim_Gear'], label="Sim Gear", color="#e040fb", linewidth=1.5, linestyle="--", drawstyle='steps-post')

    ax_rpm.set_ylabel("RPM", color="white", fontsize=10)
    ax_gear.set_ylabel("Gear", color="#e040fb", fontsize=10)
    ax_gear.set_ylim(0, 9)
    ax_gear.set_yticks(range(1, 9))
    ax_rpm.set_xlabel("Track Distance (m)", color="white", fontsize=11)

    lines = line_rpm_real + line_rpm_sim + line_gear_real + line_gear_sim
    labels = [l.get_label() for l in lines]
    ax_rpm.legend(lines, labels, loc="lower left", facecolor="#1e222d", edgecolor="none", ncol=4)

    plt.subplots_adjust(left=0.07, right=0.94, top=0.92, bottom=0.08, hspace=0.1)
    fig2.savefig(os.path.join(output_dir, f"{filename_clean}_inputs.png"), dpi=300, facecolor=fig2.get_facecolor())

    # =========================================================================
    # PLOT 3: THERMAL DYNAMICS (Tyre Temps)
    # =========================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 5))
    fig3.patch.set_facecolor(bg_canvas)
    apply_style(ax3)

    ax3.plot(distance, df_sim['Tyre_Temp_Front_C'], label="Front Tyres (Sim)", color="#ff9100", linewidth=1.4)
    ax3.plot(distance, df_sim['Tyre_Temp_Rear_C'], label="Rear Tyres (Sim)", color="#d50000", linewidth=1.4)
    ax3.axhline(105.0, color="#ffffff", linestyle=":", alpha=0.4, label="Optimal Window (105°C)")
    ax3.set_ylabel("Tyre Temp (°C)", color="white", fontsize=10)
    ax3.set_xlabel("Track Distance (m)", color="white", fontsize=11)
    ax3.set_title(f"{gp_name} TYRE THERMAL DYNAMICS", color="white", fontsize=13, weight="bold")
    ax3.legend(loc="lower left", facecolor="#1e222d", edgecolor="none", ncol=3)

    plt.subplots_adjust(left=0.07, right=0.94, top=0.88, bottom=0.15)
    fig3.savefig(os.path.join(output_dir, f"{filename_clean}_thermals.png"), dpi=300, facecolor=fig3.get_facecolor())

    print(f"[PYTHON] 3 Graphs generated successfully in '{output_dir}/' folder:")
    print(f" -> {filename_clean}_performance.png")
    print(f" -> {filename_clean}_inputs.png")
    print(f" -> {filename_clean}_thermals.png")
    
if __name__ == "__main__":
    main()
