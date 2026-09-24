import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_real = "../data/monza_pole.csv"
path_sim = "../data/sim_telemetry_monza.csv"

if not os.path.exists(path_real) or not os.path.exists(path_sim):
    print("[PYTHON] CSV files not found in ../data/")
    exit()

## reading csv files both for simulated and real telemetry
df_real = pd.read_csv(path_real)
df_sim = pd.read_csv(path_sim)

min_len = min(len(df_real), len(df_sim))
df_real = df_real.iloc[:min_len].copy()
df_sim = df_sim.iloc[:min_len].copy()
distance = np.arange(min_len)

if df_real['Throttle'].max() > 1.5:
    df_real['Throttle'] = df_real['Throttle'] / 100.0
if df_real['Brake'].max() > 1.5:
    df_real['Brake'] = df_real['Brake'] / 100.0

v_real_ms = np.maximum(df_real['Real Speed'] / 3.6, 2.0)
real_cum_time = np.cumsum(1.0 / v_real_ms)
delta_time = df_sim['Sim_Time_s'] - real_cum_time

plt.style.use('dark_background')
fig, axs = plt.subplots(5, 1, figsize=(16, 12), sharex=True, 
                        gridspec_kw={'height_ratios': [2.5, 1.2, 1.8, 1.5, 1.2]})
fig.patch.set_facecolor('#0e1015')

for ax in axs:
    ax.set_facecolor('#14171f')
    ax.grid(True, color='#282c37', linestyle='--', linewidth=0.7, alpha=0.7)

axs[0].plot(distance, df_real['Real Speed'], label="Real (Verstappen Pole)", color="#00d2be", linewidth=1.6)
axs[0].plot(distance, df_sim['Sim_Speed'], label="CUDA Simulator", color="#ff8800", linewidth=1.4, linestyle="--")
axs[0].set_ylabel("Speed\n(km/h)", color="white", fontsize=10)
axs[0].set_ylim(40, 365)
axs[0].legend(loc="lower left", facecolor="#1e222d", edgecolor="none")
axs[0].set_title(f"MONZA TELEMETRY ANALYSIS | Lap Time Delta: {delta_time.iloc[-1]:+.3f}s", color="white", fontsize=13, weight="bold")

corners = {
    900: "Variante Rettifilo",
    1500: "Curva Grande",
    2100: "Variante Roggia",
    2500: "Lesmo 1",
    2850: "Lesmo 2",
    3900: "Variante Ascari",
    5000: "Parabolica"
}

for dist, name in corners.items():
    if dist < min_len:
        axs[0].axvline(dist, color="#4f5666", linestyle=":", alpha=0.6)
        axs[0].text(dist + 20, 70, name, color="#8b949e", fontsize=8, rotation=90)

axs[1].plot(distance, delta_time, color="#ffffff", linewidth=1.5)
axs[1].axhline(0, color="#6e7681", linestyle="--", linewidth=0.8)
axs[1].fill_between(distance, delta_time, 0, where=(delta_time > 0), color="#ff3344", alpha=0.35, interpolate=True)
axs[1].fill_between(distance, delta_time, 0, where=(delta_time <= 0), color="#00e676", alpha=0.35, interpolate=True)
axs[1].set_ylabel("Delta t\n(s)", color="white", fontsize=10)

axs[2].plot(distance, df_real['Throttle'], label="Real Throttle", color="#00e676", alpha=0.5, linewidth=1.2)
axs[2].plot(distance, df_sim['Sim_Throttle'], label="Sim Throttle", color="#76ff03", linewidth=1.5, linestyle=":")
axs[2].plot(distance, -df_real['Brake'], label="Real Brake", color="#ff1744", alpha=0.5, linewidth=1.2)
axs[2].plot(distance, -df_sim['Sim_Brake'], label="Sim Brake", color="#ff5252", linewidth=1.5, linestyle=":")
axs[2].axhline(0, color="#4f5666", linewidth=0.8)
axs[2].set_ylabel("Pedals\n[-Brk / +Thr]", color="white", fontsize=10)
axs[2].set_yticks([-1, 0, 1])
axs[2].set_yticklabels(["1.0 (Brake)", "0", "1.0 (Gas)"])
axs[2].legend(loc="lower left", facecolor="#1e222d", edgecolor="none", ncol=2)

ax_rpm = axs[3]
ax_gear = ax_rpm.twinx()

line_rpm_real = ax_rpm.plot(distance, df_real['RPM'], label="Real RPM", color="#00bcd4", alpha=0.5, linewidth=1.0)
line_rpm_sim  = ax_rpm.plot(distance, df_sim['Sim_RPM'], label="Sim RPM", color="#ffc107", linewidth=1.3)
line_gear_real = ax_gear.plot(distance, df_real['nGear'], label="Real Gear", color="#9c27b0", alpha=0.6, linewidth=1.2, drawstyle='steps-post')
line_gear_sim  = ax_gear.plot(distance, df_sim['Sim_Gear'], label="Sim Gear", color="#e040fb", linewidth=1.5, linestyle="--", drawstyle='steps-post')

ax_rpm.set_ylabel("RPM", color="white", fontsize=10)
ax_gear.set_ylabel("Gear", color="#e040fb", fontsize=10)
ax_gear.set_ylim(0, 9)
ax_gear.set_yticks(range(1, 9))

lines = line_rpm_real + line_rpm_sim + line_gear_real + line_gear_sim
labels = [l.get_label() for l in lines]
ax_rpm.legend(lines, labels, loc="lower left", facecolor="#1e222d", edgecolor="none", ncol=4)

axs[4].plot(distance, df_sim['Tyre_Temp_Front_C'], label="Front Tyres", color="#ff9100", linewidth=1.4)
axs[4].plot(distance, df_sim['Tyre_Temp_Rear_C'], label="Rear Tyres", color="#d50000", linewidth=1.4)
axs[4].axhline(105.0, color="#ffffff", linestyle=":", alpha=0.4, label="Optimal Window (105°C)")
axs[4].set_ylabel("Tyre Temp\n(°C)", color="white", fontsize=10)
axs[4].set_xlabel("Track Distance (m)", color="white", fontsize=11)
axs[4].legend(loc="lower left", facecolor="#1e222d", edgecolor="none", ncol=3)

plt.subplots_adjust(left=0.07, right=0.94, top=0.94, bottom=0.05, hspace=0.15)
output_img = "../data/monza_telemetry_comparison.png"
plt.savefig(output_img, dpi=300, facecolor=fig.get_facecolor())
print(f"[PYTHON] Telemetry graph saved in: {output_img}")
plt.show()