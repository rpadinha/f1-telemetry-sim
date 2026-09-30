import os
import sys
import fastf1
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
from scipy.interpolate import splev, splprep

def main():
    print("[PYTHON] Initializing FastF1 multi-lap processor...")

    if len(sys.argv) < 4:
        print("[PYTHON] Usage: python fetch_f1data.py <year> <gp> <session>", file=sys.stderr)
        sys.exit(1)

    year = int(sys.argv[1])
    gp = sys.argv[2]
    session_type = sys.argv[3]
    print(f"[PYTHON] Loading {year} {gp} {session_type} for Max Verstappen (VER)...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)

    # 1. Filter all accurate push laps set by Max Verstappen
    ver_laps = session.laps.pick_drivers("VER").pick_quicklaps()
    valid_laps = ver_laps[ver_laps["IsAccurate"] == True]

    if valid_laps.empty:
        print("[PYTHON] Warning: No accurate VER quick laps found. Falling back to overall fastest lap.")
        valid_laps = session.laps.pick_fastest().to_frame().T
        best_lap = valid_laps.iloc[0]
    else:
        best_lap = valid_laps.pick_fastest()

    print(f"[PYTHON] Found {len(valid_laps)} valid push lap(s) from VER.")
    print(f"[PYTHON] Benchmark Lap Time: {best_lap['LapTime']} (Lap {best_lap['LapNumber']})")

    # 2. Canonical distance grid based on the fastest reference lap
    ref_tel = best_lap.get_telemetry()
    total_distance = ref_tel["Distance"].iloc[-1]
    uniform_distance = np.arange(0, total_distance, 1.0)

    circuit_info = session.get_circuit_info()
    angle_rad = -circuit_info.rotation / 180.0 * np.pi
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # 3. Collect, rotate, AND INTERPOLATE coordinates across all valid qualifying laps
    x_runs = []
    y_runs = []
    z_runs = []

    for _, lap in valid_laps.iterrows():
        tel = lap.get_telemetry()
        d = tel["Distance"].to_numpy()

        # Convert FOM decimeters to standard meters
        x_m = tel["X"].to_numpy() * 0.1
        y_m = tel["Y"].to_numpy() * 0.1
        z_m = tel["Z"].to_numpy() * 0.1

        # Rotate to official circuit orientation
        x_rot = x_m * cos_theta - y_m * sin_theta
        y_rot = x_m * sin_theta + y_m * cos_theta

        # MUST interpolate onto uniform_distance before appending!
        x_runs.append(pchip_interpolate(d, x_rot, uniform_distance))
        y_runs.append(pchip_interpolate(d, y_rot, uniform_distance))
        z_runs.append(pchip_interpolate(d, z_m, uniform_distance))

    # 4. Ensemble average across laps (cancels out zero-mean random GPS jitter)
    avg_x = np.mean(x_runs, axis=0)
    avg_y = np.mean(y_runs, axis=0)
    avg_z = np.mean(z_runs, axis=0)

    # 5. Continuous spline fitting with realistic GPS noise variance (s = N * sigma^2)
    # 5755 points with ~0.8m GPS noise -> s = 5755 * (0.8^2) ≈ 3600
    s_realistic = len(uniform_distance) * (0.8 ** 2)
    tck, u = splprep([avg_x, avg_y], s=s_realistic, k=3)

    # Evaluate analytical 1st and 2nd derivatives directly from the smooth spline
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    # Curvature invariant to parameterization speed
    kappa = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)

    # Clean straight cut: kappa < 0.00067 rad/m corresponds to radius > 1500m
    radius = np.where(kappa > (1.0 / 2500.0), 1.0 / kappa, 10000.0)
    radius = np.clip(radius, a_min=25.0, a_max=10000.0)

    # 7. Interpolate benchmark comparison channels from the fastest lap
    d_ref = ref_tel["Distance"].to_numpy()
    speed_interp = pchip_interpolate(d_ref, ref_tel["Speed"].to_numpy(), uniform_distance)
    rpm_interp = pchip_interpolate(d_ref, ref_tel["RPM"].to_numpy(), uniform_distance)
    gear_interp = np.round(pchip_interpolate(d_ref, ref_tel["nGear"].to_numpy(), uniform_distance)).astype(int)
    throttle_interp = pchip_interpolate(d_ref, ref_tel["Throttle"].to_numpy(), uniform_distance) / 100.0
    brake_interp = np.round(pchip_interpolate(d_ref, ref_tel["Brake"].to_numpy(), uniform_distance))
    drs_interp = np.where(pchip_interpolate(d_ref, ref_tel["DRS"].to_numpy(), uniform_distance) >= 10, 1, 0)

    # Segment length vector
    seg_lengths = np.diff(uniform_distance, prepend=1.0)
    seg_lengths[0] = 1.0

    df_out = pd.DataFrame({
        "Segment_Length": seg_lengths,
        "Radius": radius,
        "X": avg_x,
        "Y": avg_y,
        "Z": avg_z,
        "DRS": drs_interp,
        "Real Speed": speed_interp,
        "RPM": rpm_interp,
        "nGear": gear_interp,
        "Throttle": throttle_interp,
        "Brake": brake_interp
    })

    output_dir = f"../data/{year}_{gp}_{session_type}/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{year}_{gp}_{session_type}.csv")

    df_out[["Segment_Length", "Radius", "X", "Y", "Z", "DRS", "Real Speed", "RPM", "nGear", "Throttle", "Brake"]].to_csv(output_path, index=False)

    # df_out[["Radius"]].to_csv(output_path, index=False)
    print(f"[PYTHON] Successfully processed {len(df_out)} segments from {len(valid_laps)} lap(s).")
    print(f"[PYTHON] Exported to: {output_path}")


if __name__ == "__main__":
    main()