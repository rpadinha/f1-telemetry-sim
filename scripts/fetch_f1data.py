import os
import sys
import fastf1
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate, splprep, splev
from scipy.ndimage import gaussian_filter1d

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

    # 2. Canonical distance grid based on the fastest reference lap
    ref_tel = best_lap.get_telemetry()
    total_distance = ref_tel["Distance"].iloc[-1]

    circuit_info = session.get_circuit_info()
    angle_rad = -circuit_info.rotation / 180.0 * np.pi
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)


    # this could be done with bezier curves no?
    # to do: learn bezier curves
    # FIXED: Define an initial uniform distance grid to safely map and average our telemetry layers
    base_segment_length = 1.0
    initial_uniform_distance = np.arange(0, total_distance, base_segment_length)

    # 3. Collect, rotate, and interpolate coordinates
    x_runs, y_runs, z_runs = [], [], []

    for _, lap in valid_laps.iterrows():
        tel = lap.get_telemetry()
        d = tel["Distance"].to_numpy()

        # Scale telemetry down (FastF1 tracks in 10cm units, convert to absolute meters)
        x_m = tel["X"].to_numpy() * 0.1
        y_m = tel["Y"].to_numpy() * 0.1
        z_m = tel["Z"].to_numpy() * 0.1

        x_rot = x_m * cos_theta - y_m * sin_theta
        y_rot = x_m * sin_theta + y_m * cos_theta

        # Interpolate coordinates safely into the initial matching grid array
        x_runs.append(pchip_interpolate(d, x_rot, initial_uniform_distance))
        y_runs.append(pchip_interpolate(d, y_rot, initial_uniform_distance))
        z_runs.append(pchip_interpolate(d, z_m, initial_uniform_distance))

    # 4. Ensemble average across arrays to eliminate random GPS jitter noise
    avg_x_raw = np.mean(x_runs, axis=0)
    avg_y_raw = np.mean(y_runs, axis=0)
    avg_z_raw = np.mean(z_runs, axis=0)

    # 5. FIXED: Continuous Spline Fitting of the Averaged Circuit Shape
    # s tuning balancing value parameters: N * (sigma^2) -> 0.6m GPS noise variance
    s_realistic = len(initial_uniform_distance) * (0.6 ** 2)
    tck, u = splprep([avg_x_raw, avg_y_raw], s=s_realistic, k=3, per=True)

    # Re-evaluate the geometry on a perfectly defined final uniform distance matrix
    # This aligns our continuous parametric spline elements exactly with our target C++ track blocks
    final_segments_count = len(initial_uniform_distance)
    uniform_u = np.linspace(0, 1, final_segments_count)
    
    avg_x, avg_y = splev(uniform_u, tck, der=0)
    avg_z = pchip_interpolate(initial_uniform_distance, avg_z_raw, np.linspace(0, total_distance, final_segments_count))

    # Evaluate analytical exact derivatives directly out of the B-spline matrix
    dx, dy = splev(uniform_u, tck, der=1)
    ddx, ddy = splev(uniform_u, tck, der=2)

    # Calculate exact geometric local curvature (Kappa invariant to parametrization speed)
    raw_kappa = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    smoothed_kappa = gaussian_filter1d(raw_kappa, sigma=1.0)

    # Clean straight cut: kappa tracking limits < 0.0004 rad/m mapped out to 2000m bounds
    radius = np.where(smoothed_kappa > (1.0 / 2000.0), 1.0 / smoothed_kappa, 2000.0)
    radius = np.clip(radius, a_min=25.0, a_max=2500.0)

    # 6. FIXED: Pre-calculate static track geometry properties
    cross_z = (dx * ddy) - (dy * ddx)
    turn_dir = np.where(cross_z < 0.0, -1.0, 1.0)
    turn_dir[radius >= 2000.0] = 0.0

    # Calculate exact structural pitch angles using forward/back spatial gradients
    dz_ds = np.gradient(avg_z, base_segment_length)
    pitch_angle = np.arcsin(np.clip(dz_ds, -1.0, 1.0))

    # 7. Interpolate target benchmark performance data to match runtime segments
    d_ref = ref_tel["Distance"].to_numpy()
    target_grid = np.linspace(0, total_distance, final_segments_count)
    
    speed_interp = pchip_interpolate(d_ref, ref_tel["Speed"].to_numpy(), target_grid)
    rpm_interp = pchip_interpolate(d_ref, ref_tel["RPM"].to_numpy(), target_grid)
    gear_interp = np.round(pchip_interpolate(d_ref, ref_tel["nGear"].to_numpy(), target_grid)).astype(int)
    throttle_interp = pchip_interpolate(d_ref, ref_tel["Throttle"].to_numpy(), target_grid) / 100.0
    brake_interp = np.round(pchip_interpolate(d_ref, ref_tel["Brake"].to_numpy(), target_grid))
    drs_interp = np.where(pchip_interpolate(d_ref, ref_tel["DRS"].to_numpy(), target_grid) >= 10, 1, 0)

    seg_lengths = np.ones(final_segments_count) * base_segment_length

    # 8. Consolidate matrices into structured layout format
    df_out = pd.DataFrame({
        "Segment_Length": seg_lengths,
        "Radius": radius,
        "PitchAngle": pitch_angle,
        "TurnDirection": turn_dir,
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

    output_dir = os.path.join(script_dir, "..", "data", f"{year}_{gp}_{session_type}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{year}_{gp}_{session_type}.csv")

    # Export structured layout order matches standard C++ parser ingestion expectations
    export_cols = ["Segment_Length", "Radius", "PitchAngle", "TurnDirection", "X", "Y", "Z", "DRS", "Real Speed", "RPM", "nGear", "Throttle", "Brake"]
    df_out[export_cols].to_csv(output_path, index=False)

    print(f"[PYTHON] Successfully processed {len(df_out)} segments.")
    print(f"[PYTHON] Exported to: {output_path}")

if __name__ == "__main__":
    main()
