import sys
import os
import fastf1
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate


def main():
    print("[PYTHON] Initializing FastF1...")

    if len(sys.argv) < 4:
       print("[PYTHON] Error: Not enough args in python? how?", file=sys.stderr)
       sys.exit(1)


    year = int(sys.argv[1])
    gp = sys.argv[2]
    session_type = sys.argv[3]
    print(f"[PYTHON] Getting {year} {gp} {session_type}")


    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "cache")
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)

    # Exporting to ../data folder
    if not os.path.exists("../data"):
        os.makedirs("../data")
    
    fastf1.Cache.enable_cache(cache_dir)

    # Loading the session data (Ex: Monza 2025 Qualifying)
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)

    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()
    print(
        f"[PYTHON] Telemetry loaded. {lap['Driver']} set the fastest lap: {lap['LapTime']}."
    )

    # Interpolating telemetry data to have a uniform distance step (1 meter) for better curvature calculation
    # and x, y. z speed. rpm and gear.
    original_distance = tel["Distance"].to_numpy()
    total_distance = original_distance[-1]
    # Fixed segments of 1 meter
    uniform_distance = np.arange(0, total_distance, 1.0)

    # Interpolate
    # Using 'pchip' (Piecewise Cubic Hermite Interpolating Polynomial)
    x_interp = pchip_interpolate(original_distance, tel["X"].to_numpy(), uniform_distance)
    y_interp = pchip_interpolate(original_distance, tel["Y"].to_numpy(), uniform_distance)
    z_interp = pchip_interpolate(original_distance, tel["Z"].to_numpy(), uniform_distance)
    speed_interp = pchip_interpolate(original_distance, tel["Speed"].to_numpy(), uniform_distance)
    rpm_interp = pchip_interpolate(original_distance, tel["RPM"].to_numpy(), uniform_distance)
    # nGear goes from 1 to 8, so we can round the interpolated values to the nearest integer
    gear_interp = np.round(pchip_interpolate(original_distance, tel["nGear"].to_numpy(), uniform_distance)).astype(int)
    throttle_pedal_interp = np.round(pchip_interpolate(original_distance, tel["Throttle"].to_numpy(), uniform_distance))
    brake_pedal_interp = np.round(pchip_interpolate(original_distance, tel["Brake"].to_numpy(), uniform_distance))
    is_drs_zone = np.round(pchip_interpolate(original_distance, tel["DRS"].to_numpy(), uniform_distance))
    is_drs_zone = np.where(is_drs_zone >= 10, 1, 0) # transforming into 1 or 0 true or false

    df = pd.DataFrame({
        "Distance": uniform_distance,
        "X": x_interp,
        "Y": y_interp,
        "Z": z_interp,
        "Real Speed": speed_interp,
        "RPM": rpm_interp,
        "nGear": gear_interp,
        "Throttle": throttle_pedal_interp,
        "Brake": brake_pedal_interp,
        "DRS": is_drs_zone
    })

    df["Throttle"] = df["Throttle"] / 100

    # Track Rotation
    circuit_info = session.get_circuit_info()
    # Converts the oficial circuit rotation
    angle_rad = -circuit_info.rotation / 180 * np.pi

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # Appliying the correct rotation
    x_rot = df["X"] * cos_theta - df["Y"] * sin_theta
    y_rot = df["X"] * sin_theta + df["Y"] * cos_theta

    df["X"] = pd.Series(x_rot).to_numpy()
    df["Y"] = pd.Series(y_rot).to_numpy()

    # 1. Compute track heading angle along distance s
    dx = np.gradient(df["X"], df["Distance"])
    dy = np.gradient(df["Y"], df["Distance"])

    # 2. Continuous heading angle (unwrap prevents -pi to +pi wrap jumps)
    heading = np.unwrap(np.arctan2(dy, dx))

    # 3. First derivative of heading gives exact signed curvature kappa (rad/m)
    # Smooth heading slightly with a 100-meter Savitzky-Golay or rolling filter
    from scipy.signal import savgol_filter
    heading_smooth = savgol_filter(heading, window_length=100, polyorder=2)
    curvature = np.abs(np.gradient(heading_smooth, df["Distance"]))

    # 4. Safe Radius clamping
    df["Radius"] = np.where(curvature > 0.01, 1.0 / curvature, 10000.0)
    df["Radius"] = df["Radius"].clip(lower=30.0, upper=10000.0)

    # Segment length now will be always 1 meter
    df["Segment_Length"] = df["Distance"].diff().fillna(1.0)

    output_dir = f"../data/{year}_{gp}_{session_type}/"
    if not os.path.exists(output_dir):
       os.makedirs(output_dir)

    output_path = output_dir + f"{year}_{gp}_{session_type}.csv"
    df[["Segment_Length", "Radius", "X", "Y", "Z", "DRS", "Real Speed", "RPM", "nGear", "Throttle", "Brake"]].to_csv(output_path, index=False)
    # debug line for radius
    #df[["Radius"]].to_csv(output_path, index=False)

    print(f"[PYTHON] Success! Exported {len(df)} segments to {output_path}")

if __name__ == "__main__":
    main()