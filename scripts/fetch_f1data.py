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

    window_size = 10
    df["X"] = pd.Series(x_rot).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
    df["Y"] = pd.Series(y_rot).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()

    # Curvature radius calculation
    dx = np.gradient(df["X"])
    dy = np.gradient(df["Y"])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2) ** 1.5 + 1e-8

    curvature = numerator / denominator

    # Defines the maximum radius for curves
    df["Radius"] = np.where(curvature > 1e-4, 1 / curvature, 10000)

    df["Radius"] = df["Radius"].clip(upper=10000)

    # Segment length now will be always 1 meter
    df["Segment_Length"] = df["Distance"].diff().fillna(df["Distance"].iloc[0])

    # Exporting to ../data folder
    if not os.path.exists("../data"):
      os.makedirs("../data")

    output_path = f"../data/{year}_{gp}_{session_type}.csv"

    df[["Segment_Length", "Radius", "X", "Y", "Z", "DRS", "Real Speed", "RPM", "nGear", "Throttle", "Brake"]].to_csv(output_path, index=False)

    print(f"[PYTHON] Success! Exported {len(df)} segments to {output_path}")

if __name__ == "__main__":
    main()