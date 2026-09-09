import os
import fastf1
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate

print("Initializing FastF1...")

if not os.path.exists("cache"):
  os.makedirs("cache")
fastf1.Cache.enable_cache("cache")

# Loading the session data (Ex: Monza 2025 Qualifying)
session = fastf1.get_session(2025, "Monza", "Q")
session.load(telemetry=True, weather=False, messages=False)

lap = session.laps.pick_fastest()
tel = lap.get_telemetry()
print(
    f"Telemetry loaded. {lap['Driver']} set the fastest lap: {lap['LapTime']}."
)

# Interpolating telemetry data to have a uniform distance step (1 meter) for better curvature calculation
# and x, y. z speed. rpm and gear.
distancia_original = tel["Distance"].to_numpy()
total_distance = distancia_original[-1]
# Fixed segments of 1 meter
distancia_uniforme = np.arange(0, total_distance, 1.0)

# Interpolate
# Using 'pchip' (Piecewise Cubic Hermite Interpolating Polynomial)
x_interp = pchip_interpolate(distancia_original, tel["X"].to_numpy(), distancia_uniforme)
y_interp = pchip_interpolate(distancia_original, tel["Y"].to_numpy(), distancia_uniforme)
z_interp = pchip_interpolate(distancia_original, tel["Z"].to_numpy(), distancia_uniforme)
speed_interp = pchip_interpolate(distancia_original, tel["Speed"].to_numpy(), distancia_uniforme)
rpm_interp = pchip_interpolate(distancia_original, tel["RPM"].to_numpy(), distancia_uniforme)
# nGear goes from 1 to 8, so we can round the interpolated values to the nearest integer
gear_interp = np.round(pchip_interpolate(distancia_original, tel["nGear"].to_numpy(), distancia_uniforme)).astype(int)
throttle_pedal_interp = np.round(pchip_interpolate(distancia_original, tel["Throttle"].to_numpy(), distancia_uniforme))
brake_pedal_interp = np.round(pchip_interpolate(distancia_original, tel["Brake"].to_numpy(), distancia_uniforme))

df = pd.DataFrame({
    "Distance": distancia_uniforme,
    "X": x_interp,
    "Y": y_interp,
    "Z": z_interp,
    "Real Speed": speed_interp,
    "RPM": rpm_interp,
    "nGear": gear_interp,
    "Throttle": throttle_pedal_interp,
    "Brake": brake_pedal_interp
})

df["Throttle"] = df["Throttle"] / 100
df["Brake"] = df["Brake"] / 100

# Track Rotation
circuit_info = session.get_circuit_info()
# Converts the oficial circuit rotation
angle_rad = -circuit_info.rotation / 180 * np.pi

cos_theta = np.cos(angle_rad)
sin_theta = np.sin(angle_rad)

# Appliying the correct rotation
x_rot = df["X"] * cos_theta - df["Y"] * sin_theta
y_rot = df["X"] * sin_theta + df["Y"] * cos_theta

df["X"] = x_rot
df["Y"] = y_rot

# Curvature radius calculation
dx = np.gradient(df["X"])
dy = np.gradient(df["Y"])
ddx = np.gradient(dx)
ddy = np.gradient(dy)

denominator = (dx**2 + dy**2) ** 1.5 + 1e-8
curvature = np.abs(dx * ddy - dy * ddx) / denominator

# Defines the maximum radius for curves
df["Radius"] = np.where(curvature > 1e-3, 1 / curvature, 10000)

# Smoothing
df["Radius"] = (
    df["Radius"].rolling(window=5, min_periods=1, center=True).mean()
)

# Segment length now will be always 1 meter
df["Segment_Length"] = df["Distance"].diff().fillna(df["Distance"].iloc[0])

# Exporting to ../data folder
if not os.path.exists("../data"):
  os.makedirs("../data")

output_path = "../data/monza_pole.csv"
df[["Segment_Length", "Radius", "X", "Y", "Real Speed", "RPM", "nGear", "Throttle", "Brake"]].to_csv(output_path, index=False)

print(f"Success! Exported {len(df)} segments to {output_path}")