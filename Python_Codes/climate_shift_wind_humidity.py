import xarray as xr
import pandas as pd
import numpy as np
import glob
import re
import os
from scipy import stats

files = glob.glob("era5_cutoff_check/era5_*.nc") + glob.glob("era5_cutoff_check/era5_*.grib")
pattern = re.compile(r"era5_(\d{4})_(\d{2})\.(nc|grib)$")

target_files = []
for f in files:
    basename = os.path.basename(f)
    m = pattern.match(basename)
    if not m:
        continue
    year, month, ext = m.groups()
    if month in ("01", "07"):
        target_files.append((int(year), month, ext, f))

target_files.sort()
print(f"Processing {len(target_files)} files for wind/humidity trend check...")

records = []
for year, month, ext, f in target_files:
    try:
        ds = xr.open_dataset(f) if ext == "nc" else xr.open_dataset(f, engine="cfgrib")

        u10 = ds.u10.mean(dim=["latitude", "longitude"]).values
        v10 = ds.v10.mean(dim=["latitude", "longitude"]).values
        wind_speed = np.sqrt(u10**2 + v10**2)

        d2m = ds.d2m.mean(dim=["latitude", "longitude"]).values
        t2m = ds.t2m.mean(dim=["latitude", "longitude"]).values

        records.append({
            "year": year,
            "month": month,
            "wind_speed_mean": float(np.nanmean(wind_speed)),
            "d2m_mean": float(np.nanmean(d2m)),
            "t2m_mean": float(np.nanmean(t2m)),
        })
        ds.close()
    except Exception as e:
        print(f"FAILED {year}-{month}: {e}")

df = pd.DataFrame(records)
df["decade"] = (df["year"] // 10) * 10
df.to_csv("wind_humidity_by_year.csv", index=False)

baseline = df[df["year"] >= 2015]

print("\n=== DECADE COMPARISON: WIND SPEED ===")
for decade in sorted(df["decade"].unique()):
    decade_data = df[df["decade"] == decade]
    if len(decade_data) < 4:
        continue
    t_stat, p_value = stats.ttest_ind(decade_data["wind_speed_mean"], baseline["wind_speed_mean"], equal_var=False)
    diff = decade_data["wind_speed_mean"].mean() - baseline["wind_speed_mean"].mean()
    print(f"  {decade}s: mean wind = {decade_data['wind_speed_mean'].mean():.3f} m/s, diff = {diff:+.3f}, p = {p_value:.3f}, significant = {p_value < 0.05}")

print("\n=== DECADE COMPARISON: DEWPOINT (humidity proxy) ===")
for decade in sorted(df["decade"].unique()):
    decade_data = df[df["decade"] == decade]
    if len(decade_data) < 4:
        continue
    t_stat, p_value = stats.ttest_ind(decade_data["d2m_mean"], baseline["d2m_mean"], equal_var=False)
    diff = decade_data["d2m_mean"].mean() - baseline["d2m_mean"].mean()
    print(f"  {decade}s: mean Td = {decade_data['d2m_mean'].mean():.3f} K, diff = {diff:+.3f}, p = {p_value:.3f}, significant = {p_value < 0.05}")

print("\nSaved: wind_humidity_by_year.csv")