import xarray as xr
import pandas as pd
import numpy as np
import glob
import re
import os

FOLDER = "era5_cutoff_check"   # adjust if your folder name differs
pattern = re.compile(r"era5_(\d{4})_(\d{2})\.(nc|grib)$")

files = glob.glob(os.path.join(FOLDER, "era5_*.nc")) + glob.glob(os.path.join(FOLDER, "era5_*.grib"))

target_files = []
for f in files:
    basename = os.path.basename(f)
    m = pattern.match(basename)
    if not m:
        continue
    year, month, ext = m.groups()
    if month in ("01", "07"):          # ONLY Jan/July — ignore everything else in the folder
        target_files.append((int(year), month, ext, f))

target_files.sort()
print(f"Found {len(target_files)} Jan/July files to process (out of {len(files)} total files in folder)")

# Sanity check: confirm we're not accidentally picking up other months
other_months_present = [f for f in files if os.path.basename(f) not in
                         [os.path.basename(t[3]) for t in target_files]]
print(f"Ignoring {len(other_months_present)} other-month files (background download in progress)")

records = []
failed = []

for year, month, ext, f in target_files:
    try:
        if ext == "nc":
            ds = xr.open_dataset(f)
        else:
            ds = xr.open_dataset(f, engine="cfgrib")

        # Box-averaged means (consistent with your earlier decade analysis)
        sst = ds.sst.mean(dim=["latitude", "longitude"]).values
        t2m = ds.t2m.mean(dim=["latitude", "longitude"]).values
        d2m = ds.d2m.mean(dim=["latitude", "longitude"]).values
        u10 = ds.u10.mean(dim=["latitude", "longitude"]).values
        v10 = ds.v10.mean(dim=["latitude", "longitude"]).values
        wind_speed = np.sqrt(u10**2 + v10**2)

        delta_T = t2m - sst   # air - sea

        records.append({
            "year": year,
            "month": month,
            "format": ext,
            "sst_mean": float(np.nanmean(sst)),
            "t2m_mean": float(np.nanmean(t2m)),
            "d2m_mean": float(np.nanmean(d2m)),
            "deltaT_mean": float(np.nanmean(delta_T)),
            "wind_speed_mean": float(np.nanmean(wind_speed)),
        })
        ds.close()

    except Exception as e:
        failed.append((year, month, ext, str(e)))
        print(f"FAILED {year}-{month} ({ext}): {e}")

df = pd.DataFrame(records)
df = df.sort_values(["month", "year"]).reset_index(drop=True)
df.to_csv("jan_jul_master_1940_2025.csv", index=False)

print(f"\nSuccessfully processed: {len(records)}")
print(f"Failed: {len(failed)}")
print(f"\nSaved: jan_jul_master_1940_2025.csv")
print(f"\nFormat breakdown:")
print(df.groupby(["month", "format"]).size())

print(f"\nYear range check per month:")
for m in ["01", "07"]:
    sub = df[df["month"] == m]
    print(f"  Month {m}: {sub['year'].min()}–{sub['year'].max()}, {len(sub)} years present")
    missing_years = set(range(1940, 2026)) - set(sub["year"])
    if missing_years:
        print(f"    Missing years: {sorted(missing_years)}")