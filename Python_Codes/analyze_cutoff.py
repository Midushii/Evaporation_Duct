import xarray as xr
import pandas as pd
import numpy as np
import glob
import re
import os
import matplotlib.pyplot as plt

files = glob.glob("era5_cutoff_check/era5_*.nc") + glob.glob("era5_cutoff_check/era5_*.grib")
pattern = re.compile(r"era5_(\d{4})_(\d{2})\.(nc|grib)$")

# Filter to only Jan/July files, ignore everything else in the folder
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
print(f"Processing {len(target_files)} Jan/Jul files...")

records = []
failed = []

for year, month, ext, f in target_files:
    try:
        if ext == "nc":
            ds = xr.open_dataset(f)
        else:  # grib
            ds = xr.open_dataset(f, engine="cfgrib")

        sst = ds.sst.mean(dim=["latitude", "longitude"]).values
        t2m = ds.t2m.mean(dim=["latitude", "longitude"]).values

        delta_T = t2m - sst

        records.append({
            "year": year,
            "month": month,
            "format": ext,
            "sst_mean": float(np.nanmean(sst)),
            "sst_std": float(np.nanstd(sst)),
            "t2m_mean": float(np.nanmean(t2m)),
            "delta_T_mean": float(np.nanmean(delta_T)),
            "delta_T_std": float(np.nanstd(delta_T)),
            "n_hours": len(sst)
        })
        ds.close()

    except Exception as e:
        failed.append((year, month, ext, str(e)))
        print(f"FAILED {year}-{month} ({ext}): {e}")

df = pd.DataFrame(records)
df = df.sort_values(["month", "year"]).reset_index(drop=True)
df.to_csv("cutoff_analysis_summary.csv", index=False)

print(f"\nSuccessfully processed: {len(records)} / {len(target_files)}")
print(f"Failed: {len(failed)}")

if failed:
    print("\n=== FAILED FILES ===")
    for year, month, ext, err in failed:
        print(f"  {year}-{month} ({ext}): {err}")

print("\n=== SUMMARY TABLE (first 10 rows) ===")
print(df.head(10))

print("\n=== FORMAT CHECK: any systematic difference between .nc and .grib readings? ===")
print(df.groupby("format")[["sst_mean", "delta_T_mean"]].describe())

# ---------- Plot: SST mean and ΔT variability over time ----------
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

for month_label, color in [("01", "blue"), ("07", "red")]:
    sub = df[df["month"] == month_label]
    axes[0].plot(sub["year"], sub["sst_mean"], marker='o', markersize=3, label=f"Month {month_label}", color=color)
    axes[1].plot(sub["year"], sub["delta_T_mean"], marker='o', markersize=3, label=f"Month {month_label}", color=color)
    axes[2].plot(sub["year"], sub["delta_T_std"], marker='o', markersize=3, label=f"Month {month_label}", color=color)

for ax, title, ylabel in zip(
    axes,
    ["Mean SST Over Time — Mumbai Region", "Mean Air-Sea ΔT Over Time", "Variability (Std Dev) of Air-Sea ΔT Over Time"],
    ["Mean SST (K)", "Mean ΔT (K)", "Std Dev of ΔT (K)"]
):
    ax.axvline(1979, color='gray', linestyle='--', label='1979 (satellite era)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

axes[2].set_xlabel("Year")
plt.tight_layout()
plt.savefig("cutoff_analysis_plot.png")
plt.show()

print("\nSaved: cutoff_analysis_summary.csv, cutoff_analysis_plot.png")