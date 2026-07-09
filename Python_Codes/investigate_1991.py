import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Find the January 1991 file, whichever format it's in
candidates = glob.glob("era5_cutoff_check/era5_1991_01.nc") + glob.glob("era5_cutoff_check/era5_1991_01.grib")

if not candidates:
    print("ERROR: era5_1991_01 file not found in era5_cutoff_check/")
else:
    f = candidates[0]
    ext = os.path.splitext(f)[1]
    print(f"Found: {f}")

    if ext == ".nc":
        ds = xr.open_dataset(f)
    else:
        ds = xr.open_dataset(f, engine="cfgrib")

    sst = ds.sst.mean(dim=["latitude", "longitude"]).values
    t2m = ds.t2m.mean(dim=["latitude", "longitude"]).values
    delta_T = t2m - sst
    times = ds.valid_time.values

    print(f"\nJanuary 1991 — Mumbai box average, hourly ΔT (air - sea):")
    print(f"  Mean: {np.nanmean(delta_T):.3f} K")
    print(f"  Std:  {np.nanstd(delta_T):.3f} K")
    print(f"  Min:  {np.nanmin(delta_T):.3f} K")
    print(f"  Max:  {np.nanmax(delta_T):.3f} K")

    # Check for missing/suspicious values
    n_nan = np.isnan(delta_T).sum()
    print(f"  NaN count: {n_nan} / {len(delta_T)}")

    # Plot the full hourly time series for the month
    plt.figure(figsize=(14, 5))
    plt.plot(times, delta_T, linewidth=0.8)
    plt.xlabel("Time (January 1991)")
    plt.ylabel("Air-Sea ΔT (K)")
    plt.title("Hourly Air-Sea Temperature Difference — January 1991, Mumbai Box")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("investigate_1991_timeseries.png")
    plt.show()

    print("\nSaved: investigate_1991_timeseries.png")

    # Compare against neighboring years (1989, 1990, 1992, 1993) for context
    print("\n=== COMPARISON WITH NEIGHBORING YEARS ===")
    for yr in [1989, 1990, 1992, 1993]:
        neighbor_candidates = glob.glob(f"era5_cutoff_check/era5_{yr}_01.nc") + glob.glob(f"era5_cutoff_check/era5_{yr}_01.grib")
        if not neighbor_candidates:
            continue
        nf = neighbor_candidates[0]
        next_ = os.path.splitext(nf)[1]
        nds = xr.open_dataset(nf) if next_ == ".nc" else xr.open_dataset(nf, engine="cfgrib")
        nsst = nds.sst.mean(dim=["latitude", "longitude"]).values
        nt2m = nds.t2m.mean(dim=["latitude", "longitude"]).values
        ndelta = nt2m - nsst
        print(f"  {yr}: mean ΔT = {np.nanmean(ndelta):.3f} K, std = {np.nanstd(ndelta):.3f} K")
        nds.close()

    ds.close()