import xarray as xr

filepath = "era5_mumbai_2025.grib"

ds = xr.open_dataset(filepath, engine="cfgrib")

print("=== DATASET STRUCTURE ===")
print(ds)

print("\n=== TIME RANGE ===")
print("Start:", ds.time.values[0])
print("End:  ", ds.time.values[-1])
print("Total timesteps:", ds.time.size)

print("\n=== VARIABLES PRESENT ===")
for var in ds.data_vars:
    print(f"  {var}: {ds[var].attrs.get('long_name', 'N/A')} ({ds[var].attrs.get('units', 'N/A')})")

print("\n=== GRID ===")
print("Latitude:", ds.latitude.values)
print("Longitude:", ds.longitude.values)

print("\n=== SAMPLE VALUES (first timestep, first grid point) ===")
sample = ds.isel(time=0, latitude=0, longitude=0)
for var in ds.data_vars:
    print(f"  {var}: {sample[var].values}")

print("\n=== QUICK QUALITY CHECK ===")
for var in ds.data_vars:
    vals = ds[var].values
    print(f"  {var}: min={vals.min():.2f}, max={vals.max():.2f}, has_nan={bool(__import__('numpy').isnan(vals).any())}")