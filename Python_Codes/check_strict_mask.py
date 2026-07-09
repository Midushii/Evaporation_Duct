import xarray as xr

static = xr.open_dataset("era5_static.nc")
lsm = static.lsm.isel(valid_time=0)

for threshold in [0.05, 0.01, 0.001, 0.0001, 0.00001]:
    count = 0
    points = []
    for lat_val in lsm.latitude.values:
        for lon_val in lsm.longitude.values:
            lsm_val = lsm.sel(latitude=lat_val, longitude=lon_val).values.item()
            if lsm_val < threshold:
                count += 1
                points.append((lat_val, lon_val, round(lsm_val, 8)))
    print(f"\nThreshold {threshold}: {count} points")
    for p in points:
        print(f"  {p}")