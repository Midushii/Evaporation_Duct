import xarray as xr

ds = xr.open_dataset("era5_raw/era5_2025_01.nc")
print(ds)