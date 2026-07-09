import xarray as xr

# Load static file (land-sea mask + geopotential)
static = xr.open_dataset("era5_static.nc")
print("=== STATIC FILE ===")
print(static)
print()

# Load one monthly data file
data = xr.open_dataset("era5_raw/era5_2025_01.nc")
print("=== JANUARY 2025 DATA FILE ===")
print(data)
print()

# Check that lat/lon grids match between the two files
print("=== GRID MATCH CHECK ===")
print("Static latitudes: ", static.latitude.values)
print("Data latitudes:   ", data.latitude.values)
print("Static longitudes:", static.longitude.values)
print("Data longitudes:  ", data.longitude.values)

# Show the land-sea mask values themselves
print()
print("=== LAND-SEA MASK VALUES (0=ocean, 1=land) ===")
print(static.lsm.values)