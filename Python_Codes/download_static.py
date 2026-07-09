import cdsapi

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": ["reanalysis"],
    "variable": ["land_sea_mask", "geopotential"],
    "year": ["2020"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [20, 70, 17, 74],
    "grid": ["1.0", "1.0"]
}

client.retrieve(dataset, request).download("era5_static.nc")