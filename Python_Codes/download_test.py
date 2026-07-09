import cdsapi
import os

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"

os.makedirs("era5_raw", exist_ok=True)

year = "2025"
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]

for month in months:
    target = f"era5_raw/era5_{year}_{month}.nc"
    if os.path.exists(target):
        print(f"Skipping {year}-{month}, already exists")
        continue

    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "mean_sea_level_pressure",
            "sea_surface_temperature"
        ],
        "year": [year],
        "month": [month],
        "day": ["01","02","03","04","05","06","07","08","09","10","11","12",
                "13","14","15","16","17","18","19","20","21","22","23","24",
                "25","26","27","28","29","30","31"],
        "time": ["00:00","01:00","02:00","03:00","04:00","05:00","06:00","07:00",
                 "08:00","09:00","10:00","11:00","12:00","13:00","14:00","15:00",
                 "16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [20, 70, 17, 74],
        "grid": ["1.0", "1.0"]
    }

    try:
        client.retrieve(dataset, request).download(target)
        print(f"Downloaded {year}-{month}")
    except Exception as e:
        print(f"FAILED {year}-{month}: {e}")