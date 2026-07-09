import cdsapi
import os

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"

os.makedirs("era5_cutoff_check", exist_ok=True)

years = [str(y) for y in range(1940, 2026)]
months = ["01", "07"]  # January (winter) and July (monsoon peak)

variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature"
]

days = [f"{d:02d}" for d in range(1, 32)]
hours = [f"{h:02d}:00" for h in range(24)]

total_requests = len(years) * len(months)
done = 0

for year in years:
    for month in months:
        target = f"era5_cutoff_check/era5_{year}_{month}.nc"
        done += 1

        if os.path.exists(target):
            print(f"[{done}/{total_requests}] Skipping {year}-{month}, already exists")
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": [year],
            "month": [month],
            "day": days,
            "time": hours,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [20, 70, 17, 74],
            "grid": ["1.0", "1.0"]
        }

        try:
            print(f"[{done}/{total_requests}] Requesting {year}-{month}...")
            client.retrieve(dataset, request).download(target)
            print(f"[{done}/{total_requests}] Downloaded {year}-{month}")
        except Exception as e:
            print(f"[{done}/{total_requests}] FAILED {year}-{month}: {e}")

print("\nPhase A download complete.")