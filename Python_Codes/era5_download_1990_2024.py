import cdsapi
import os

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"

output_folder = "era5_cutoff_check"
os.makedirs(output_folder, exist_ok=True)

years = [str(y) for y in range(1990, 2025)]  # 1990-2024

# 01 and 07 excluded — already downloaded as .nc, not needed again
months = ["02", "03", "04", "05", "06", "08", "09", "10", "11", "12"]

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
        target = os.path.join(output_folder, f"era5_{year}_{month}.grib")
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
            "data_format": "grib",
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

print("\n1990-2024 GRIB download complete (months 01 and 07 excluded).")