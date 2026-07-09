import cdsapi
import os

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"

OUTPUT_DIR = "era5_cutoff_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
]

AREA = [20, 70, 17, 74]
GRID = ["1.0", "1.0"]

years = [str(y) for y in range(2024, 2026)]
months = ["01", "07"]

jobs = [(y, m) for y in years for m in months]
total = len(jobs)

for idx, (year, month) in enumerate(jobs, start=1):
    target = os.path.join(OUTPUT_DIR, f"era5_{year}_{month}.grib")

    if os.path.exists(target):
        print(f"[{idx}/{total}] Skipping {year}-{month}, already exists")
        continue

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [year],
        "month": [month],
        "day": ["01","02","03","04","05","06","07","08","09","10","11","12",
                "13","14","15","16","17","18","19","20","21","22","23","24",
                "25","26","27","28","29","30","31"],
        "time": ["00:00","01:00","02:00","03:00","04:00","05:00","06:00","07:00",
                 "08:00","09:00","10:00","11:00","12:00","13:00","14:00","15:00",
                 "16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
        "grid": GRID,
    }

    print(f"[{idx}/{total}] Requesting {year}-{month}...")
    try:
        client.retrieve(dataset, request).download(target)
        print(f"[{idx}/{total}] Downloaded {year}-{month}")
    except Exception as e:
        print(f"[{idx}/{total}] FAILED {year}-{month}: {e}")