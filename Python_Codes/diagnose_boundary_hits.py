import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import glob

# Import everything from your existing model
import Python_Codes.ocean_points_loop as model

print("="*70)
print("BOUNDARY HIT DIAGNOSTIC")
print("="*70)

# -------------------------------
# Load the CSV produced previously
# -------------------------------

df = pd.read_csv(
    "year_2025_all_ocean_points_duct_heights_v2.csv",
    parse_dates=["valid_time"]
)

print(f"Loaded {len(df)} records")

# ------------------------------------------
# Keep only boundary-hit cases
# ------------------------------------------

boundary = df[
    (df["flag"] == "boundary_hit") &
    (df["latitude"] == 20.0) &
    (df["longitude"] == 72.0)
].copy()

print(f"Boundary hits at (20N,72E): {len(boundary)}")

if len(boundary) == 0:
    raise RuntimeError("No boundary hits found!")

# ------------------------------------------
# Select first 10 cases
# ------------------------------------------

cases = boundary.head(10).reset_index(drop=True)

print("\nSelected cases:\n")

print(
    cases[
        [
            "valid_time",
            "delta_T",
            "wind_speed"
        ]
    ]
)

print("\nDiagnostic will analyse these 10 cases.")