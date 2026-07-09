import pandas as pd

df = pd.read_csv("year_2025_all_ocean_points_duct_heights_80m.csv")

hotspot = df[(df["latitude"]==20.0) & (df["longitude"]==72.0) & (df["flag"]=="boundary_hit")]
print(f"Total boundary hits at (20N, 72E): {len(hotspot)}")
print("\nMonth distribution:")
print(pd.to_datetime(hotspot["valid_time"]).dt.month.value_counts().sort_index())
print("\nConditions summary:")
print(hotspot[["T2m_K", "SST_K", "delta_T", "wind_speed"]].describe())