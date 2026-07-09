import pandas as pd

df = pd.read_csv("year_2025_all_ocean_points_duct_heights_80m.csv")

print("=== Boundary-hit count by grid point (80m run) ===")
boundary_by_point = df[df["flag"] == "boundary_hit"].groupby(["latitude", "longitude"]).size()
print(boundary_by_point)

print(f"\nTotal boundary_hit hours: {df['flag'].eq('boundary_hit').sum()}")
print(f"Of which at (17.0N, 73.0E): {df[(df['flag']=='boundary_hit') & (df['latitude']==17.0) & (df['longitude']==73.0)].shape[0]}")