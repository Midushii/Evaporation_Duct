import pandas as pd

old = pd.read_csv("year_2025_all_ocean_points_duct_heights_v2.csv")
new = pd.read_csv("year_2025_all_ocean_points_duct_heights_80m.csv")

old_unresolved = old[old["duct_height_m"].isna()]
print(f"Old (40m) unresolved count: {len(old_unresolved)}")

# Merge on time+location to see what happened to those specific hours in the 80m run
merged = old_unresolved.merge(
    new[["valid_time", "latitude", "longitude", "duct_height_m", "flag"]],
    on=["valid_time", "latitude", "longitude"],
    suffixes=("_40m", "_80m")
)

print(f"\nOf the {len(old_unresolved)} previously unresolved hours:")
now_resolved = merged[merged["duct_height_m_80m"].notna()]
still_unresolved = merged[merged["duct_height_m_80m"].isna()]
print(f"  Now resolved (found a value 40-80m): {len(now_resolved)}")
print(f"  Still unresolved even at 80m: {len(still_unresolved)}")

print("\n=== NEWLY RESOLVED VALUES (were NaN at 40m, now have a number) ===")
print(now_resolved["duct_height_m_80m"].describe())

print("\n=== STILL UNRESOLVED AT 80m (save these timestamps for profile plotting) ===")
print(still_unresolved[["valid_time", "latitude", "longitude", "T2m_K", "SST_K", "wind_speed"]])
still_unresolved.to_csv("still_unresolved_at_80m.csv", index=False)