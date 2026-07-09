import pandas as pd
import numpy as np
from scipy import stats
import pymannkendall as mk

df = pd.read_csv("jan_jul_master_1940_2025.csv", dtype={"month": str})
df["month"] = df["month"].str.zfill(2)

variables = {
    "sst_mean": "SST",
    "t2m_mean": "2m Temperature",
    "d2m_mean": "Dewpoint",
    "deltaT_mean": "Air-Sea ΔT",
    "wind_speed_mean": "Wind Speed",
}

results = []

for month_code, month_name in [("01", "January"), ("07", "July")]:
    sub = df[df["month"] == month_code].sort_values("year")
    years = sub["year"].values

    print(f"\n{'='*60}")
    print(f"  {month_name}  (n={len(sub)} years)")
    print(f"{'='*60}")

    if len(sub) < 3:
        print("  Not enough data points, skipping.")
        continue

    for col, label in variables.items():
        values = sub[col].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
        total_change = slope * (years.max() - years.min())

        mk_result = mk.original_test(values)

        sig_lr = "SIGNIFICANT" if p_value < 0.05 else "not significant"
        sig_mk = "SIGNIFICANT" if mk_result.p < 0.05 else "not significant"

        print(f"\n{label}:")
        print(f"  Linear regression: slope = {slope:.5f}/year, "
              f"total change over {years.max()-years.min()} yrs = {total_change:+.3f}, "
              f"R² = {r_value**2:.3f}, p = {p_value:.4f} [{sig_lr}]")
        print(f"  Mann-Kendall:      trend = {mk_result.trend}, "
              f"p = {mk_result.p:.4f} [{sig_mk}], "
              f"Sen's slope = {mk_result.slope:.5f}/year")

        results.append({
            "month": month_name,
            "variable": label,
            "lr_slope_per_year": slope,
            "lr_total_change": total_change,
            "lr_r_squared": r_value**2,
            "lr_p_value": p_value,
            "lr_significant": p_value < 0.05,
            "mk_trend": mk_result.trend,
            "mk_p_value": mk_result.p,
            "mk_significant": mk_result.p < 0.05,
            "mk_sen_slope_per_year": mk_result.slope,
        })

results_df = pd.DataFrame(results)
results_df.to_csv("trend_analysis_results.csv", index=False)
print(f"\n\nSaved: trend_analysis_results.csv")