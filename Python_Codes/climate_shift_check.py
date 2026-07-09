import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("cutoff_analysis_summary.csv")

# Assign each year to a decade
df["decade"] = (df["year"] // 10) * 10

# Baseline: most recent complete decade (2015-2024, since 2025 data may be partial)
baseline = df[df["year"] >= 2015]
baseline_sst_mean = baseline["sst_mean"].mean()
baseline_deltaT_mean = baseline["delta_T_mean"].mean()

print(f"Baseline (2015-2025) — Mean SST: {baseline_sst_mean:.3f} K, Mean ΔT: {baseline_deltaT_mean:.3f} K")
print(f"Baseline sample size: {len(baseline)} year-month records\n")

print("=== DECADE-BY-DECADE COMPARISON AGAINST BASELINE ===")
results = []

for decade in sorted(df["decade"].unique()):
    decade_data = df[df["decade"] == decade]
    if len(decade_data) < 4:  # skip decades with too few records for a meaningful test
        continue

    sst_diff = decade_data["sst_mean"].mean() - baseline_sst_mean
    deltaT_diff = decade_data["delta_T_mean"].mean() - baseline_deltaT_mean

    # Statistical test: is this decade's SST significantly different from baseline?
    t_stat, p_value = stats.ttest_ind(decade_data["sst_mean"], baseline["sst_mean"], equal_var=False)

    results.append({
        "decade": decade,
        "n_records": len(decade_data),
        "mean_sst": decade_data["sst_mean"].mean(),
        "sst_diff_from_baseline": sst_diff,
        "mean_deltaT": decade_data["delta_T_mean"].mean(),
        "deltaT_diff_from_baseline": deltaT_diff,
        "p_value": p_value,
        "significantly_different": p_value < 0.05
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv("climate_shift_by_decade.csv", index=False)
print("\nSaved: climate_shift_by_decade.csv")

# Identify the earliest decade that is NOT significantly different from baseline
not_different = results_df[~results_df["significantly_different"]]
if len(not_different) > 0:
    earliest_ok_decade = not_different["decade"].min()
    print(f"\nEarliest decade statistically consistent with 2015-2025 baseline: {earliest_ok_decade}s")
else:
    print("\nAll decades show statistically significant difference from baseline.")

print("\n=== INTERPRETATION GUIDE ===")
print("A 'significant' difference (p < 0.05) means that decade's average SST")
print("is unlikely to be explained by random year-to-year variability alone —")
print("i.e., there's a real, detectable climate shift relative to today.")
print("This does NOT mean older data is unreliable — it means older data")
print("represents different average conditions than today's climate.")