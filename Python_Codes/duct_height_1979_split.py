import pandas as pd
import numpy as np
from scipy import stats
import pymannkendall as mk
import matplotlib.pyplot as plt

df = pd.read_csv("duct_height_monthly_mean_by_year.csv", dtype={"month": str})
df["month"] = df["month"].str.zfill(2)
df = df.dropna(subset=["duct_height_m"])

SPLIT_YEAR = 1979  # ERA5 satellite-era boundary

def run_trend(years, heights, label):
    if len(years) < 3:
        print(f"  {label}: not enough points ({len(years)})")
        return None
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, heights)
    mk_result = mk.original_test(heights)
    total_change = slope * (years.max() - years.min())
    print(f"  {label} (n={len(years)}, {years.min()}-{years.max()}):")
    print(f"    Linear regression: slope={slope:.5f} m/yr, total change={total_change:+.3f} m, "
          f"R²={r_value**2:.3f}, p={p_value:.4f} "
          f"[{'SIGNIFICANT' if p_value < 0.05 else 'not significant'}]")
    print(f"    Mann-Kendall: trend={mk_result.trend}, p={mk_result.p:.4f}, "
          f"Sen's slope={mk_result.slope:.5f} m/yr "
          f"[{'SIGNIFICANT' if mk_result.p < 0.05 else 'not significant'}]")
    return {"slope": slope, "p_value": p_value, "r_squared": r_value**2,
            "mk_p": mk_result.p, "mk_slope": mk_result.slope,
            "years": years, "heights": heights, "intercept": intercept}

results = {}

for month_code, month_name in [("01", "January"), ("07", "July")]:
    sub = df[df["month"] == month_code].sort_values("year")
    years = sub["year"].values
    heights = sub["duct_height_m"].values

    print(f"\n{'='*65}")
    print(f"  {month_name} — full record vs. pre/post {SPLIT_YEAR} split")
    print(f"{'='*65}")

    print(f"\nFull record:")
    run_trend(years, heights, "1940-2025")

    pre_mask = years < SPLIT_YEAR
    post_mask = years >= SPLIT_YEAR

    print(f"\nPre-{SPLIT_YEAR} (pre-satellite era):")
    r_pre = run_trend(years[pre_mask], heights[pre_mask], f"1940-{SPLIT_YEAR-1}")

    print(f"\nPost-{SPLIT_YEAR} (satellite era):")
    r_post = run_trend(years[post_mask], heights[post_mask], f"{SPLIT_YEAR}-2025")

    results[month_name] = {"years": years, "heights": heights,
                            "pre": r_pre, "post": r_post}

# ---------- Plot: full record with pre/post trend lines overlaid ----------

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for ax, (month_name, color) in zip(axes, [("January", "tab:blue"), ("July", "tab:red")]):
    years = results[month_name]["years"]
    heights = results[month_name]["heights"]

    ax.scatter(years, heights, color=color, alpha=0.5, s=25, label="data")
    ax.axvline(SPLIT_YEAR, color="black", linestyle=":", alpha=0.7, label=f"{SPLIT_YEAR} split")

    r_pre = results[month_name]["pre"]
    r_post = results[month_name]["post"]

    if r_pre is not None:
        yrs_pre = np.array([years.min(), SPLIT_YEAR - 1])
        line_pre = r_pre["slope"] * yrs_pre + r_pre["intercept"]
        ax.plot(yrs_pre, line_pre, color="black", linestyle="--",
                label=f"pre-{SPLIT_YEAR}: {r_pre['slope']:.4f} m/yr (p={r_pre['p_value']:.3f})")

    if r_post is not None:
        yrs_post = np.array([SPLIT_YEAR, years.max()])
        line_post = r_post["slope"] * yrs_post + r_post["intercept"]
        ax.plot(yrs_post, line_post, color="darkgreen", linestyle="--",
                label=f"post-{SPLIT_YEAR}: {r_post['slope']:.4f} m/yr (p={r_post['p_value']:.3f})")

    ax.set_title(f"{month_name}: pre/post {SPLIT_YEAR} trend")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Evaporation Duct Height (m) — monthly mean")
plt.tight_layout()
plt.savefig("duct_height_1979_split.png")
plt.show()

print(f"\nSaved plot: duct_height_1979_split.png")