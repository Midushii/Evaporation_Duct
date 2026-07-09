import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\hp\OneDrive\Desktop\era5_project"

df = pd.read_csv(f"{BASE}\\Tables\\jan_jul_master_1940_2025.csv", dtype={"month": str})
df["month"] = df["month"].str.zfill(2)

variables = {
    "sst_mean": ("Sea Surface Temperature", "K"),
    "t2m_mean": ("2m Air Temperature", "K"),
    "d2m_mean": ("Dewpoint (Humidity proxy)", "K"),
    "deltaT_mean": ("Air-Sea ΔT", "K"),
    "wind_speed_mean": ("Wind Speed", "m/s"),
}

month_info = [("01", "January", "tab:blue"), ("07", "July", "tab:red")]

for col, (label, unit) in variables.items():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for month_code, month_name, color in month_info:
        sub = df[df["month"] == month_code]
        values = sub[col].dropna().values

        # ---- PDF (histogram, normalized so area sums to 1) ----
        axes[0].hist(values, bins=15, density=True, alpha=0.5,
                     color=color, label=month_name, edgecolor="black")

        # ---- CDF (sorted values vs cumulative fraction) ----
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        axes[1].plot(sorted_vals, cdf, color=color, label=month_name, linewidth=2)

    axes[0].set_title(f"{label} — PDF (Histogram)")
    axes[0].set_xlabel(f"{label} ({unit})")
    axes[0].set_ylabel("Probability Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"{label} — CDF")
    axes[1].set_xlabel(f"{label} ({unit})")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"{label}: Climatological Distribution, 1940–2025 (19°N 71°E)")
    plt.tight_layout()

    safe_name = col.replace("_mean", "")
    plt.savefig(f"{BASE}\\Plots\\pdf_cdf_{safe_name}.png")
    plt.show()

print("Done. Saved one PDF+CDF figure per parameter in Plots/:")
for col in variables:
    print(f"  pdf_cdf_{col.replace('_mean','')}.png")