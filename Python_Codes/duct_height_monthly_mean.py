import xarray as xr
import pandas as pd
import numpy as np
import glob
import re
import os
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk

FOLDER = "era5_cutoff_check"
pattern = re.compile(r"era5_(\d{4})_(\d{2})\.(nc|grib)$")

LAT_PT, LON_PT = 19.0, 71.0

# ---------- Physics functions (unchanged) ----------

def saturation_vapor_pressure(T_K):
    T_C = T_K - 273.15
    return 6.1094 * np.exp(17.625 * T_C / (T_C + 243.04))

def specific_humidity(e_hPa, P_hPa):
    return 0.622 * e_hPa / (P_hPa - 0.378 * e_hPa)

def psi_m(zeta):
    zeta = np.atleast_1d(zeta).astype(float)
    x = (1 - 16*zeta)**0.25
    unstable = 2*np.log((1+x)/2) + np.log((1+x**2)/2) - 2*np.arctan(x) + np.pi/2
    stable = -5*zeta
    result = np.where(zeta < 0, unstable, stable)
    return result if result.size > 1 else result.item()

def psi_h(zeta):
    zeta = np.atleast_1d(zeta).astype(float)
    x = (1 - 16*zeta)**0.25
    unstable = 2*np.log((1+x**2)/2)
    stable = -5*zeta
    result = np.where(zeta < 0, unstable, stable)
    return result if result.size > 1 else result.item()

def refractivity(T_K, q_kgkg, P_hPa):
    e_hPa = q_kgkg * P_hPa / (0.622 + 0.378*q_kgkg)
    N = 77.6*(P_hPa/T_K) + 3.73e5*(e_hPa/T_K**2)
    return N

def compute_duct_height(T2m_K, Td2m_K, SST_K, MSLP_hPa, U10):
    k = 0.4
    g = 9.81
    z_air = 2.0
    z_wind = 10.0

    e_air = saturation_vapor_pressure(Td2m_K)
    q_air = specific_humidity(e_air, MSLP_hPa)
    e_sea = saturation_vapor_pressure(SST_K)
    q_sea = 0.98 * specific_humidity(e_sea, MSLP_hPa)

    Ta_virtual = T2m_K * (1 + 0.61*q_air)
    delta_T = T2m_K - SST_K
    delta_q = q_air - q_sea

    z0 = 0.0002
    L = 1e6

    for _ in range(20):
        u_star = k * U10 / (np.log(z_wind/z0) - psi_m(z_wind/L))
        if u_star <= 0 or np.isnan(u_star):
            return np.nan
        T_star = k * delta_T / (np.log(z_air/z0) - psi_h(z_air/L))
        q_star = k * delta_q / (np.log(z_air/z0) - psi_h(z_air/L))

        L_new = (u_star**2 * Ta_virtual) / (k * g * (T_star + 0.61*Ta_virtual*q_star))
        z0_new = 0.011 * u_star**2 / g + 0.11 * 1.5e-5 / u_star

        if abs(L_new - L) < 0.01 and abs(z0_new - z0) < 1e-6:
            L, z0 = L_new, z0_new
            break
        L, z0 = L_new, z0_new

    heights = np.linspace(0.05, 40, 200)
    T_profile = SST_K + (T_star/k) * (np.log(heights/z0) - psi_h(heights/L))
    q_profile = q_sea + (q_star/k) * (np.log(heights/z0) - psi_h(heights/L))
    P_profile = MSLP_hPa * np.exp(-heights * g / (287.05 * T2m_K))

    N_profile = refractivity(T_profile, q_profile, P_profile)
    M_profile = N_profile + 0.157 * heights

    min_idx = np.argmin(M_profile)
    return heights[min_idx]

# ---------- Gather files ----------

files = glob.glob(os.path.join(FOLDER, "era5_*.nc")) + glob.glob(os.path.join(FOLDER, "era5_*.grib"))

target_files = []
for f in files:
    basename = os.path.basename(f)
    m = pattern.match(basename)
    if not m:
        continue
    year, month, ext = m.groups()
    if month in ("01", "07"):
        target_files.append((int(year), month, ext, f))

target_files.sort()
print(f"Computing MONTHLY-MEAN duct height for {len(target_files)} Jan/July files...")

records = []
for year, month, ext, f in target_files:
    try:
        ds = xr.open_dataset(f) if ext == "nc" else xr.open_dataset(f, engine="cfgrib")

        # Handle both possible time coordinate names
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"

        point = ds.sel(latitude=LAT_PT, longitude=LON_PT)

        # KEY CHANGE: mean over ALL hours in the month instead of isel one hour
        T2m_K   = float(point.t2m.mean(dim=time_dim).values)
        Td2m_K  = float(point.d2m.mean(dim=time_dim).values)
        SST_K   = float(point.sst.mean(dim=time_dim).values)
        MSLP_hPa = float(point.msl.mean(dim=time_dim).values) / 100.0
        u10     = float(point.u10.mean(dim=time_dim).values)
        v10     = float(point.v10.mean(dim=time_dim).values)
        U10     = np.sqrt(u10**2 + v10**2)

        duct_h = compute_duct_height(T2m_K, Td2m_K, SST_K, MSLP_hPa, U10)

        records.append({"year": year, "month": month, "duct_height_m": duct_h})
        ds.close()

    except Exception as e:
        print(f"FAILED {year}-{month}: {e}")
        records.append({"year": year, "month": month, "duct_height_m": np.nan})

df = pd.DataFrame(records)
df.to_csv("duct_height_monthly_mean_by_year.csv", index=False)

print(f"\nSaved: duct_height_monthly_mean_by_year.csv")
print(f"NaN count: {df['duct_height_m'].isna().sum()} out of {len(df)}")
print(df.groupby("month")["duct_height_m"].describe())

# ---------- Plot + trend test ----------

df_clean = df.dropna(subset=["duct_height_m"])

fig, ax = plt.subplots(figsize=(10, 6))

for month_code, month_name, color in [("01", "January", "tab:blue"), ("07", "July", "tab:red")]:
    sub = df_clean[df_clean["month"] == month_code].sort_values("year")
    years = sub["year"].values
    heights = sub["duct_height_m"].values

    if len(sub) < 3:
        print(f"Not enough data for {month_name}, skipping trend line.")
        continue

    ax.scatter(years, heights, label=month_name, color=color, alpha=0.6, s=20)

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, heights)
    line = slope * years + intercept
    ax.plot(years, line, color=color, linestyle="--",
             label=f"{month_name} trend (slope={slope:.4f} m/yr, p={p_value:.4f})")

    mk_result = mk.original_test(heights)
    print(f"\n{month_name} duct height trend (monthly mean):")
    print(f"  Linear regression: slope={slope:.5f} m/yr, total change={slope*(years.max()-years.min()):+.3f} m, "
          f"R²={r_value**2:.3f}, p={p_value:.4f} "
          f"[{'SIGNIFICANT' if p_value<0.05 else 'not significant'}]")
    print(f"  Mann-Kendall: trend={mk_result.trend}, p={mk_result.p:.4f}, "
          f"Sen's slope={mk_result.slope:.5f} m/yr "
          f"[{'SIGNIFICANT' if mk_result.p<0.05 else 'not significant'}]")

ax.set_xlabel("Year")
ax.set_ylabel("Evaporation Duct Height (m) — monthly mean")
ax.set_title("Evaporation Duct Height Trend (Monthly Mean), 1940–2025 (19°N 71°E)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("duct_height_monthly_mean_trend.png")
plt.show()

print("\nSaved plot: duct_height_monthly_mean_trend.png")