import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import glob

k = 0.4
g = 9.81
z_air = 2.0
z_wind = 10.0
LSM_THRESHOLD = 0.05

def saturation_vapor_pressure(T_K):
    T_C = T_K - 273.15
    return 6.1094 * np.exp(17.625 * T_C / (T_C + 243.04))

def specific_humidity(e_hPa, P_hPa):
    return 0.622 * e_hPa / (P_hPa - 0.378 * e_hPa)

def psi_m(zeta):
    zeta = np.atleast_1d(zeta).astype(float)
    x = np.full_like(zeta, np.nan)
    unstable_mask = zeta < 0
    x[unstable_mask] = (1 - 16*zeta[unstable_mask])**0.25
    unstable = np.zeros_like(zeta)
    unstable[unstable_mask] = (
        2*np.log((1+x[unstable_mask])/2)
        + np.log((1+x[unstable_mask]**2)/2)
        - 2*np.arctan(x[unstable_mask]) + np.pi/2
    )
    stable = -5*zeta
    result = np.where(zeta < 0, unstable, stable)
    return result if result.size > 1 else result.item()

def psi_h(zeta):
    zeta = np.atleast_1d(zeta).astype(float)
    x = np.full_like(zeta, np.nan)
    unstable_mask = zeta < 0
    x[unstable_mask] = (1 - 16*zeta[unstable_mask])**0.25
    unstable = np.zeros_like(zeta)
    unstable[unstable_mask] = 2*np.log((1+x[unstable_mask]**2)/2)
    stable = -5*zeta
    result = np.where(zeta < 0, unstable, stable)
    return result if result.size > 1 else result.item()

def refractivity(T_K, q_kgkg, P_hPa):
    e_hPa = q_kgkg * P_hPa / (0.622 + 0.378*q_kgkg)
    N = 77.6*(P_hPa/T_K) + 3.73e5*(e_hPa/T_K**2)
    return N

# Ceiling stays at 40m — standard, literature-consistent range for evaporation ducts
HEIGHT_MAX = 40
N_LEVELS = 200
heights_global = np.linspace(0.05, HEIGHT_MAX, N_LEVELS)

# Tightened check: flag anything whose minimum falls in the LAST 10% of the
# height range, not just the last 1-2 points. This catches "near-boundary"
# cases like 38.8m that used to slip through as if they were reliable.
BOUNDARY_ZONE_FRACTION = 0.10
boundary_zone_start_idx = int(N_LEVELS * (1 - BOUNDARY_ZONE_FRACTION))

def compute_duct_height(T2m_K, Td2m_K, SST_K, MSLP_hPa, U10):
    try:
        e_air = saturation_vapor_pressure(Td2m_K)
        q_air = specific_humidity(e_air, MSLP_hPa)
        e_sea = saturation_vapor_pressure(SST_K)
        q_sea = 0.98 * specific_humidity(e_sea, MSLP_hPa)

        Ta_virtual = T2m_K * (1 + 0.61*q_air)
        delta_T = T2m_K - SST_K
        delta_q = q_air - q_sea

        z0 = 0.0002
        L = 1e6

        if U10 < 0.5:
            U10 = 0.5

        for _ in range(20):
            u_star = k * U10 / (np.log(z_wind/z0) - psi_m(z_wind/L))
            T_star = k * delta_T / (np.log(z_air/z0) - psi_h(z_air/L))
            q_star = k * delta_q / (np.log(z_air/z0) - psi_h(z_air/L))

            if u_star <= 0 or not np.isfinite(u_star):
                return np.nan, "invalid_ustar", delta_T, U10

            L_new = (u_star**2 * Ta_virtual) / (k * g * (T_star + 0.61*Ta_virtual*q_star))
            z0_new = 0.011 * u_star**2 / g + 0.11 * 1.5e-5 / u_star

            if not np.isfinite(L_new) or not np.isfinite(z0_new):
                return np.nan, "invalid_L_or_z0", delta_T, U10

            if abs(L_new - L) < 0.01 and abs(z0_new - z0) < 1e-6:
                L, z0 = L_new, z0_new
                break
            L, z0 = L_new, z0_new

        T_profile = SST_K + (T_star/k) * (np.log(heights_global/z0) - psi_h(heights_global/L))
        q_profile = q_sea + (q_star/k) * (np.log(heights_global/z0) - psi_h(heights_global/L))
        P_profile = MSLP_hPa * np.exp(-heights_global * g / (287.05 * T2m_K))

        N_profile = refractivity(T_profile, q_profile, P_profile)
        M_profile = N_profile + 0.157 * heights_global

        min_idx = np.argmin(M_profile)

        if min_idx >= boundary_zone_start_idx:
            # Falls in the last 10% of the range -- not reliably resolved,
            # regardless of the exact number. Report NaN, but keep the
            # raw value and delta_T/wind so we can inspect the pattern.
            return np.nan, "boundary_hit", delta_T, U10

        if min_idx == 0:
            return np.nan, "surface_hit", delta_T, U10

        return heights_global[min_idx], None, delta_T, U10

    except Exception:
        return np.nan, "exception", np.nan, U10


# ---------- Load static mask ----------
static = xr.open_dataset("era5_static.nc")
lsm = static.lsm.isel(valid_time=0)

ocean_points = []
for lat_val in lsm.latitude.values:
    for lon_val in lsm.longitude.values:
        lsm_val = lsm.sel(latitude=lat_val, longitude=lon_val).values.item()
        if lsm_val < LSM_THRESHOLD:
            ocean_points.append((lat_val, lon_val, lsm_val))

print(f"Found {len(ocean_points)} ocean grid points (lsm < {LSM_THRESHOLD})")

# ---------- Loop over all monthly files x all ocean points ----------
files = sorted(glob.glob("era5_raw/era5_2025_*.nc"))
print(f"Found {len(files)} monthly files")

results = []

for f in files:
    data = xr.open_dataset(f)
    n_hours = data.valid_time.size
    month_label = f.split("_")[-1].replace(".nc", "")
    print(f"Processing {month_label}: {n_hours} hours x {len(ocean_points)} points...")

    for lat_pt, lon_pt, lsm_val in ocean_points:
        point_data = data.sel(latitude=lat_pt, longitude=lon_pt)

        T2m_arr   = point_data.t2m.values
        Td2m_arr  = point_data.d2m.values
        SST_arr   = point_data.sst.values
        MSLP_arr  = point_data.msl.values / 100.0
        u10_arr   = point_data.u10.values
        v10_arr   = point_data.v10.values
        times     = point_data.valid_time.values

        for i in range(n_hours):
            U10 = np.sqrt(u10_arr[i]**2 + v10_arr[i]**2)
            duct_h, flag, delta_T, wind = compute_duct_height(
                T2m_arr[i], Td2m_arr[i], SST_arr[i], MSLP_arr[i], U10
            )
            results.append({
                "valid_time": times[i],
                "latitude": lat_pt,
                "longitude": lon_pt,
                "T2m_K": T2m_arr[i],
                "SST_K": SST_arr[i],
                "delta_T": delta_T,
                "wind_speed": wind,
                "duct_height_m": duct_h,
                "flag": flag
            })

df = pd.DataFrame(results)
df = df.sort_values(["valid_time", "latitude", "longitude"]).reset_index(drop=True)
df.to_csv("year_2025_all_ocean_points_duct_heights_v2.csv", index=False)

n_total = len(df)
n_failed = df["duct_height_m"].isna().sum()

print(f"\nTotal point-hours: {n_total}")
print(f"Unresolved (total): {n_failed} ({100*n_failed/n_total:.2f}%)")

print("\n=== FLAG BREAKDOWN (overall) ===")
print(df["flag"].value_counts(dropna=True))

# ---------- Point 2: is this concentrated at specific points? ----------
print("\n=== UNRESOLVED-HOUR PERCENTAGE, BY GRID POINT ===")
unresolved_by_point = df.groupby(["latitude", "longitude"]).apply(
    lambda g: pd.Series({
        "total_hours": len(g),
        "unresolved_hours": g["duct_height_m"].isna().sum(),
        "unresolved_pct": 100 * g["duct_height_m"].isna().sum() / len(g)
    })
)
print(unresolved_by_point)

# ---------- Is it concentrated in specific conditions (stable + low wind)? ----------
print("\n=== CONDITIONS DURING UNRESOLVED HOURS vs ALL HOURS ===")
unresolved_rows = df[df["duct_height_m"].isna()]
resolved_rows = df[df["duct_height_m"].notna()]

print("Unresolved hours — mean delta_T (air-sea, K):", unresolved_rows["delta_T"].mean())
print("Resolved hours   — mean delta_T (air-sea, K):", resolved_rows["delta_T"].mean())
print("Unresolved hours — mean wind speed (m/s):    ", unresolved_rows["wind_speed"].mean())
print("Resolved hours   — mean wind speed (m/s):    ", resolved_rows["wind_speed"].mean())

# ---------- Is it concentrated in a specific season? ----------
df["month"] = pd.to_datetime(df["valid_time"]).dt.month
print("\n=== UNRESOLVED-HOUR PERCENTAGE, BY MONTH ===")
unresolved_by_month = df.groupby("month").apply(
    lambda g: 100 * g["duct_height_m"].isna().sum() / len(g)
)
print(unresolved_by_month)

# ---------- Summary stats per grid point (Point 3: report clearly) ----------
print("\n=== DUCT HEIGHT SUMMARY BY GRID POINT (resolved hours only) ===")
point_stats = df.groupby(["latitude", "longitude"])["duct_height_m"].agg(
    ["count", "min", "max", "mean", "median"]
)
print(point_stats)

# ---------- Spatial map of unresolved percentage ----------
pivot_unresolved = unresolved_by_point["unresolved_pct"].unstack()
plt.figure(figsize=(6,5))
plt.imshow(pivot_unresolved.values, cmap="Reds", aspect="auto",
           extent=[pivot_unresolved.columns.min(), pivot_unresolved.columns.max(),
                   pivot_unresolved.index.min(), pivot_unresolved.index.max()],
           origin="lower")
plt.colorbar(label="Unresolved Hours (%)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("% of Hours With No Reliable Duct Height (2025)")
plt.tight_layout()
plt.savefig("year_2025_unresolved_pct_map.png")
plt.show()

# ---------- Mean duct height map (resolved hours only) ----------
pivot_mean = df.groupby(["latitude", "longitude"])["duct_height_m"].mean().unstack()
plt.figure(figsize=(6,5))
plt.imshow(pivot_mean.values, cmap="viridis", aspect="auto",
           extent=[pivot_mean.columns.min(), pivot_mean.columns.max(),
                   pivot_mean.index.min(), pivot_mean.index.max()],
           origin="lower")
plt.colorbar(label="Mean Duct Height (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Mean Evaporation Duct Height — 2025, Ocean Points (Resolved Hours)")
plt.tight_layout()
plt.savefig("year_2025_spatial_mean_duct_v2.png")
plt.show()

print("\nSaved: year_2025_all_ocean_points_duct_heights_v2.csv")
print("Saved: year_2025_unresolved_pct_map.png, year_2025_spatial_mean_duct_v2.png")