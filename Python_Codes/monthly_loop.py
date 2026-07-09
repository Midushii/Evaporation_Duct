import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Load data ----------
data = xr.open_dataset("era5_raw/era5_2025_01.nc")
lat_pt, lon_pt = 19.0, 71.0

k = 0.4
g = 9.81
z_air = 2.0
z_wind = 10.0

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
    """Runs the full Paulus-Jeske + ITU-R P.453 chain for one hour.
       Returns duct_height (float) or np.nan if it fails/doesn't converge."""
    try:
        e_air = saturation_vapor_pressure(Td2m_K)
        e_sat_air = saturation_vapor_pressure(T2m_K)
        q_air = specific_humidity(e_air, MSLP_hPa)
        e_sea = saturation_vapor_pressure(SST_K)
        q_sea = 0.98 * specific_humidity(e_sea, MSLP_hPa)

        Ta_virtual = T2m_K * (1 + 0.61*q_air)
        delta_T = T2m_K - SST_K
        delta_q = q_air - q_sea

        z0 = 0.0002
        L = 1e6

        # Guard against near-zero wind (division blow-up)
        if U10 < 0.5:
            U10 = 0.5

        for _ in range(20):
            u_star = k * U10 / (np.log(z_wind/z0) - psi_m(z_wind/L))
            T_star = k * delta_T / (np.log(z_air/z0) - psi_h(z_air/L))
            q_star = k * delta_q / (np.log(z_air/z0) - psi_h(z_air/L))

            if u_star <= 0 or not np.isfinite(u_star):
                return np.nan

            L_new = (u_star**2 * Ta_virtual) / (k * g * (T_star + 0.61*Ta_virtual*q_star))
            z0_new = 0.011 * u_star**2 / g + 0.11 * 1.5e-5 / u_star

            if not np.isfinite(L_new) or not np.isfinite(z0_new):
                return np.nan

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

    except Exception:
        return np.nan


# ---------- Loop over all hours ----------
results = []
n_hours = data.valid_time.size
print(f"Processing {n_hours} hours...")

for i in range(n_hours):
    t = data.valid_time.values[i]
    point = data.isel(valid_time=i).sel(latitude=lat_pt, longitude=lon_pt)

    T2m_K   = point.t2m.values.item()
    Td2m_K  = point.d2m.values.item()
    SST_K   = point.sst.values.item()
    MSLP_hPa= point.msl.values.item() / 100.0
    u10     = point.u10.values.item()
    v10     = point.v10.values.item()
    U10     = np.sqrt(u10**2 + v10**2)

    duct_h = compute_duct_height(T2m_K, Td2m_K, SST_K, MSLP_hPa, U10)

    results.append({
        "valid_time": t,
        "T2m_K": T2m_K,
        "SST_K": SST_K,
        "wind_speed": U10,
        "duct_height_m": duct_h
    })

df = pd.DataFrame(results)
df.to_csv("january_2025_duct_heights.csv", index=False)

# ---------- Summary stats ----------
n_failed = df["duct_height_m"].isna().sum()
n_ok = df["duct_height_m"].notna().sum()

print("\n=== SUMMARY: January 2025, point (19N, 71E) ===")
print(f"Total hours: {n_hours}")
print(f"Successful:  {n_ok}")
print(f"Failed/NaN:  {n_failed}")
print(f"Min duct height:  {df['duct_height_m'].min():.2f} m")
print(f"Max duct height:  {df['duct_height_m'].max():.2f} m")
print(f"Mean duct height: {df['duct_height_m'].mean():.2f} m")
print(f"Median duct height: {df['duct_height_m'].median():.2f} m")

# ---------- Plot time series ----------
plt.figure(figsize=(14,5))
plt.plot(df["valid_time"], df["duct_height_m"], marker='.', linewidth=0.8)
plt.xlabel("Time (January 2025)")
plt.ylabel("Evaporation Duct Height (m)")
plt.title("Evaporation Duct Height — January 2025, Point (19N, 71E)")
plt.grid(True)
plt.tight_layout()
plt.savefig("january_2025_duct_timeseries.png")
plt.show()

print("\nSaved: january_2025_duct_heights.csv, january_2025_duct_timeseries.png")