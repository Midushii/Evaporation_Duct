import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import glob

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

def get_M_profile(T2m_K, Td2m_K, SST_K, MSLP_hPa, U10, height_max=80, n_levels=400):
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
        L_new = (u_star**2 * Ta_virtual) / (k * g * (T_star + 0.61*Ta_virtual*q_star))
        z0_new = 0.011 * u_star**2 / g + 0.11 * 1.5e-5 / u_star
        if abs(L_new - L) < 0.01 and abs(z0_new - z0) < 1e-6:
            L, z0 = L_new, z0_new
            break
        L, z0 = L_new, z0_new

    heights = np.linspace(0.05, height_max, n_levels)
    T_profile = SST_K + (T_star/k) * (np.log(heights/z0) - psi_h(heights/L))
    q_profile = q_sea + (q_star/k) * (np.log(heights/z0) - psi_h(heights/L))
    P_profile = MSLP_hPa * np.exp(-heights * g / (287.05 * T2m_K))
    N_profile = refractivity(T_profile, q_profile, P_profile)
    M_profile = N_profile + 0.157 * heights
    return heights, M_profile, delta_T, L


# ---------- Pick 4 hotspot hours to inspect ----------
df = pd.read_csv("year_2025_all_ocean_points_duct_heights_80m.csv")
hotspot = df[(df["latitude"]==20.0) & (df["longitude"]==72.0) & (df["flag"]=="boundary_hit")]

sample_hours = hotspot.sample(n=min(4, len(hotspot)), random_state=42)
print("Selected hours for plotting:")
print(sample_hours[["valid_time", "T2m_K", "SST_K", "delta_T", "wind_speed"]])

fig, axes = plt.subplots(1, len(sample_hours), figsize=(5*len(sample_hours), 6), sharey=True)
if len(sample_hours) == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, sample_hours.iterrows()):
    t = row["valid_time"]
    month_str = pd.to_datetime(t).strftime("%m")
    fname = f"era5_raw/era5_2025_{month_str}.nc"
    data = xr.open_dataset(fname)
    point = data.sel(latitude=20.0, longitude=72.0, valid_time=t)

    T2m_K   = point.t2m.values.item()
    Td2m_K  = point.d2m.values.item()
    SST_K   = point.sst.values.item()
    MSLP_hPa= point.msl.values.item() / 100.0
    u10     = point.u10.values.item()
    v10     = point.v10.values.item()
    U10     = np.sqrt(u10**2 + v10**2)

    heights, M_profile, delta_T, L = get_M_profile(T2m_K, Td2m_K, SST_K, MSLP_hPa, U10)

    ax.plot(M_profile, heights)
    ax.set_title(f"{t}\nΔT={delta_T:.2f}K, U10={U10:.1f}m/s, L={L:.1f}m")
    ax.set_xlabel("M (M-units)")
    ax.grid(True)

axes[0].set_ylabel("Height (m)")
plt.tight_layout()
plt.savefig("hotspot_profile_check.png")
plt.show()
print("\nSaved: hotspot_profile_check.png")