import numpy as np
import xarray as xr

# ---------- STEP 0: Load and select data ----------

data = xr.open_dataset("era5_raw/era5_2025_01.nc")
static = xr.open_dataset("era5_static.nc")

# Pick a clean ocean point (lsm = 0.0), e.g. lat=19N, lon=71E
lat_pt, lon_pt = 19.0, 71.0
lsm_val = static.lsm.sel(latitude=lat_pt, longitude=lon_pt).values.item()
print(f"Selected point ({lat_pt}N, {lon_pt}E) — land-sea mask value: {lsm_val:.4f} (0=ocean)")

# Pick one sample hour to test on
sample_time = data.valid_time.values[12]  # 12:00 on Jan 1
point = data.sel(latitude=lat_pt, longitude=lon_pt, valid_time=sample_time)

T2m_K   = point.t2m.values.item()          # 2m air temp, Kelvin
Td2m_K  = point.d2m.values.item()          # 2m dewpoint, Kelvin
SST_K   = point.sst.values.item()          # sea surface temp, Kelvin
MSLP_hPa= point.msl.values.item() / 100.0  # convert Pa -> hPa
u10     = point.u10.values.item()
v10     = point.v10.values.item()
U10     = np.sqrt(u10**2 + v10**2)         # wind speed at 10m

print(f"\nInputs at {sample_time}:")
print(f"  2m Temp     = {T2m_K:.2f} K ({T2m_K-273.15:.2f} C)")
print(f"  2m Dewpoint = {Td2m_K:.2f} K ({Td2m_K-273.15:.2f} C)")
print(f"  SST         = {SST_K:.2f} K ({SST_K-273.15:.2f} C)")
print(f"  MSLP        = {MSLP_hPa:.2f} hPa")
print(f"  Wind speed  = {U10:.2f} m/s")


# ---------- STEP 1: Humidity calculations ----------

def saturation_vapor_pressure(T_K):
    """Magnus formula, T in Kelvin, returns hPa"""
    T_C = T_K - 273.15
    return 6.1094 * np.exp(17.625 * T_C / (T_C + 243.04))

def specific_humidity(e_hPa, P_hPa):
    """Specific humidity from vapor pressure and total pressure"""
    return 0.622 * e_hPa / (P_hPa - 0.378 * e_hPa)

e_air = saturation_vapor_pressure(Td2m_K)          # actual vapor pressure at 2m (from dewpoint)
e_sat_air = saturation_vapor_pressure(T2m_K)       # saturation vapor pressure at air temp
RH = 100 * e_air / e_sat_air                       # relative humidity, %

q_air = specific_humidity(e_air, MSLP_hPa)         # actual specific humidity at 2m
e_sea = saturation_vapor_pressure(SST_K)           # saturation vapor pressure at sea surface
q_sea = 0.98 * specific_humidity(e_sea, MSLP_hPa)  # 0.98 factor: salinity reduction of sat. humidity over seawater

print(f"\n  RH (2m)         = {RH:.1f} %")
print(f"  q_air (2m)      = {q_air*1000:.3f} g/kg")
print(f"  q_sea (surface) = {q_sea*1000:.3f} g/kg")


# ---------- STEP 2: Constants ----------

k = 0.4        # von Karman constant
g = 9.81       # gravity, m/s^2
z_air = 2.0    # height of temp/humidity measurement, m
z_wind = 10.0  # height of wind measurement, m
cp = 1005.0    # specific heat of air, J/(kg K)
Lv = 2.5e6     # latent heat of vaporization, J/kg


# ---------- STEP 3: Stability functions (Businger-Dyer) ----------

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


# ---------- STEP 4: Iterative solution for u*, T*, q*, L ----------

Ta_virtual = T2m_K * (1 + 0.61*q_air)   # virtual temperature, air
delta_T = T2m_K - SST_K                  # air - sea temperature difference
delta_q = q_air - q_sea

# Initial guesses (neutral stability)
z0 = 0.0002       # initial roughness length guess, m
L = 1e6           # initial Monin-Obukhov length (near-neutral)

for iteration in range(20):
    u_star = k * U10 / (np.log(z_wind/z0) - psi_m(z_wind/L))
    T_star = k * delta_T / (np.log(z_air/z0) - psi_h(z_air/L))
    q_star = k * delta_q / (np.log(z_air/z0) - psi_h(z_air/L))

    # Update Monin-Obukhov length
    L_new = (u_star**2 * Ta_virtual) / (k * g * (T_star + 0.61*Ta_virtual*q_star))

    # Update roughness length (Charnock relation + smooth-flow term)
    z0_new = 0.011 * u_star**2 / g + 0.11 * 1.5e-5 / u_star

    if abs(L_new - L) < 0.01 and abs(z0_new - z0) < 1e-6:
        L, z0 = L_new, z0_new
        break
    L, z0 = L_new, z0_new

print(f"\nAfter {iteration+1} iterations:")
print(f"  u* = {u_star:.4f} m/s")
print(f"  T* = {T_star:.4f} K")
print(f"  q* = {q_star:.6f} kg/kg")
print(f"  L  = {L:.2f} m  (Monin-Obukhov length)")
print(f"  z0 = {z0*1000:.4f} mm  (roughness length)")


# ---------- STEP 5: Build the vertical profile (0 to 40m) ----------

heights = np.linspace(0.05, 40, 200)  # avoid z=0 (log undefined)

T_profile = SST_K + (T_star/k) * (np.log(heights/z0) - psi_h(heights/L))
q_profile = q_sea + (q_star/k) * (np.log(heights/z0) - psi_h(heights/L))
P_profile = MSLP_hPa * np.exp(-heights * g / (287.05 * T2m_K))  # simple hydrostatic approx


# ---------- STEP 6: Convert to refractivity (ITU-R P.453) ----------

def refractivity(T_K, q_kgkg, P_hPa):
    """N via ITU-R P.453, using specific humidity converted to vapor pressure"""
    e_hPa = q_kgkg * P_hPa / (0.622 + 0.378*q_kgkg)
    N = 77.6*(P_hPa/T_K) + 3.73e5*(e_hPa/T_K**2)
    return N

N_profile = refractivity(T_profile, q_profile, P_profile)
M_profile = N_profile + 0.157 * heights  # modified refractivity


# ---------- STEP 7: Find duct height ----------

min_idx = np.argmin(M_profile)
duct_height = heights[min_idx]

print(f"\n=== RESULT ===")
print(f"Evaporation duct height: {duct_height:.2f} m")
print(f"M at duct height: {M_profile[min_idx]:.2f} M-units")
print(f"M at surface: {M_profile[0]:.2f} M-units")

import matplotlib.pyplot as plt

plt.figure(figsize=(6,8))
plt.plot(M_profile, heights)
plt.axhline(duct_height, color='red', linestyle='--', label=f'Duct height = {duct_height:.2f} m')
plt.xlabel('Modified Refractivity M (M-units)')
plt.ylabel('Height (m)')
plt.title('Evaporation Duct Profile')
plt.legend()
plt.grid(True)
plt.savefig('duct_profile.png')
plt.show()

import pandas as pd

profile_df = pd.DataFrame({
    'height_m': heights,
    'temperature_K': T_profile,
    'specific_humidity_kgkg': q_profile,
    'pressure_hPa': P_profile,
    'refractivity_N': N_profile,
    'modified_refractivity_M': M_profile
})
profile_df.to_csv('duct_profile_sample.csv', index=False)
print("\nProfile saved to duct_profile_sample.csv")