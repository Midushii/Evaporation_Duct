import glob
import re
import os

files = glob.glob("era5_cutoff_check/era5_*.nc") + glob.glob("era5_cutoff_check/era5_*.grib")

pattern = re.compile(r"era5_(\d{4})_(\d{2})\.(nc|grib)$")

valid_0107 = []
other_months = []
unparsed = []

for f in files:
    basename = os.path.basename(f)
    m = pattern.match(basename)
    if not m:
        unparsed.append(f)
        continue
    year, month, ext = m.groups()
    if month in ("01", "07"):
        valid_0107.append((int(year), month, ext, f))
    else:
        other_months.append(f)

valid_0107.sort()

print(f"Total files in folder: {len(files)}")
print(f"Valid Jan/Jul files: {len(valid_0107)}")
print(f"Other-month files (will be IGNORED): {len(other_months)}")
print(f"Unparsed filenames: {len(unparsed)}")

# Check format breakdown
nc_count = sum(1 for y, m, ext, f in valid_0107 if ext == "nc")
grib_count = sum(1 for y, m, ext, f in valid_0107 if ext == "grib")
print(f"\nOf the valid Jan/Jul files: {nc_count} are .nc, {grib_count} are .grib")

# Check for missing years (should have both 01 and 07 for every year 1940-2025)
years_present = {}
for y, m, ext, f in valid_0107:
    years_present.setdefault(y, []).append(m)

missing = []
for y in range(1940, 2026):
    months_found = years_present.get(y, [])
    if "01" not in months_found or "07" not in months_found:
        missing.append((y, months_found))

print(f"\nYears with missing Jan or Jul: {len(missing)}")
for y, months_found in missing:
    print(f"  {y}: only has {months_found}")

if other_months:
    print(f"\nSample of other-month files being ignored (first 5):")
    for f in other_months[:5]:
        print(f"  {f}")