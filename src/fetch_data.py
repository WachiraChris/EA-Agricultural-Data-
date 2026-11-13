"""Download the external open data used to augment the 8-record seed sample.

Sources
  FAOSTAT  Production: Crops and livestock products (bulk file, Africa)
  NASA POWER  monthly precipitation and temperature (MERRA-2), 1981-2024
  ISRIC SoilGrids v2  topsoil properties (0-5 cm)

Outputs go to data/raw/. Re-running skips files that already exist.
"""
import io, json, sys, time, zipfile
from pathlib import Path
import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FAO_URL = "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_Africa.zip"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

COUNTRIES = ["Kenya", "Uganda", "United Republic of Tanzania", "Ethiopia", "Ethiopia PDR",
             "Rwanda", "Burundi", "Somalia", "South Sudan", "Sudan", "Sudan (former)", "Eritrea"]

# Representative farming locations per country (lat, lon). Country-level climate is
# the mean over these points; this is a simplification documented in the report.
POINTS = {
    "Kenya": [("Kitale", 1.02, 35.00), ("Nakuru", -0.30, 36.07), ("Machakos", -1.52, 37.26),
              ("Kisumu", -0.09, 34.77), ("Meru", 0.05, 37.65), ("Kilifi", -3.63, 39.85)],
    "Uganda": [("Kampala", 0.35, 32.58), ("Gulu", 2.77, 32.30), ("Mbarara", -0.61, 30.65),
               ("Mbale", 1.08, 34.18), ("Lira", 2.25, 32.90)],
    "United Republic of Tanzania": [("Arusha", -3.37, 36.68), ("Dodoma", -6.17, 35.74),
               ("Mbeya", -8.90, 33.45), ("Morogoro", -6.82, 37.66), ("Mwanza", -2.52, 32.90),
               ("Iringa", -7.77, 35.69)],
    "Ethiopia": [("Addis Ababa", 9.00, 38.75), ("Bahir Dar", 11.60, 37.38), ("Hawassa", 7.05, 38.47),
                 ("Jimma", 7.67, 36.83), ("Mekelle", 13.50, 39.47), ("Adama", 8.54, 39.27)],
    "Rwanda": [("Kigali", -1.95, 30.06), ("Musanze", -1.50, 29.63), ("Huye", -2.60, 29.74),
               ("Nyagatare", -1.30, 30.33)],
    "Burundi": [("Gitega", -3.43, 29.93), ("Ngozi", -2.91, 29.83), ("Bujumbura", -3.38, 29.36),
                ("Rutana", -3.93, 30.00)],
    "Somalia": [("Baidoa", 3.11, 43.65), ("Jowhar", 2.78, 45.50), ("Afgooye", 2.14, 45.12),
                ("Jilib", 0.49, 42.77)],
    "South Sudan": [("Juba", 4.85, 31.60), ("Yambio", 4.57, 28.40), ("Wau", 7.70, 28.00),
                    ("Renk", 11.75, 32.80)],
    "Sudan": [("Gedaref", 14.03, 35.38), ("Wad Medani", 14.40, 33.52), ("El Obeid", 13.18, 30.22),
              ("Kassala", 15.45, 36.40), ("Damazin", 11.79, 34.36)],
    "Eritrea": [("Asmara", 15.33, 38.93), ("Keren", 15.78, 38.45), ("Barentu", 15.10, 37.60),
                ("Mendefera", 14.89, 38.82)],
}


def get(url, params=None, tries=4, timeout=120):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            print("  http", r.status_code, file=sys.stderr)
        except requests.RequestException as e:
            print("  retry:", type(e).__name__, file=sys.stderr)
        time.sleep(3 * (i + 1))
    return None


def fetch_fao():
    out = RAW / "faostat_ea_crops.csv"
    if out.exists():
        return
    r = get(FAO_URL, timeout=600)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open("Production_Crops_Livestock_E_Africa.csv"), encoding="latin-1", low_memory=False)
    df = df[df["Area"].isin(COUNTRIES) & df["Element"].isin(["Yield", "Area harvested", "Production"])
            & df["Unit"].isin(["kg/ha", "ha", "t"])]
    years = [c[1:] for c in df.columns if c.startswith("Y") and c[1:].isdigit()]
    rows = []
    for y in years:
        part = df[["Area", "Item", "Element", "Unit", "Y" + y, f"Y{y}F"]].copy()
        part.columns = ["area", "item", "element", "unit", "value", "flag"]
        part["year"] = int(y)
        rows.append(part.dropna(subset=["value"]))
    pd.concat(rows).to_csv(out, index=False)
    print("FAOSTAT rows:", sum(len(p) for p in rows))


def fetch_power():
    out = RAW / "nasa_power_monthly.csv"
    if out.exists():
        return
    rows = []
    for country, pts in POINTS.items():
        for name, lat, lon in pts:
            r = get(POWER_URL, dict(parameters="PRECTOTCORR,T2M,T2M_MAX,T2M_MIN", community="AG",
                                    longitude=lon, latitude=lat, start=1981, end=2024, format="JSON"))
            if r is None:
                print("POWER failed:", name); continue
            par = r.json()["properties"]["parameter"]
            for key, val in par["PRECTOTCORR"].items():
                m = int(key[4:])
                if m > 12:
                    continue
                rows.append(dict(country=country, point=name, lat=lat, lon=lon, year=int(key[:4]), month=m,
                                 precip_mm_day=val, t2m=par["T2M"][key], t2m_max=par["T2M_MAX"][key],
                                 t2m_min=par["T2M_MIN"][key]))
            print("POWER ok:", country, name)
    pd.DataFrame(rows).to_csv(out, index=False)


def fetch_soil():
    out = RAW / "soilgrids_topsoil.csv"
    if out.exists():
        return
    rows = []
    for country, pts in POINTS.items():
        for name, lat, lon in pts:
            r = get(SOIL_URL, [("lon", lon), ("lat", lat), ("depth", "0-5cm"), ("value", "mean")] +
                    [("property", p) for p in ["phh2o", "clay", "sand", "soc", "nitrogen", "cec"]],
                    tries=3, timeout=90)
            if r is None:
                print("SoilGrids failed:", name); continue
            rec = dict(country=country, point=name, lat=lat, lon=lon)
            for layer in r.json()["properties"]["layers"]:
                v = layer["depths"][0]["values"]["mean"]
                d = layer["unit_measure"]["d_factor"]
                rec[layer["name"]] = None if v is None else v / d
            rows.append(rec)
            print("SoilGrids ok:", country, name)
            time.sleep(13)  # service allows 5 calls per minute
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    fetch_fao(); fetch_power(); fetch_soil()
