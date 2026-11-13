"""Clean the raw sources and build the modelling table.

Steps
  1. Parse the 8-record seed sample into structured crop requirements.
  2. Clean FAOSTAT yields: reshape, recompute yield from production/area,
     flag carried-forward repeats and implausible jumps.
  3. Aggregate NASA POWER monthly climate to country-year features.
  4. Aggregate SoilGrids topsoil properties to country level (if downloaded).
  5. Build history features (past years only) and the two targets.

Outputs: data/processed/seed_crops.csv, seed_other.csv, climate_country_year.csv,
         yield_clean.csv, model_table.csv, cleaning_log.json
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CROPS = {
    "Maize (corn)": "Maize", "Sorghum": "Sorghum", "Millet": "Millet", "Beans, dry": "Beans",
    "Groundnuts, excluding shelled": "Groundnuts", "Wheat": "Wheat", "Rice": "Rice",
    "Cassava, fresh": "Cassava", "Sweet potatoes": "Sweet potato", "Potatoes": "Potato",
    "Bananas": "Banana", "Sesame seed": "Sesame", "Cow peas, dry": "Cowpea",
    "Pigeon peas, dry": "Pigeon pea",
}
CROP_GROUP = {"Maize": "Cereal", "Sorghum": "Cereal", "Millet": "Cereal", "Wheat": "Cereal", "Rice": "Cereal",
              "Beans": "Pulse", "Cowpea": "Pulse", "Pigeon pea": "Pulse", "Groundnuts": "Oilseed",
              "Sesame": "Oilseed", "Cassava": "Root/tuber", "Sweet potato": "Root/tuber",
              "Potato": "Root/tuber", "Banana": "Fruit"}
CLIMATE_OF = {"Sudan (former)": "Sudan", "Ethiopia PDR": "Ethiopia"}
SHORT = {"United Republic of Tanzania": "Tanzania"}
BASELINE = (1981, 2010)          # climatology period for anomalies
SHORTFALL = 0.90                 # yield below 90% of the trailing 5-year mean
DAYS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
log = {}


def nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text)]


def parse_seed():
    recs = json.loads((RAW / "crops_water_soil.json").read_text())
    crops, other = [], []
    for r in recs:
        if r["category"] in ("crop", "tree"):
            w = nums(r["water_requirement"])[:2]
            y = nums(r["yield_range_kg_ha"])[:2]
            crops.append(dict(
                id=r["id"], category=r["category"],
                crop=r["common_name"].split(" (")[0], swahili=re.search(r"\((.*?)\)", r["common_name"]).group(1),
                scientific_name=r["scientific_name"], region=r["region"],
                water_min_mm=w[0], water_max_mm=w[1], yield_min_kg_ha=y[0], yield_max_kg_ha=y[1],
                drought_tolerant="drought" in r["water_requirement"].lower(),
                tolerates_poor_soil=bool(re.search(r"poor|any|wide range", r["soil_type"].lower())),
                planting_season=r["planting_season"], soil_type=r["soil_type"],
                storage=r["storage_months"], uses=r["uses"], source=r["source"], n_tags=len(r["tags"])))
        else:
            other.append({k: (", ".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    pd.DataFrame(crops).to_csv(OUT / "seed_crops.csv", index=False)
    pd.DataFrame(other).to_csv(OUT / "seed_other.csv", index=False)
    log["seed_records"] = len(recs)
    log["seed_categories"] = pd.Series([r["category"] for r in recs]).value_counts().to_dict()


def clean_fao():
    f = pd.read_csv(RAW / "faostat_ea_crops.csv")
    f = f[f["item"].isin(CROPS) & (f["year"] >= 1976)]
    log["fao_rows_selected"] = int(len(f))
    wide = f.pivot_table(index=["area", "item", "year"], columns="element", values="value").reset_index()
    flag = (f[f["element"] == "Yield"].set_index(["area", "item", "year"])["flag"]
            .rename("yield_flag").reset_index())
    d = wide.merge(flag, how="left", on=["area", "item", "year"])
    d.columns = [c.lower().replace(" ", "_") for c in d.columns]
    d = d.rename(columns={"yield": "yield_kg_ha", "area_harvested": "area_ha", "production": "production_t"})
    d["crop"] = d["item"].map(CROPS)
    d["country"] = d["area"].replace(SHORT)
    n0 = len(d)

    # 1. missing or non-positive yield, tiny harvested area (ratios become unstable)
    d = d[d["yield_kg_ha"].notna() & (d["yield_kg_ha"] > 0)]
    log["dropped_missing_yield"] = n0 - len(d)
    n1 = len(d)
    d = d[d["area_ha"].fillna(0) >= 500]
    log["dropped_area_under_500ha"] = n1 - len(d)

    # 2. consistency of reported yield with production / area
    calc = d["production_t"] * 1000 / d["area_ha"]
    d["yield_mismatch"] = (np.abs(calc / d["yield_kg_ha"] - 1) > 0.05) & calc.notna()
    log["yield_vs_production_area_mismatch_over_5pct"] = int(d["yield_mismatch"].sum())

    d = d.sort_values(["area", "crop", "year"]).reset_index(drop=True)
    g = d.groupby(["area", "crop"])
    # 3. carried-forward values: identical yield to the previous year
    d["is_repeat"] = g["yield_kg_ha"].diff().eq(0) & g["year"].diff().eq(1)
    # 4. implausible jumps: more than 3x away from the median of the 5 nearest years
    med = g["yield_kg_ha"].transform(lambda s: s.rolling(5, center=True, min_periods=3).median())
    ratio = d["yield_kg_ha"] / med
    d["is_outlier"] = (ratio > 3) | (ratio < 1 / 3)
    log["flagged_repeat_values"] = int(d["is_repeat"].sum())
    log["flagged_outliers"] = int(d["is_outlier"].sum())
    log["yield_flag_counts"] = d["yield_flag"].value_counts(dropna=False).to_dict()
    d["official"] = d["yield_flag"].eq("A")
    d.drop(columns=["item"]).to_csv(OUT / "yield_clean.csv", index=False)
    return d


def climate():
    p = pd.read_csv(RAW / "nasa_power_monthly.csv")
    p = p.replace(-999.0, np.nan)
    log["climate_missing_cells"] = int(p[["precip_mm_day", "t2m"]].isna().sum().sum())
    p["rain_mm"] = p["precip_mm_day"] * p["month"].map(lambda m: DAYS[m - 1])
    m = p.groupby(["country", "year", "month"])[["rain_mm", "t2m", "t2m_max", "t2m_min"]].mean().reset_index()

    m[m["year"].between(*BASELINE)].groupby(["country", "month"])[["rain_mm", "t2m", "t2m_max", "t2m_min"]] \
        .mean().reset_index().to_csv(OUT / "climatology_monthly.csv", index=False)

    def season(months, name):
        return m[m["month"].isin(months)].groupby(["country", "year"])["rain_mm"].sum().rename(name)

    a = m.groupby(["country", "year"]).agg(
        rain_annual=("rain_mm", "sum"), rain_max_month=("rain_mm", "max"),
        dry_months=("rain_mm", lambda s: int((s < 30).sum())),
        temp_mean=("t2m", "mean"), temp_max=("t2m_max", "max"), temp_min=("t2m_min", "min"),
        n_months=("month", "count"))
    a = a.join([season([3, 4, 5], "rain_mam"), season([6, 7, 8, 9], "rain_jjas"),
                season([10, 11, 12], "rain_ond")]).reset_index()
    a = a[a["n_months"] == 12].drop(columns="n_months")
    a["rain_concentration"] = a["rain_max_month"] / a["rain_annual"]
    base = a[a["year"].between(*BASELINE)].groupby("country").agg(
        rain_clim=("rain_annual", "mean"), rain_sd=("rain_annual", "std"), temp_clim=("temp_mean", "mean"),
        mam_clim=("rain_mam", "mean"), jjas_clim=("rain_jjas", "mean"), ond_clim=("rain_ond", "mean"))
    a = a.merge(base, on="country")
    a["rain_anom_pct"] = (a["rain_annual"] / a["rain_clim"] - 1) * 100
    a["rain_z"] = (a["rain_annual"] - a["rain_clim"]) / a["rain_sd"]
    a["temp_anom"] = a["temp_mean"] - a["temp_clim"]
    # anomaly of the wettest season (the main growing season differs by country)
    seas = a[["mam_clim", "jjas_clim", "ond_clim"]].values
    main = seas.argmax(axis=1)
    cur = a[["rain_mam", "rain_jjas", "rain_ond"]].values
    a["main_season"] = np.array(["MAM", "JJAS", "OND"])[main]
    idx = np.arange(len(a))
    a["main_season_anom_pct"] = (cur[idx, main] / seas[idx, main] - 1) * 100
    a = a.sort_values(["country", "year"])
    a["rain_anom_prev"] = a.groupby("country")["rain_anom_pct"].shift(1)
    a = a.drop(columns=["rain_sd", "mam_clim", "jjas_clim", "ond_clim", "rain_max_month"])
    a.to_csv(OUT / "climate_country_year.csv", index=False)
    return a


def soil():
    f = RAW / "soilgrids_topsoil.csv"
    if not f.exists():
        log["soil"] = "not available (SoilGrids did not respond)"
        return None
    s = pd.read_csv(f).groupby("country")[["phh2o", "clay", "sand", "soc", "nitrogen", "cec"]].mean()
    s.columns = ["soil_ph", "soil_clay_pct", "soil_sand_pct", "soil_soc_g_kg", "soil_n_g_kg", "soil_cec"]
    log["soil"] = f"{len(s)} countries"
    return s.reset_index()


def build(d, clim, s):
    d = d.copy()
    d["clim_country"] = d["area"].replace(CLIMATE_OF)
    # history features use valid past observations only
    valid = d["yield_kg_ha"].where(~d["is_outlier"])
    d["_y"] = valid
    g = d.groupby(["area", "crop"])
    d["yield_lag1"] = g["_y"].shift(1)
    d["yield_mean5"] = g["_y"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    d["yield_sd5"] = g["_y"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).std())
    d["yield_trend5"] = d["yield_lag1"] / d["yield_mean5"] - 1
    d["yield_cv5"] = d["yield_sd5"] / d["yield_mean5"]
    d["area_lag1"] = g["area_ha"].shift(1)
    d["log_area_lag1"] = np.log10(d["area_lag1"])
    d["prev_shortfall"] = (g["_y"].shift(1) < SHORTFALL * g["_y"].transform(
        lambda x: x.shift(2).rolling(5, min_periods=3).mean())).astype(int)
    d["crop_group"] = d["crop"].map(CROP_GROUP)

    t = d.merge(clim, left_on=["clim_country", "year"], right_on=["country", "year"],
                how="inner", suffixes=("", "_c")).drop(columns=["country_c"])
    if s is not None:
        t = t.merge(s, left_on="clim_country", right_on="country", how="left", suffixes=("", "_s")).drop(
            columns=["country_s"])
    n0 = len(t)
    t = t[~t["is_repeat"] & ~t["is_outlier"]]
    log["dropped_repeat_or_outlier_targets"] = n0 - len(t)
    n1 = len(t)
    t = t[t["yield_mean5"].notna() & t["yield_lag1"].notna()]
    log["dropped_insufficient_history"] = n1 - len(t)
    t["yield_ratio"] = t["yield_kg_ha"] / t["yield_mean5"]
    t["shortfall"] = (t["yield_ratio"] < SHORTFALL).astype(int)
    t = t.drop(columns=["_y", "yield_sd5", "area_lag1", "clim_country"])
    t.to_csv(OUT / "model_table.csv", index=False)
    log["model_rows"] = int(len(t))
    log["model_years"] = [int(t["year"].min()), int(t["year"].max())]
    log["shortfall_rate"] = round(float(t["shortfall"].mean()), 4)
    log["rows_per_crop"] = t["crop"].value_counts().to_dict()
    log["rows_per_country"] = t["country"].value_counts().to_dict()
    return t


if __name__ == "__main__":
    parse_seed()
    t = build(clean_fao(), climate(), soil())
    (OUT / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))
