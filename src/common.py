"""Shared feature definitions and the scenario builder used by training and the app."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

CAT = ["crop", "country", "crop_group", "main_season"]
NUM = ["year", "log_yield_lag1", "log_yield_mean5", "yield_trend5", "yield_cv5", "log_area_lag1",
       "prev_shortfall", "rain_annual", "rain_mam", "rain_jjas", "rain_ond", "dry_months",
       "rain_concentration", "temp_mean", "temp_max", "temp_min", "rain_clim", "temp_clim",
       "rain_anom_pct", "rain_z", "temp_anom", "main_season_anom_pct", "rain_anom_prev"]
SOIL = ["soil_ph", "soil_clay_pct", "soil_sand_pct", "soil_soc_g_kg", "soil_n_g_kg", "soil_cec"]
TRAIN_END = 2012
SHORTFALL = 0.90
CLIMATE_OF = {"Sudan (former)": "Sudan", "Ethiopia PDR": "Ethiopia", "Tanzania": "United Republic of Tanzania"}
SEASONS = {"MAM": [3, 4, 5], "JJAS": [6, 7, 8, 9], "OND": [10, 11, 12]}

LABELS = {
    "log_yield_lag1": "Last year's yield", "log_yield_mean5": "5-year average yield",
    "yield_trend5": "Last year vs 5-year average", "yield_cv5": "Yield variability (5 years)",
    "log_area_lag1": "Harvested area (last year)", "prev_shortfall": "Shortfall last year",
    "rain_annual": "Annual rainfall", "rain_mam": "Mar-May rainfall", "rain_jjas": "Jun-Sep rainfall",
    "rain_ond": "Oct-Dec rainfall", "dry_months": "Dry months (<30 mm)",
    "rain_concentration": "Share of rain in wettest month", "temp_mean": "Mean temperature",
    "temp_max": "Hottest month max temp", "temp_min": "Coldest month min temp",
    "rain_clim": "Normal rainfall (1981-2010)", "temp_clim": "Normal temperature (1981-2010)",
    "rain_anom_pct": "Rainfall vs normal (%)", "rain_z": "Rainfall anomaly (z-score)",
    "temp_anom": "Temperature vs normal", "main_season_anom_pct": "Main-season rainfall vs normal (%)",
    "rain_anom_prev": "Previous year's rainfall vs normal (%)", "year": "Year", "crop": "Crop",
    "country": "Country", "crop_group": "Crop group", "main_season": "Main rainy season",
}


def load_table():
    t = pd.read_csv(PROC / "model_table.csv")
    t["log_yield_lag1"] = np.log10(t["yield_lag1"])
    t["log_yield_mean5"] = np.log10(t["yield_mean5"])
    return t


def features(t):
    return CAT + NUM   # soil is constant per country, so it is kept as context only


def latest_state():
    """Most recent history for every country-crop series that is still reported."""
    y = pd.read_csv(PROC / "yield_clean.csv")
    y = y[~y["is_outlier"] & ~y["country"].isin(["Sudan (former)", "Ethiopia PDR"])]
    rows = []
    for (country, crop), g in y.sort_values("year").groupby(["country", "crop"]):
        if g["year"].max() < 2020 or len(g) < 5:
            continue
        last5 = g.tail(5)
        prev5 = g.iloc[-6:-1]
        rows.append(dict(country=country, crop=crop, crop_group=None, last_year=int(g["year"].max()),
                         yield_lag1=last5["yield_kg_ha"].iloc[-1], yield_mean5=last5["yield_kg_ha"].mean(),
                         yield_cv5=last5["yield_kg_ha"].std() / last5["yield_kg_ha"].mean(),
                         log_area_lag1=np.log10(last5["area_ha"].iloc[-1]),
                         prev_shortfall=int(last5["yield_kg_ha"].iloc[-1] < SHORTFALL * prev5["yield_kg_ha"].mean())))
    return pd.DataFrame(rows)


def scenario_row(state, clim_monthly, clim_year, rain_pct, temp_delta, groups):
    """One model input row: a country-crop state plus a rainfall/temperature scenario.

    Monthly normals are scaled by the rainfall change and shifted by the temperature change.
    """
    cc = CLIMATE_OF.get(state["country"], state["country"])
    m = clim_monthly[clim_monthly["country"] == cc].sort_values("month")
    rain = m["rain_mm"].values * (1 + rain_pct / 100)
    cy = clim_year[clim_year["country"] == cc].sort_values("year")
    main = cy["main_season"].iloc[0]
    sd = cy[cy["year"].between(1981, 2010)]["rain_annual"].std()
    rain_clim = m["rain_mm"].sum()
    r = dict(crop=state["crop"], country=state["country"], crop_group=groups[state["crop"]], main_season=main,
             year=state["last_year"] + 1, log_yield_lag1=np.log10(state["yield_lag1"]),
             log_yield_mean5=np.log10(state["yield_mean5"]),
             yield_trend5=state["yield_lag1"] / state["yield_mean5"] - 1, yield_cv5=state["yield_cv5"],
             log_area_lag1=state["log_area_lag1"], prev_shortfall=state["prev_shortfall"],
             rain_annual=rain.sum(), dry_months=int((rain < 30).sum()),
             rain_concentration=rain.max() / max(rain.sum(), 1e-9),
             temp_mean=m["t2m"].mean() + temp_delta, temp_max=m["t2m_max"].max() + temp_delta,
             temp_min=m["t2m_min"].min() + temp_delta, rain_clim=rain_clim, temp_clim=m["t2m"].mean(),
             rain_anom_pct=rain_pct, rain_z=rain_pct / 100 * rain_clim / sd, temp_anom=temp_delta,
             main_season_anom_pct=rain_pct, rain_anom_prev=float(cy["rain_anom_pct"].iloc[-1]))
    for s, months in SEASONS.items():
        r["rain_" + s.lower()] = rain[[i - 1 for i in months]].sum()
    return r
