"""Pre-compute model outputs over a scenario grid for the static HTML report."""
import json
import joblib
import numpy as np
import pandas as pd
from common import MODELS, PROC, ROOT, latest_state, scenario_row

GROUPS = pd.read_csv(PROC / "model_table.csv").drop_duplicates("crop").set_index("crop")["crop_group"].to_dict()
clf = joblib.load(MODELS / "shortfall_classifier.joblib")
reg = joblib.load(MODELS / "yield_regressor.joblib")
cm, cy = pd.read_csv(PROC / "climatology_monthly.csv"), pd.read_csv(PROC / "climate_country_year.csv")
RAIN = list(range(-50, 51, 10))
out = []
for _, s in latest_state().iterrows():
    rows = pd.DataFrame([scenario_row(s, cm, cy, r, 0.0, GROUPS) for r in RAIN])
    risk = clf["model"].predict_proba(rows[clf["features"]])[:, 1]
    ratio = np.exp(reg["model"].predict(rows[reg["features"]]))
    out.append(dict(country=s["country"], crop=s["crop"], year=int(s["last_year"]) + 1,
                    last=round(float(s["yield_lag1"])), mean5=round(float(s["yield_mean5"])),
                    normal_rain=round(float(rows["rain_clim"].iloc[0])),
                    risk=[round(float(v), 3) for v in risk], ratio=[round(float(v), 3) for v in ratio]))
(ROOT / "reports" / "demo_grid.json").write_text(json.dumps(dict(rain=RAIN, cases=out), separators=(",", ":")))
print(len(out), "country-crop cases")
