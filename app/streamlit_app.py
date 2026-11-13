"""East Africa crop shortfall advisor: try the model, check its performance, browse the analysis.

Run from the repository root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import PROC, MODELS, latest_state, scenario_row  # noqa: E402

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
BANDS = [(0.35, "Low"), (0.5, "Moderate"), (0.65, "High"), (1.01, "Very high")]
GROUPS = {"Maize": "Cereal", "Sorghum": "Cereal", "Millet": "Cereal", "Wheat": "Cereal", "Rice": "Cereal",
          "Beans": "Pulse", "Cowpea": "Pulse", "Pigeon pea": "Pulse", "Groundnuts": "Oilseed", "Sesame": "Oilseed",
          "Cassava": "Root/tuber", "Sweet potato": "Root/tuber", "Potato": "Root/tuber", "Banana": "Fruit"}
SEED_NAME = {"Sorghum": "Sorghum", "Millet": "Finger Millet", "Cassava": "Cassava", "Cowpea": "Cowpea"}

st.set_page_config(page_title="East Africa crop shortfall advisor", page_icon="🌾", layout="wide")


@st.cache_resource
def load():
    return dict(
        clf=joblib.load(MODELS / "shortfall_classifier.joblib"), reg=joblib.load(MODELS / "yield_regressor.joblib"),
        metrics=json.loads((ROOT / "reports" / "metrics.json").read_text()),
        eda=json.loads((ROOT / "reports" / "eda_summary.json").read_text()),
        state=latest_state(), clim_m=pd.read_csv(PROC / "climatology_monthly.csv"),
        clim_y=pd.read_csv(PROC / "climate_country_year.csv"), yields=pd.read_csv(PROC / "yield_clean.csv"),
        preds=pd.read_csv(PROC / "test_predictions.csv"), seed=pd.read_csv(PROC / "seed_crops.csv"),
        seed_other=pd.read_csv(PROC / "seed_other.csv"))


D = load()
M = D["metrics"]
C = M["classifier"]


def band_of(score):
    return next(name for limit, name in BANDS if score < limit)


def predict(state, rain_pct, temp_delta):
    row = pd.DataFrame([scenario_row(state, D["clim_m"], D["clim_y"], rain_pct, temp_delta, GROUPS)])
    risk = float(D["clf"]["model"].predict_proba(row[D["clf"]["features"]])[0, 1])
    ratio = float(np.exp(D["reg"]["model"].predict(row[D["reg"]["features"]])[0]))
    return risk, ratio, row.iloc[0]


def layout(fig, height=320, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), showlegend=kw.pop("showlegend", False),
                      legend=dict(orientation="h", y=1.12), **kw)
    return fig


st.title("East Africa crop shortfall advisor")
st.caption("Estimates the risk that a crop's national yield falls more than 10% below its recent 5-year average, "
           "using FAOSTAT yield history and NASA POWER weather. A prototype for testing, not an operational forecast.")
tab_try, tab_perf, tab_data, tab_about = st.tabs(["Try the model", "Is the model working?", "Data analysis", "About and limits"])

# ------------------------------------------------------------------ try the model
with tab_try:
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.subheader("Scenario")
        country = st.selectbox("Country", sorted(D["state"]["country"].unique()), index=2)
        crops = sorted(D["state"][D["state"]["country"] == country]["crop"])
        crop = st.selectbox("Crop", crops, index=crops.index("Maize") if "Maize" in crops else 0)
        rain = st.slider("Season rainfall vs normal (%)", -50, 50, 0, 5,
                         help="Scales every month of the 1981-2010 normal rainfall for the country.")
        temp = st.slider("Temperature vs normal (°C)", -1.0, 2.0, 0.0, 0.25)
        state = D["state"][(D["state"]["country"] == country) & (D["state"]["crop"] == crop)].iloc[0]
        st.caption(f"Yield history is filled in automatically from FAOSTAT up to {state['last_year']}. "
                   f"The forecast is for {state['last_year'] + 1}.")
    risk, ratio, row = predict(state, rain, temp)
    band = band_of(risk)
    obs = next(b for b in C["risk_bands"] if b["band"] == band)
    expected = state["yield_mean5"] * ratio
    lo, hi = (expected * np.exp(q) for q in M["regressor"]["resid_q"])
    with right:
        st.subheader(f"{crop} in {country}, {state['last_year'] + 1}")
        a, b, c, d = st.columns(4)
        a.metric("Shortfall risk band", band)
        b.metric("Risk score", f"{risk:.2f}", help="Model score from 0 to 1. Scores at or above "
                 f"{C['threshold']:.2f} are flagged as a likely shortfall.")
        c.metric("Expected yield", f"{expected:,.0f} kg/ha", f"{(ratio - 1) * 100:+.1f}% vs 5-year average")
        d.metric("Last reported yield", f"{state['yield_lag1']:,.0f} kg/ha")
        flag = "flagged as a likely shortfall" if risk >= C["threshold"] else "not flagged"
        st.info(f"**{band} risk, {flag}.** In the test years (2013 onwards), {obs['observed_rate']:.0%} of the "
                f"{obs['n']} cases in this band ended in a shortfall, against {M['data']['shortfall_rate_test']:.0%} overall. "
                f"Likely yield range: {lo:,.0f} to {hi:,.0f} kg/ha (middle 80% of past errors).")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**How rainfall changes the risk score**")
            xs = list(range(-50, 55, 5))
            ys = [predict(state, x, temp)[0] for x in xs]
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=2),
                                       hovertemplate="Rain %{x:+d}%: score %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[rain], y=[risk], mode="markers", marker=dict(color=BLUE, size=11,
                                     line=dict(color="white", width=2)), hoverinfo="skip"))
            fig.add_hline(y=C["threshold"], line=dict(color=MUTED, dash="dash", width=1),
                          annotation_text="flag threshold", annotation_position="top left")
            fig.update_xaxes(title="rainfall vs normal (%)"); fig.update_yaxes(title="risk score", range=[0, 1])
            st.plotly_chart(layout(fig), width='stretch')
        with g2:
            st.markdown("**Reported yield history (kg/ha)**")
            h = D["yields"][(D["yields"]["country"] == country) & (D["yields"]["crop"] == crop)
                            & ~D["yields"]["is_outlier"] & (D["yields"]["year"] >= 1990)]
            fig = go.Figure(go.Scatter(x=h["year"], y=h["yield_kg_ha"], mode="lines", line=dict(color=BLUE, width=2),
                                       name="Reported", hovertemplate="%{x}: %{y:,.0f} kg/ha<extra></extra>"))
            fig.add_trace(go.Scatter(x=[state["last_year"] + 1], y=[expected], mode="markers", name="Model estimate",
                                     error_y=dict(type="data", symmetric=False, array=[hi - expected],
                                                  arrayminus=[expected - lo], color=ORANGE, thickness=1.5),
                                     marker=dict(color=ORANGE, size=10, line=dict(color="white", width=2)),
                                     hovertemplate="Estimate: %{y:,.0f} kg/ha<extra></extra>"))
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(layout(fig, showlegend=True), width='stretch')
        st.caption("Most of the model's skill comes from recent yield history; rainfall shifts the score modestly. "
                   "See the next tab for the evidence.")
    if crop in SEED_NAME:
        s = D["seed"][D["seed"]["crop"] == SEED_NAME[crop]].iloc[0]
        normal = float(row["rain_clim"])
        scen = float(row["rain_annual"])
        verdict = ("within" if s.water_min_mm <= scen <= s.water_max_mm else
                   "above" if scen > s.water_max_mm else "below")
        with st.expander(f"Historical crop profile from the seed dataset: {s.crop} ({s.swahili})", expanded=True):
            p, q = st.columns(2)
            p.markdown(f"- **Water need:** {s.water_min_mm:.0f} to {s.water_max_mm:.0f} mm per year\n"
                       f"- **Scenario rainfall for {country}:** {scen:,.0f} mm ({verdict} the stated range; normal is {normal:,.0f} mm)\n"
                       f"- **Planting season:** {s.planting_season}\n- **Soil:** {s.soil_type}")
            q.markdown(f"- **Historical yield range:** {s.yield_min_kg_ha:,.0f} to {s.yield_max_kg_ha:,.0f} kg/ha\n"
                       f"- **Storage:** {s.storage}\n- **Uses:** {s.uses}\n- **Source cited:** {s.source}")
            st.caption("Seed records are unverified historical notes (pre-1928 sources); they are shown as context "
                       "and are not model inputs.")

# ------------------------------------------------------------------ performance
with tab_perf:
    T, U, N = C["test"], C["test_without_balancing"], C["test_no_skill"]
    cm = C["confusion"]
    st.subheader("Shortfall classifier on years it never saw")
    st.caption(f"{C['selected_model']} with {C['selected_balancing'].lower()}, trained on "
               f"{M['data']['train_years'][0]}-{M['data']['train_years'][1]} ({M['data']['train_rows']:,} rows), "
               f"tested on {M['data']['test_years'][0]}-{M['data']['test_years'][1]} ({M['data']['test_rows']:,} rows).")
    k = st.columns(5)
    k[0].metric("ROC-AUC", f"{T['roc_auc']:.2f}", f"{T['roc_auc'] - 0.5:+.2f} vs chance")
    k[1].metric("PR-AUC", f"{T['pr_auc']:.2f}", f"{T['pr_auc'] - N['pr_auc']:+.2f} vs no-skill")
    k[2].metric("Shortfalls caught (recall)", f"{T['recall']:.0%}", f"{(T['recall'] - U['recall']) * 100:+.0f} pts vs unbalanced")
    k[3].metric("Flags that were right (precision)", f"{T['precision']:.0%}")
    k[4].metric("Balanced accuracy", f"{T['balanced_accuracy']:.0%}", f"{(T['balanced_accuracy'] - U['balanced_accuracy']) * 100:+.0f} pts vs unbalanced")
    st.caption(f"95% bootstrap interval for ROC-AUC: {C['test_ci']['roc_auc'][0]:.2f} to {C['test_ci']['roc_auc'][1]:.2f}.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Does a higher risk band mean more real shortfalls?**")
        rb = pd.DataFrame(C["risk_bands"])
        fig = go.Figure(go.Bar(x=rb["band"], y=rb["observed_rate"] * 100, marker_color=BLUE, width=0.55,
                               text=[f"{v:.0%}" for v in rb["observed_rate"]], textposition="outside",
                               customdata=rb["n"], hovertemplate="%{x}: %{y:.0f}% of %{customdata} cases<extra></extra>"))
        fig.add_hline(y=M["data"]["shortfall_rate_test"] * 100, line=dict(color=MUTED, dash="dash", width=1),
                      annotation_text="overall rate", annotation_position="top left")
        fig.update_yaxes(title="% that ended in a shortfall", range=[0, 80])
        st.plotly_chart(layout(fig), width='stretch')
    with c2:
        st.markdown("**Confusion matrix (test years)**")
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted shortfall"], y=["Actual normal", "Actual shortfall"],
                                   colorscale=[[0, "#cde2fb"], [1, "#184f95"]], showscale=False, xgap=2, ygap=2,
                                   text=z, texttemplate="%{text}", textfont=dict(size=18),
                                   hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(layout(fig), width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**What balancing changed** (same model, test years)")
        st.dataframe(pd.DataFrame({
            "": ["No balancing, 0.5 cut-off", f"{C['selected_balancing']}, tuned cut-off"],
            "Shortfalls caught": [f"{U['recall']:.0%}", f"{T['recall']:.0%}"],
            "Precision": [f"{U['precision']:.0%}", f"{T['precision']:.0%}"],
            "Balanced accuracy": [f"{U['balanced_accuracy']:.0%}", f"{T['balanced_accuracy']:.0%}"],
            "F1": [f"{U['f1']:.2f}", f"{T['f1']:.2f}"]}), hide_index=True, width='stretch')
        st.caption(f"Training data had {M['data']['class_counts_train']['normal']:,} normal years and "
                   f"{M['data']['class_counts_train']['shortfall']:,} shortfall years "
                   f"({M['data']['shortfall_rate_train']:.0%}). Without balancing the model misses most shortfalls.")
        st.markdown("**Where the skill comes from** (ROC-AUC on test years)")
        ab = C["ablation"]
        fig = go.Figure(go.Bar(y=list(ab), x=list(ab.values()), orientation="h", marker_color=BLUE, width=0.5,
                               text=[f"{v:.2f}" for v in ab.values()], textposition="outside",
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.add_vline(x=0.5, line=dict(color=MUTED, dash="dash", width=1), annotation_text="chance")
        fig.update_xaxes(range=[0.4, 0.95]); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(layout(fig, 220), width='stretch')
    with c4:
        st.markdown("**Most influential inputs** (drop in ROC-AUC when shuffled)")
        imp = pd.DataFrame(C["importance"]).head(10)
        fig = go.Figure(go.Bar(y=imp["label"], x=imp["importance"], orientation="h", marker_color=BLUE, width=0.6,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(layout(fig, 380), width='stretch')

    with st.expander("Model and balancing comparison (cross-validation on training years)"):
        cv = pd.DataFrame(C["cv"]).rename(columns={"model": "Model", "balancing": "Balancing", "pr_auc": "PR-AUC",
                                                   "roc_auc": "ROC-AUC", "balanced_accuracy": "Balanced accuracy",
                                                   "f1": "F1", "recall": "Recall", "precision": "Precision", "brier": "Brier"})
        st.dataframe(cv.style.format({c: "{:.3f}" for c in cv.columns[2:]}), hide_index=True, width='stretch')
    with st.expander("Performance by crop (test years)"):
        pc = pd.DataFrame(C["per_crop"]).rename(columns={"crop": "Crop", "n": "Cases", "shortfalls": "Shortfalls",
                                                         "roc_auc": "ROC-AUC", "recall": "Recall", "precision": "Precision"})
        st.dataframe(pc.style.format({"ROC-AUC": "{:.2f}", "Recall": "{:.0%}", "Precision": "{:.0%}"}),
                     hide_index=True, width='stretch')

    st.subheader("Yield estimate")
    R = M["regressor"]["test"]
    st.dataframe(pd.DataFrame({
        "Method": [f"Model ({M['regressor']['selected_model']})", "Baseline: same as last year", "Baseline: 5-year average"],
        "Mean absolute error (kg/ha)": [f"{R[k]['mae_kg_ha']:,.0f}" for k in ["model", "baseline_last_year", "baseline_5yr_mean"]],
        "Median % error": [f"{R[k]['median_pct_error']:.1f}%" for k in ["model", "baseline_last_year", "baseline_5yr_mean"]]}),
        hide_index=True, width='stretch')
    st.warning("The yield model beats the 5-year average but does not beat simply repeating last year's yield. "
               "Treat its number as a scenario indicator, and rely on the risk band for decisions.")

    st.subheader("Check individual past predictions")
    p = D["preds"]
    f1, f2 = st.columns(2)
    pcn = f1.selectbox("Country ", ["All"] + sorted(p["country"].unique()))
    pcr = f2.selectbox("Crop ", ["All"] + sorted(p["crop"].unique()))
    v = p[((p["country"] == pcn) | (pcn == "All")) & ((p["crop"] == pcr) | (pcr == "All"))].copy()
    v["Outcome"] = np.select([(v.flagged == 1) & (v.shortfall == 1), (v.flagged == 1) & (v.shortfall == 0),
                              (v.flagged == 0) & (v.shortfall == 1)], ["Correct warning", "False alarm", "Missed shortfall"],
                             "Correct all-clear")
    v = v.rename(columns={"country": "Country", "crop": "Crop", "year": "Year", "yield_kg_ha": "Actual yield",
                          "yield_mean5": "5-year average", "yield_pred": "Estimated yield", "risk": "Risk score"})
    st.dataframe(v[["Country", "Crop", "Year", "Risk score", "Outcome", "Actual yield", "Estimated yield", "5-year average"]]
                 .sort_values(["Year", "Risk score"], ascending=[False, False])
                 .style.format({"Risk score": "{:.2f}", "Actual yield": "{:,.0f}", "Estimated yield": "{:,.0f}",
                                "5-year average": "{:,.0f}"}), hide_index=True, width='stretch', height=320)

# ------------------------------------------------------------------ analysis
with tab_data:
    E = D["eda"]
    FIG = ROOT / "reports" / "figures"
    st.subheader("The seed dataset")
    st.write("The original dataset holds 8 hand-assembled records: 4 crops, 1 tree, 1 rainfall zone, 1 soil type and "
             "1 water feature. That is too little to train on, so it was parsed into structured crop profiles and "
             f"augmented with FAOSTAT yields and NASA POWER weather, giving {E['rows']:,} country-crop-year rows.")
    st.image(str(FIG / "01_seed_sample.png")); st.image(str(FIG / "02_seed_vs_modern_yield.png"))
    with st.expander("The three non-crop seed records"):
        st.dataframe(D["seed_other"].drop(columns=["tags"]).T.rename(columns=str), width='stretch')
    sections = [
        ("Univariate analysis: one variable at a time", [
            ("03_uni_yield_distributions", "Yields fall into two groups (grains and pulses near 1,000 kg/ha, roots and bananas near 8,000), so the model works with yield relative to each crop's own recent average."),
            ("04_uni_yield_by_crop", None),
            ("05_uni_class_balance", f"Crops are unevenly represented ({E['crop_rows_max_min'][0]} rows for the largest, {E['crop_rows_max_min'][1]} for the smallest) and only {E['shortfall_rate']:.0%} of rows are shortfall years, which is why class balancing is needed."),
            ("06_uni_data_quality", f"Noise: {E['repeats']} values were exact repeats of the previous year (carried-forward estimates) and {E['outliers']} were implausible jumps. Both were removed as targets."),
            ("07_uni_rainfall", None)]),
        ("Bivariate analysis: pairs of variables", [
            ("08_bi_shortfall_by_rainfall", "Dry seasons raise the shortfall rate, most clearly for cereals and pulses with official figures."),
            ("09_bi_shortfall_by_crop_country", None),
            ("10_bi_yield_ratio_drivers", "Recent yield momentum relates to the target far more strongly than rainfall does at national scale."),
            ("11_bi_yield_trends", None)]),
        ("Multivariate analysis: many variables together", [
            ("12_multi_correlation", "Climate variables are strongly inter-correlated (wet places are cooler), so a regularised model is appropriate."),
            ("13_multi_climate_pca", f"Two components capture {sum(E['pca_variance']):.0%} of climate variation and separate the countries into three climate groups."),
            ("14_multi_crop_interactions", None)])]
    for title, figs in sections:
        st.subheader(title)
        for name, note in figs:
            st.image(str(FIG / f"{name}.png"))
            if note:
                st.caption(note)

# ------------------------------------------------------------------ about
with tab_about:
    st.subheader("What this is")
    st.markdown(
        "- **Goal:** flag, for each country and crop, whether the coming harvest is likely to fall more than 10% below "
        "its recent 5-year average, and show a likely yield.\n"
        "- **Data:** the 8-record public-domain seed sample (context only), FAOSTAT crop yields for 10 East African "
        "countries and 14 crops (1981-2024), and NASA POWER monthly rainfall and temperature.\n"
        "- **Method:** logistic regression with class balancing, chosen by cross-validation over three model families "
        "and five balancing strategies, then tested on years after 2012.")
    st.subheader("Limits to keep in mind")
    st.markdown(
        "- Predictions are **national averages**, not field or district level.\n"
        "- Most skill comes from yield history; weather adds little at this scale, so the rainfall slider moves the score modestly.\n"
        "- About 40% of FAOSTAT yield values are estimates rather than official figures.\n"
        "- Country weather is the average of 4 to 6 farming locations per country.\n"
        "- The seed records are unverified and are not used as model inputs.\n"
        "- The risk score is not a calibrated probability; read it through the risk bands.")
