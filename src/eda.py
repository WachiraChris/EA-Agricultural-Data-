"""Exploratory analysis: univariate, bivariate and multivariate figures.

Writes PNG figures to reports/figures/ and key numbers to reports/eda_summary.json.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import LABELS, PROC, ROOT, load_table

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SEC, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": SEC,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": SEC, "ytick.labelcolor": SEC, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "savefig.dpi": 160, "savefig.bbox": "tight", "legend.frameon": False})
S = {}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def barh(ax, labels, values, color=BLUE, fmt="{:.0f}"):
    ax.barh(labels, values, color=color, height=0.62)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt.format(v), va="center", color=SEC, fontsize=9)
    ax.set_xlim(0, max(values) * 1.14)


t = load_table()
y = pd.read_csv(PROC / "yield_clean.csv")
y = y[y["year"] >= 1981]
clim = pd.read_csv(PROC / "climate_country_year.csv")
seed = pd.read_csv(PROC / "seed_crops.csv")
order = t.groupby("crop")["yield_kg_ha"].median().sort_values().index.tolist()

# ------------------------------------------------------------------ seed sample
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for i, r in seed.iterrows():
    ax[0].plot([r.water_min_mm, r.water_max_mm], [i, i], color=BLUE, lw=6, solid_capstyle="round")
    ax[0].text(r.water_max_mm + 20, i, f"{r.water_min_mm:.0f}-{r.water_max_mm:.0f}", va="center", color=SEC, fontsize=9)
ax[0].set_yticks(range(len(seed)), seed["crop"]); ax[0].invert_yaxis(); ax[0].grid(axis="y", visible=False)
ax[0].set_xlim(0, 1200); ax[0].set_xlabel("mm of rain per year")
ax[0].set_title("Seed sample: stated water requirement")
cat = pd.Series({"Crop": 4, "Tree": 1, "Rainfall zone": 1, "Soil type": 1, "Water feature": 1})
barh(ax[1], cat.index, cat.values)
ax[1].set_title("Seed sample: 8 records by category"); ax[1].set_xlabel("records")
save(fig, "01_seed_sample")

# seed (pre-1928) yield ranges against modern FAOSTAT yields
pairs = {"Sorghum": "Sorghum", "Finger Millet": "Millet", "Cassava": "Cassava", "Cowpea": "Cowpea"}
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
S["seed_vs_modern"] = []
for ax, (sc, fc) in zip(axes, pairs.items()):
    r = seed[seed["crop"] == sc].iloc[0]
    v = y[(y["crop"] == fc) & (y["year"] >= 2014) & ~y["is_outlier"]]["yield_kg_ha"]
    ax.axhspan(r.yield_min_kg_ha, r.yield_max_kg_ha, color="#cde2fb", lw=0)
    ax.scatter(np.random.default_rng(1).uniform(-0.25, 0.25, len(v)), v, s=14, color=ORANGE, alpha=0.7, lw=0)
    ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.grid(axis="x", visible=False)
    ax.set_title(fc); ax.set_ylim(0, max(r.yield_max_kg_ha, v.max()) * 1.08)
    inside = float(((v >= r.yield_min_kg_ha) & (v <= r.yield_max_kg_ha)).mean())
    S["seed_vs_modern"].append(dict(crop=fc, seed_min=r.yield_min_kg_ha, seed_max=r.yield_max_kg_ha,
                                    modern_median=float(v.median()), share_inside=inside, n=int(len(v))))
    ax.set_xlabel(f"{inside:.0%} of country-years\ninside the seed range", fontsize=9)
axes[0].set_ylabel("kg per hectare")
fig.suptitle("Seed yield ranges (blue band) against FAOSTAT country yields, 2014-2024 (orange dots)",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout()
save(fig, "02_seed_vs_modern_yield")

# ------------------------------------------------------------------ univariate
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(np.log10(t["yield_kg_ha"]), bins=45, color=BLUE, edgecolor=SURF, lw=0.6)
ax[0].set_xticks([2, 2.5, 3, 3.5, 4, 4.5], ["100", "316", "1,000", "3,162", "10,000", "31,623"])
ax[0].set_xlabel("yield, kg per hectare (log scale)"); ax[0].set_ylabel("country-crop-years")
ax[0].set_title("Yield has two groups: grains vs roots")
ax[1].hist(t["yield_ratio"].clip(0.4, 1.8), bins=50, color=BLUE, edgecolor=SURF, lw=0.6)
ax[1].axvline(0.9, color=INK, lw=1.2, ls="--")
ax[1].text(0.89, ax[1].get_ylim()[1] * 0.93, "shortfall\nthreshold 0.9", ha="right", color=SEC, fontsize=9)
ax[1].set_xlabel("yield relative to the previous 5-year average"); ax[1].set_title("Target variable: yield ratio")
save(fig, "03_uni_yield_distributions")

fig, ax = plt.subplots(figsize=(11, 4.2))
data = [np.log10(t[t["crop"] == c]["yield_kg_ha"]) for c in order]
bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6, flierprops=dict(marker="o", ms=2.5, mfc=MUTED, mec="none"),
                medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
for b in bp["boxes"]:
    b.set(facecolor="#9ec5f4", edgecolor=BLUE, lw=1)
ax.set_yticks(range(1, len(order) + 1), order); ax.grid(axis="y", visible=False)
ax.set_xticks([2, 2.5, 3, 3.5, 4, 4.5], ["100", "316", "1,000", "3,162", "10,000", "31,623"])
ax.set_xlabel("yield, kg per hectare (log scale)"); ax.set_title("Yield by crop, 1981-2024")
save(fig, "04_uni_yield_by_crop")

fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
c = t["crop"].value_counts(); barh(ax[0], c.index, c.values); ax[0].set_title("Rows per crop")
c = t["country"].value_counts(); barh(ax[1], c.index, c.values); ax[1].set_title("Rows per country")
c = pd.Series({"Normal year": int((t.shortfall == 0).sum()), "Shortfall year": int(t.shortfall.sum())})
barh(ax[2], c.index, c.values); ax[2].set_title(f"Target classes ({t.shortfall.mean():.0%} shortfall)")
fig.tight_layout(); save(fig, "05_uni_class_balance")
S["rows"] = int(len(t)); S["shortfall_rate"] = float(t.shortfall.mean())
S["crop_rows_max_min"] = [int(t.crop.value_counts().max()), int(t.crop.value_counts().min())]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
fl = y["yield_flag"].map({"A": "Official figure", "E": "Estimated", "X": "External organisation"}).value_counts()
barh(ax[0], fl.index, fl.values); ax[0].set_title("FAOSTAT yield values by source flag")
q = pd.Series({"Kept for modelling": int(len(t)), "Repeated value (carried forward)": int(y.is_repeat.sum()),
               "Implausible jump (>3x)": int(y.is_outlier.sum())})
barh(ax[1], q.index, q.values); ax[1].set_title("Noise found in the yield records")
fig.tight_layout(); save(fig, "06_uni_data_quality")
S["flags"] = {k: int(v) for k, v in fl.items()}; S["repeats"] = int(y.is_repeat.sum()); S["outliers"] = int(y.is_outlier.sum())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
co = clim.groupby("country")["rain_annual"].median().sort_values().index
bp = ax[0].boxplot([clim[clim.country == k]["rain_annual"] for k in co], vert=False, patch_artist=True, widths=0.6,
                   flierprops=dict(marker="o", ms=2.5, mfc=MUTED, mec="none"), medianprops=dict(color=INK, lw=1.4),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
for b in bp["boxes"]:
    b.set(facecolor="#9ec5f4", edgecolor=BLUE, lw=1)
ax[0].set_yticks(range(1, len(co) + 1), [k.replace("United Republic of ", "") for k in co]); ax[0].grid(axis="y", visible=False)
ax[0].set_xlabel("mm per year"); ax[0].set_title("Annual rainfall by country, 1981-2024")
ax[1].hist(clim["rain_anom_pct"], bins=35, color=BLUE, edgecolor=SURF, lw=0.6)
ax[1].set_xlabel("% difference from the 1981-2010 normal"); ax[1].set_title("Rainfall anomaly")
fig.tight_layout(); save(fig, "07_uni_rainfall")

# ------------------------------------------------------------------ bivariate
bins = [-100, -20, -10, 10, 20, 400]; labs = ["Very dry\n(< -20%)", "Dry\n(-20 to -10%)", "Near normal", "Wet\n(+10 to +20%)", "Very wet\n(> +20%)"]
t["rain_band"] = pd.cut(t["main_season_anom_pct"], bins, labels=labs)
grain = t[t["crop_group"].isin(["Cereal", "Pulse"]) & t["official"]]
a = t.groupby("rain_band", observed=False)["shortfall"].agg(["mean", "size"])
b = grain.groupby("rain_band", observed=False)["shortfall"].agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(9, 3.9))
x = np.arange(5); w = 0.36
ax.bar(x - w / 2 - 0.01, a["mean"] * 100, w, color=BLUE, label="All crops")
ax.bar(x + w / 2 + 0.01, b["mean"] * 100, w, color=ORANGE, label="Cereals and pulses, official figures only")
for i in range(5):
    ax.text(x[i] - w / 2, a["mean"].iloc[i] * 100 + 0.6, f"{a['mean'].iloc[i]:.0%}", ha="center", color=SEC, fontsize=9)
    ax.text(x[i] + w / 2, b["mean"].iloc[i] * 100 + 0.6, f"{b['mean'].iloc[i]:.0%}", ha="center", color=SEC, fontsize=9)
ax.set_xticks(x, labs); ax.grid(axis="x", visible=False); ax.set_ylabel("% of years with a shortfall")
ax.set_xlabel("main-season rainfall vs normal"); ax.legend(loc="upper right")
ax.set_title("Shortfalls are more common in dry seasons")
save(fig, "08_bi_shortfall_by_rainfall")
S["shortfall_by_rain_all"] = {l.replace("\n", " "): float(v) for l, v in zip(labs, a["mean"])}
S["shortfall_by_rain_grain"] = {l.replace("\n", " "): float(v) for l, v in zip(labs, b["mean"])}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
c = t.groupby("crop")["shortfall"].mean().sort_values(ascending=False) * 100
barh(ax[0], c.index, c.values, fmt="{:.0f}%"); ax[0].set_title("Shortfall rate by crop")
c = t.groupby("country")["shortfall"].mean().sort_values(ascending=False) * 100
barh(ax[1], c.index, c.values, fmt="{:.0f}%"); ax[1].set_title("Shortfall rate by country")
fig.tight_layout(); save(fig, "09_bi_shortfall_by_crop_country")
S["shortfall_by_crop"] = {k: float(v) for k, v in (t.groupby("crop")["shortfall"].mean()).items()}

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
ax[0].scatter(t["yield_trend5"].clip(-0.6, 0.8) * 100, t["yield_ratio"].clip(0.4, 1.8), s=6, color=BLUE, alpha=0.25, lw=0)
ax[0].axhline(0.9, color=INK, lw=1, ls="--")
ax[0].set_xlabel("last year's yield vs its 5-year average (%)"); ax[0].set_ylabel("this year's yield ratio")
ax[0].set_title(f"Yield momentum (r = {t['yield_trend5'].corr(t['yield_ratio']):.2f})")
ax[1].scatter(t["main_season_anom_pct"].clip(-60, 100), t["yield_ratio"].clip(0.4, 1.8), s=6, color=BLUE, alpha=0.25, lw=0)
ax[1].axhline(0.9, color=INK, lw=1, ls="--")
ax[1].set_xlabel("main-season rainfall vs normal (%)")
ax[1].set_title(f"Rainfall (r = {t['main_season_anom_pct'].corr(t['yield_ratio']):.2f})")
fig.tight_layout(); save(fig, "10_bi_yield_ratio_drivers")
S["corr_ratio_trend"] = float(t["yield_trend5"].corr(t["yield_ratio"])); S["corr_ratio_rain"] = float(t["main_season_anom_pct"].corr(t["yield_ratio"]))

fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True)
for ax, c in zip(axes, ["Maize", "Sorghum", "Cassava", "Beans"]):
    g = y[(y.crop == c) & ~y.is_outlier].groupby("year")["yield_kg_ha"].median()
    ax.plot(g.index, g.values, color=BLUE, lw=2); ax.set_title(c); ax.set_ylim(0, g.max() * 1.15)
axes[0].set_ylabel("median kg per hectare")
fig.suptitle("Median yield across East African countries over time", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(); save(fig, "11_bi_yield_trends")

# ------------------------------------------------------------------ multivariate
cols = ["yield_ratio", "shortfall", "yield_trend5", "yield_cv5", "prev_shortfall", "log_area_lag1", "rain_annual",
        "rain_anom_pct", "main_season_anom_pct", "rain_anom_prev", "dry_months", "rain_concentration",
        "temp_mean", "temp_max", "temp_anom", "year"]
cm = t[cols].corr()
names = ["Yield ratio (target)", "Shortfall (target)"] + [LABELS[c] for c in cols[2:]]
fig, ax = plt.subplots(figsize=(9.5, 7.6))
im = ax.imshow(cm, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), names, rotation=45, ha="right"); ax.set_yticks(range(len(cols)), names); ax.grid(False)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{cm.iloc[i, j]:.1f}".replace("-0.0", "0.0"), ha="center", va="center", fontsize=7,
                color="white" if abs(cm.iloc[i, j]) > 0.6 else INK)
fig.colorbar(im, shrink=0.7, label="correlation"); ax.set_title("Correlation matrix")
save(fig, "12_multi_correlation")

cf = ["rain_annual", "rain_mam", "rain_jjas", "rain_ond", "dry_months", "rain_concentration", "temp_mean", "temp_max", "temp_min"]
pca = PCA(2).fit(StandardScaler().fit_transform(clim[cf]))
z = pca.transform(StandardScaler().fit_transform(clim[cf]))
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for s, col, name in [("MAM", BLUE, "Mar-May main rains"), ("JJAS", ORANGE, "Jun-Sep main rains")]:
    m = clim["main_season"] == s
    ax.scatter(z[m, 0], z[m, 1], s=16, color=col, alpha=0.6, lw=0, label=name)
for k, g in pd.DataFrame(z, columns=["a", "b"]).assign(c=clim["country"].values).groupby("c"):
    ax.text(g.a.mean(), g.b.mean(), k.replace("United Republic of ", ""), fontsize=9, color=INK, ha="center",
            fontweight="bold", bbox=dict(fc=SURF, ec="none", alpha=0.75, pad=1.5))
ev = pca.explained_variance_ratio_
ax.set_xlabel(f"component 1 ({ev[0]:.0%} of variance): hot and dry to cool and wet")
ax.set_ylabel(f"component 2 ({ev[1]:.0%} of variance)"); ax.legend(loc="best")
ax.set_title("PCA of climate: each dot is one country-year")
save(fig, "13_multi_climate_pca")
S["pca_variance"] = [float(ev[0]), float(ev[1])]

piv = t.pivot_table(index="crop", columns="rain_band", values="shortfall", aggfunc="mean", observed=False) * 100
piv = piv.loc[piv.iloc[:, 0].sort_values(ascending=False).index]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.5]))
im = ax[0].imshow(piv, cmap=SEQ, aspect="auto", vmin=5, vmax=45)
ax[0].set_xticks(range(5), [l.split("\n")[0] for l in labs]); ax[0].set_yticks(range(len(piv)), piv.index); ax[0].grid(False)
for i in range(piv.shape[0]):
    for j in range(5):
        ax[0].text(j, i, f"{piv.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                   color="white" if piv.iloc[i, j] > 27 else INK)
ax[0].set_title("Shortfall rate (%) by crop and season rainfall")
keep = [c for c in t["country"].value_counts().index if c not in ("Sudan (former)", "Ethiopia PDR")]
p2 = t[t.country.isin(keep)].pivot_table(index="crop", columns="country", values="shortfall", aggfunc="mean") * 100
p2 = p2.loc[piv.index]
im2 = ax[1].imshow(p2, cmap=SEQ, aspect="auto", vmin=5, vmax=45)
ax[1].set_xticks(range(p2.shape[1]), p2.columns, rotation=40, ha="right"); ax[1].set_yticks(range(len(p2)), p2.index); ax[1].grid(False)
for i in range(p2.shape[0]):
    for j in range(p2.shape[1]):
        if not np.isnan(p2.iloc[i, j]):
            ax[1].text(j, i, f"{p2.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                       color="white" if p2.iloc[i, j] > 27 else INK)
ax[1].set_title("Shortfall rate (%) by crop and country (blank = not reported)")
fig.colorbar(im2, ax=ax, shrink=0.75, label="% of years")
save(fig, "14_multi_crop_interactions")

(ROOT / "reports" / "eda_summary.json").write_text(json.dumps(S, indent=1))
print(json.dumps(S, indent=1))
