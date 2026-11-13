"""Model selection, class balancing comparison, training and evaluation.

Two models are trained on years up to 2012 and tested on 2013 onwards:
  A. Shortfall classifier: will yield fall below 90% of its trailing 5-year mean?
  B. Yield regressor: expected yield relative to the trailing 5-year mean.

Outputs: models/*.joblib, reports/metrics.json, data/processed/test_predictions.csv
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, mean_absolute_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CAT, LABELS, MODELS, ROOT, TRAIN_END, features, load_table

warnings.filterwarnings("ignore")
SEED = 42
MODELS.mkdir(exist_ok=True)

t = load_table().sort_values("year").reset_index(drop=True)
FEATS = features(t)
NUMS = [f for f in FEATS if f not in CAT]
train, test = t[t["year"] <= TRAIN_END], t[t["year"] > TRAIN_END]
Xtr, Xte = train[FEATS], test[FEATS]
ytr, yte = train["shortfall"].values, test["shortfall"].values

# Expanding-window folds: train on the past, validate on the next block of years.
BLOCKS = [(1996, 1999), (2000, 2003), (2004, 2007), (2008, 2012)]
FOLDS = [(np.where(train["year"] < a)[0], np.where(train["year"].between(a, b))[0]) for a, b in BLOCKS]


def prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMS)])


def clf(name, weighted):
    w = "balanced" if weighted else None
    if name == "Logistic regression":
        return LogisticRegression(max_iter=2000, C=0.3, class_weight=w)
    if name == "Random forest":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=8, max_features="sqrt",
                                      class_weight=w, n_jobs=-1, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_depth=3, min_samples_leaf=25,
                                          l2_regularization=1.0, class_weight=w, random_state=SEED)


SAMPLERS = {"None": None, "Class weights": None, "Random oversampling": RandomOverSampler(random_state=SEED),
            "SMOTE": SMOTE(random_state=SEED, k_neighbors=5),
            "Random undersampling": RandomUnderSampler(random_state=SEED)}


def make(model, balance):
    steps = [("prep", prep())]
    if SAMPLERS[balance] is not None:
        steps.append(("balance", SAMPLERS[balance]))
    steps.append(("model", clf(model, balance == "Class weights")))
    return ImbPipeline(steps)


def oof(pipe):
    """Out-of-fold probabilities for the validation blocks (NaN for the first, train-only years)."""
    p = np.full(len(train), np.nan)
    for tr, va in FOLDS:
        p[va] = pipe.fit(Xtr.iloc[tr], ytr[tr]).predict_proba(Xtr.iloc[va])[:, 1]
    return p


def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                balanced_accuracy=balanced_accuracy_score(y, pred), f1=f1_score(y, pred),
                recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                brier=brier_score_loss(y, p))


# ---------------------------------------------------------------- A. classifier
rows, oofs = [], {}
mask = ~np.isnan(oof(make("Logistic regression", "None")))
for model in ["Logistic regression", "Random forest", "Gradient boosting"]:
    for balance in SAMPLERS:
        p = oof(make(model, balance))
        oofs[(model, balance)] = p
        s = scores(ytr[mask], p[mask], 0.5)
        rows.append(dict(model=model, balancing=balance, **s))
        print(f"{model:20s} {balance:22s} PR-AUC {s['pr_auc']:.3f} ROC {s['roc_auc']:.3f} "
              f"bal-acc {s['balanced_accuracy']:.3f} F1 {s['f1']:.3f} recall {s['recall']:.3f}")
cv = pd.DataFrame(rows)

# Selection: rank on PR-AUC (ranking quality) and balanced accuracy at 0.5 (does it find the minority class).
cv["rank_score"] = cv["pr_auc"].rank(ascending=False) + cv["balanced_accuracy"].rank(ascending=False)
best = cv.sort_values(["rank_score", "pr_auc"], ascending=[True, False]).iloc[0]
bm, bb = best["model"], best["balancing"]
print("\nSelected:", bm, "+", bb)

# Decision threshold chosen on out-of-fold predictions (never on the test set).
po = oofs[(bm, bb)][mask]
prec, rec, thr = precision_recall_curve(ytr[mask], po)
f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
threshold = float(thr[f1s.argmax()])

final = make(bm, bb).fit(Xtr, ytr)
pte = final.predict_proba(Xte)[:, 1]
test_scores = scores(yte, pte, threshold)
dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
unbal = make(bm, "None").fit(Xtr, ytr)
unbal_scores = scores(yte, unbal.predict_proba(Xte)[:, 1], 0.5)
cm = confusion_matrix(yte, (pte >= threshold).astype(int))
fpr, tpr, _ = roc_curve(yte, pte)
pr_p, pr_r, _ = precision_recall_curve(yte, pte)

# Bootstrap interval for the headline test metrics.
rng = np.random.default_rng(SEED)
boot = []
for _ in range(1000):
    i = rng.integers(0, len(yte), len(yte))
    if yte[i].min() == yte[i].max():
        continue
    boot.append((roc_auc_score(yte[i], pte[i]), average_precision_score(yte[i], pte[i])))
boot = np.array(boot)

# Risk bands: does a higher predicted risk mean more observed shortfalls?
bands = pd.cut(pte, [0, 0.35, 0.5, 0.65, 1.0], labels=["Low", "Moderate", "High", "Very high"], include_lowest=True)
band_tab = pd.DataFrame({"band": bands, "y": yte, "p": pte}).groupby("band", observed=False).agg(
    n=("y", "size"), observed_rate=("y", "mean"), mean_score=("p", "mean")).reset_index()

per_crop = []
for c, g in test.assign(p=pte).groupby("crop"):
    if g["shortfall"].nunique() == 2:
        per_crop.append(dict(crop=c, n=len(g), shortfalls=int(g["shortfall"].sum()),
                             roc_auc=roc_auc_score(g["shortfall"], g["p"]),
                             recall=recall_score(g["shortfall"], g["p"] >= threshold),
                             precision=precision_score(g["shortfall"], g["p"] >= threshold, zero_division=0)))

imp = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1)
importance = sorted([dict(feature=f, label=LABELS.get(f, f), importance=float(m), sd=float(s))
                     for f, m, s in zip(FEATS, imp.importances_mean, imp.importances_std)],
                    key=lambda d: -d["importance"])

# Ablation: how much skill comes from yield history and how much from weather?
HIST = ["year", "log_yield_lag1", "log_yield_mean5", "yield_trend5", "yield_cv5", "log_area_lag1", "prev_shortfall"]
WX = [f for f in NUMS if f not in HIST]
ablation = {}
for label, cat, num in [("Yield history only", [], HIST), ("Weather and location only", CAT, WX),
                        ("All features", CAT, NUMS)]:
    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)])
    m = ImbPipeline([("prep", ct), ("balance", RandomUnderSampler(random_state=SEED)),
                     ("model", LogisticRegression(max_iter=2000, C=0.3))]).fit(train[cat + num], ytr)
    ablation[label] = roc_auc_score(yte, m.predict_proba(test[cat + num])[:, 1])
print("ablation", ablation)

# ---------------------------------------------------------------- B. regressor
ytr_r, yte_r = np.log(train["yield_ratio"].values), np.log(test["yield_ratio"].values)
REG = {
    "Ridge regression": Ridge(alpha=10.0),
    "Random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=8, max_features=0.4,
                                           n_jobs=-1, random_state=SEED),
    "Gradient boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_depth=3,
                                                       min_samples_leaf=25, l2_regularization=1.0,
                                                       loss="absolute_error", random_state=SEED),
}


def mae_kg(frame, log_ratio_pred):
    return mean_absolute_error(frame["yield_kg_ha"], frame["yield_mean5"] * np.exp(log_ratio_pred))


reg_rows = []
for name, est in REG.items():
    pipe = Pipeline([("prep", prep()), ("model", est)])
    p = np.full(len(train), np.nan)
    for tr, va in FOLDS:
        p[va] = pipe.fit(Xtr.iloc[tr], ytr_r[tr]).predict(Xtr.iloc[va])
    reg_rows.append(dict(model=name, cv_mae_log_ratio=mean_absolute_error(ytr_r[mask], p[mask]),
                         cv_mae_kg_ha=mae_kg(train[mask], p[mask])))
reg_cv = pd.DataFrame(reg_rows)
rbest = reg_cv.sort_values("cv_mae_log_ratio").iloc[0]["model"]
reg = Pipeline([("prep", prep()), ("model", REG[rbest])]).fit(Xtr, ytr_r)
pr = reg.predict(Xte)
resid_q = np.quantile(ytr_r[mask] - p[mask] if rbest == list(REG)[-1] else
                      ytr_r - reg.predict(Xtr), [0.1, 0.9])  # spread used for the app's range
yhat = test["yield_mean5"].values * np.exp(pr)


def pct_err(pred):
    return float(np.median(np.abs(pred / test["yield_kg_ha"].values - 1)) * 100)


reg_test = {
    "model": dict(mae_kg_ha=mean_absolute_error(test["yield_kg_ha"], yhat), median_pct_error=pct_err(yhat),
                  r2_log_yield=r2_score(np.log(test["yield_kg_ha"]), np.log(yhat)),
                  r2_ratio=r2_score(yte_r, pr)),
    "baseline_5yr_mean": dict(mae_kg_ha=mean_absolute_error(test["yield_kg_ha"], test["yield_mean5"]),
                              median_pct_error=pct_err(test["yield_mean5"].values),
                              r2_log_yield=r2_score(np.log(test["yield_kg_ha"]), np.log(test["yield_mean5"])),
                              r2_ratio=r2_score(yte_r, np.zeros(len(yte_r)))),
    "baseline_last_year": dict(mae_kg_ha=mean_absolute_error(test["yield_kg_ha"], test["yield_lag1"]),
                               median_pct_error=pct_err(test["yield_lag1"].values),
                               r2_log_yield=r2_score(np.log(test["yield_kg_ha"]), np.log(test["yield_lag1"])),
                               r2_ratio=r2_score(yte_r, np.log(test["yield_lag1"] / test["yield_mean5"]))),
}

# ---------------------------------------------------------------- save
joblib.dump(dict(model=final, threshold=threshold, features=FEATS, name=bm, balancing=bb),
            MODELS / "shortfall_classifier.joblib")
joblib.dump(dict(model=reg, features=FEATS, name=rbest, resid_q=resid_q.tolist()), MODELS / "yield_regressor.joblib")
out = test[["country", "crop", "year", "yield_kg_ha", "yield_mean5", "yield_ratio", "shortfall",
            "rain_anom_pct", "official"]].copy()
out["risk"] = pte
out["flagged"] = (pte >= threshold).astype(int)
out["yield_pred"] = yhat
out.to_csv(ROOT / "data" / "processed" / "test_predictions.csv", index=False)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


step = max(1, len(fpr) // 60)
metrics = clean(dict(
    data=dict(rows=len(t), train_rows=len(train), test_rows=len(test), train_years=[int(train.year.min()), TRAIN_END],
              test_years=[TRAIN_END + 1, int(test.year.max())], features=len(FEATS),
              shortfall_rate_train=ytr.mean(), shortfall_rate_test=yte.mean(),
              class_counts_train=dict(normal=int((ytr == 0).sum()), shortfall=int((ytr == 1).sum()))),
    classifier=dict(selected_model=bm, selected_balancing=bb, threshold=threshold,
                    cv=cv.drop(columns="rank_score").to_dict("records"), test=test_scores,
                    test_ci=dict(roc_auc=np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                                 pr_auc=np.percentile(boot[:, 1], [2.5, 97.5]).tolist()),
                    test_without_balancing=unbal_scores,
                    test_no_skill=dict(pr_auc=yte.mean(), roc_auc=0.5, balanced_accuracy=0.5, recall=0.0,
                                       brier=brier_score_loss(yte, dummy.predict_proba(Xte)[:, 1])),
                    confusion=dict(tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1]),
                    roc=dict(fpr=fpr[::step].tolist() + [1.0], tpr=tpr[::step].tolist() + [1.0]),
                    pr=dict(recall=pr_r[::max(1, len(pr_r) // 60)].tolist(),
                            precision=pr_p[::max(1, len(pr_p) // 60)].tolist()),
                    risk_bands=band_tab.to_dict("records"), ablation=ablation, per_crop=per_crop, importance=importance),
    regressor=dict(selected_model=rbest, cv=reg_cv.to_dict("records"), test=reg_test, resid_q=resid_q.tolist()),
))
(ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=1))
print("\nthreshold", round(threshold, 3), "\nTEST classifier:", {k: round(v, 3) for k, v in test_scores.items()})
print("TEST without balancing @0.5:", {k: round(v, 3) for k, v in unbal_scores.items()})
print("confusion", cm.tolist(), "\nbands\n", band_tab)
print("regressor", rbest, json.dumps(clean(reg_test), indent=1))
print("top features", [(d["label"], round(d["importance"], 4)) for d in importance[:8]])
