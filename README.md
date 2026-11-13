# East Africa Agricultural Data: crop shortfall advisor

A prototype that estimates the risk of a poor harvest for 14 crops in 10 East African countries, built on a
small public-domain seed dataset and augmented with open FAOSTAT and NASA POWER data.

- **Full report for third parties:** [reports/REPORT.md](reports/REPORT.md)
- **Interactive app:** `app/streamlit_app.py`

## What is here

| Path | Contents |
|---|---|
| `data/raw/crops_water_soil.json` | The original 8-record seed sample (public-domain sources, unverified) |
| `data/raw/` | FAOSTAT yields and NASA POWER monthly weather, as downloaded |
| `data/processed/` | Cleaned tables, the modelling table (4,131 rows) and test-year predictions |
| `src/fetch_data.py` | Downloads the external data |
| `src/prepare.py` | Cleaning, feature building, targets |
| `src/eda.py` | Univariate, bivariate and multivariate analysis (figures in `reports/figures/`) |
| `src/train.py` | Model and balancing comparison, training, evaluation (`reports/metrics.json`) |
| `app/streamlit_app.py` | Interface to try the model and inspect its performance |
| `models/` | Trained shortfall classifier and yield regressor |

## Headline results (tested on 2013-2024, years the model never saw)

| Measure | Result |
|---|---|
| ROC-AUC | 0.81 (95% interval 0.77 to 0.84) |
| Shortfalls caught (recall) | 58%, up from 26% without class balancing |
| Flags that were right (precision) | 55% |
| Shortfall rate in the "Very high" risk band | 66%, against 7% in the "Low" band |

Most of the skill comes from recent yield history; weather adds little at national scale. The yield
estimate does not beat a "same as last year" baseline. See the report for details and limits.

## Run it

```bash
pip install -r requirements.txt
```

```bash
streamlit run app/streamlit_app.py
```

To rebuild everything from the raw sources:

```bash
python src/fetch_data.py && python src/prepare.py && python src/eda.py && python src/train.py && python src/export_demo.py
```

## Deploy the app online

The app can be hosted free on Streamlit Community Cloud: sign in at share.streamlit.io with GitHub, choose
this repository, the branch, and `app/streamlit_app.py` as the main file.

## Data sources and licences

- Seed sample: CC0, drawn from pre-1928 public-domain reports. Contents are unverified.
- FAOSTAT, Crops and livestock products (FAO, CC BY 4.0).
- NASA POWER monthly meteorology (MERRA-2), freely available.
