"""
Uncertainty-aware smart irrigation: soil moisture forecasting UI
Run:  python3 -m streamlit run app.py
Keep this file in the same folder as sm_Uttarakhand_2020.csv (or upload the CSV in the sidebar).
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Soil moisture forecasting",
    page_icon="💧",
    layout="wide",
)


# ------------------------------------------------------------
# UI THEME: Change colors and visual styling in this section
# ------------------------------------------------------------

st.markdown("""
<style>
/* ===== MAIN BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg, #f4fbf6 0%, #eef8fb 52%, #f7fbff 100%
    );
    color: #173b32;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* ===== SIDEBAR GRADIENT ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #103d32 0%, #126653 55%, #168b78 100%
    );
}

[data-testid="stSidebar"] * {
    color: #f4fffb;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    color: #173b32 !important;
    background-color: #ffffff !important;
    border-radius: 10px;
}

/* ===== CSV UPLOAD AREA ===== */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border: 1px dashed #64a994 !important;
    border-radius: 12px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section *,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section small {
    color: #244f43 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: linear-gradient(
        100deg, #087f70, #24a98d
    ) !important;
    color: #ffffff !important;
    border: none !important;
}

/* ===== HERO BANNER ===== */
.hero-banner {
    padding: 1.7rem 2rem;
    border-radius: 22px;
    background: linear-gradient(
        115deg, #0b3d2e 0%, #087f70 58%, #35b7a0 100%
    );
    color: white;
    margin: 0.2rem 0 1.3rem 0;
    box-shadow: 0 12px 30px rgba(11, 90, 70, 0.16);
}

.hero-banner h1 {
    color: white !important;
    font-size: clamp(1.8rem, 3vw, 2.7rem);
    line-height: 1.15;
    margin: 0 0 0.55rem 0;
    font-weight: 800;
}

.hero-banner p {
    color: #e4fff6 !important;
    font-size: 1rem;
    margin: 0;
    line-height: 1.6;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-bottom: 0.8rem;
    letter-spacing: 0.04em;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.95);
    border: 1px solid #d7eee4;
    padding: 1rem 1.1rem;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(20, 90, 70, 0.07);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #52766b !important;
    font-weight: 650;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #087f70 !important;
    font-weight: 800;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
    border-bottom: 1px solid #d5e9e1;
}

.stTabs [data-baseweb="tab"] {
    background: #ffffff !important;
    border-radius: 12px 12px 0 0;
    padding: 0.65rem 1rem;
    color: #244f43 !important;
    font-weight: 650;
    opacity: 1 !important;
}

.stTabs [data-testid="stMarkdownContainer"] p {
    color: #244f43 !important;
    opacity: 1 !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(
        100deg, #d7f5e8, #d9f4f3
    ) !important;
    color: #075b4c !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"]
[data-testid="stMarkdownContainer"] p {
    color: #075b4c !important;
}
/* ===== MAIN PAGE TEXT AND FORM LABELS ===== */
[data-testid="stAppViewContainer"] .main h1,
[data-testid="stAppViewContainer"] .main h2,
[data-testid="stAppViewContainer"] .main h3,
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] [data-testid="stText"],
[data-testid="stAppViewContainer"] [data-testid="stSlider"] label,
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label,
[data-testid="stAppViewContainer"] [data-testid="stSelectbox"] label,
[data-testid="stAppViewContainer"] [data-testid="stMultiSelect"] label {
    color: #244f43;
}

/* Slider values and radio option text */
[data-testid="stSlider"] *,
[data-testid="stRadio"] label,
[data-testid="stRadio"] label *,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {
    opacity: 1 !important;
}

/* ===== WARNING BOX ===== */
[data-testid="stAlert"] {
    background: #fff8d9 !important;
    border: 1px solid #ead58b !important;
    border-radius: 12px !important;
}

[data-testid="stAlert"],
[data-testid="stAlert"] *,
[data-testid="stAlert"] p {
    color: #594500 !important;
    opacity: 1 !important;
}

/* ===== BUTTONS ===== */
.stButton > button,
.stFormSubmitButton > button,
[data-testid="stDownloadButton"] > button {
    border: none !important;
    border-radius: 11px;
    color: #ffffff !important;
    background: linear-gradient(
        100deg, #087f70, #24a98d
    ) !important;
    font-weight: 700;
    padding: 0.55rem 1rem;
    box-shadow: 0 5px 12px rgba(8, 127, 112, 0.16);
    transition: transform 0.18s ease, filter 0.18s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
}

/* ===== TABLES AND EXPANDERS ===== */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #dcece5;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid #dcece5;
    border-radius: 12px;
}

[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    border-radius: 10px;
}

h2, h3 {
    color: #164d40;
}

hr {
    border-color: #d7eae2;
}

.stTabs [data-testid="stTab"] p {
    color: #244f43 !important;
}

.stTabs [data-testid="stTab"][aria-selected="true"] p {
    color: #075b4c !important;
}

</style>
""", unsafe_allow_html=True)


DATE_COL = "Date"
DISTRICT_COL = "DistrictName"
TARGET = "Aggregate Soilmoisture Percentage (at 15cm)"
DEFAULT_PATH = Path(__file__).parent / "sm_Uttarakhand_2020.csv"
ALL = "All districts (average)"


# ------------------------------------------------------------
# Data loading and cleaning
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean(file_bytes):
    if file_bytes is None:
        if not DEFAULT_PATH.exists():
            raise FileNotFoundError(
                f"Could not find {DEFAULT_PATH.name} next to app.py. "
                "Upload the CSV in the sidebar."
            )
        df = pd.read_csv(DEFAULT_PATH)
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    missing = [c for c in (DATE_COL, DISTRICT_COL, TARGET) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Found: {df.columns.tolist()}"
        )

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[DATE_COL, DISTRICT_COL, TARGET]).copy()
    df[DISTRICT_COL] = df[DISTRICT_COL].astype(str).str.strip()
    df = df[df[DISTRICT_COL] != ""].copy()
    df = df.sort_values([DISTRICT_COL, DATE_COL]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain after cleaning.")
    return df


# ------------------------------------------------------------
# Modelling pipeline (cached, so it only retrains when settings change)
# ------------------------------------------------------------
def score(name, actual, predicted):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2_score(actual, predicted),
    }


@st.cache_data(show_spinner="Training models...")
def run_pipeline(df, test_frac, level, rf_trees, rf_depth,
                 xgb_trees, xgb_depth, xgb_lr):
    d = df.copy()
    g = d.groupby(DISTRICT_COL)

    d["target_next"] = g[TARGET].shift(-1)
    d["target_date"] = g[DATE_COL].shift(-1)
    d["lag_1"] = g[TARGET].shift(1)
    d["lag_2"] = g[TARGET].shift(2)
    d["previous_mean_3"] = g[TARGET].transform(
        lambda s: s.shift(1).rolling(window=3).mean()
    )
    d["month"] = d[DATE_COL].dt.month
    d = d.dropna(
        subset=["target_next", "target_date", "lag_1", "lag_2", "previous_mean_3"]
    ).reset_index(drop=True)

    if d["target_date"].nunique() < 3:
        raise ValueError("Too few distinct dates for a chronological split.")

    numeric = [TARGET, "lag_1", "lag_2", "previous_mean_3", "month"]
    dummies = pd.get_dummies(d[DISTRICT_COL], prefix="district", dtype=int)
    X = pd.concat([d[numeric], dummies], axis=1)
    y = d["target_next"]

    dates = np.array(sorted(d["target_date"].unique()))
    split = min(max(int(len(dates) * (1 - test_frac)), 1), len(dates) - 1)
    cutoff = dates[split]

    train = d["target_date"].to_numpy() < cutoff
    test = ~train
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    if len(X_train) < 10 or len(X_test) < 2:
        raise ValueError("Too few training or test rows. Lower the test share.")

    # Models
    persistence = X_test[TARGET].to_numpy()

    rf = RandomForestRegressor(
        n_estimators=rf_trees, max_depth=rf_depth, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=xgb_trees, max_depth=xgb_depth, learning_rate=xgb_lr,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    metrics = pd.DataFrame([
        score("Persistence baseline", y_test, persistence),
        score("Random Forest", y_test, rf_pred),
        score("XGBoost", y_test, xgb_pred),
    ])

    # Uncertainty: central interval from quantile models
    lo_q = (1 - level) / 2
    hi_q = 1 - lo_q

    gb_lo = GradientBoostingRegressor(
        loss="quantile", alpha=lo_q, n_estimators=100, max_depth=2, random_state=42
    ).fit(X_train, y_train).predict(X_test)
    gb_hi = GradientBoostingRegressor(
        loss="quantile", alpha=hi_q, n_estimators=100, max_depth=2, random_state=42
    ).fit(X_train, y_train).predict(X_test)

    xq = dict(n_estimators=xgb_trees, max_depth=min(xgb_depth, 3),
              learning_rate=xgb_lr, subsample=0.8, colsample_bytree=0.8,
              objective="reg:quantileerror", random_state=42, n_jobs=-1)
    xg_lo = XGBRegressor(quantile_alpha=lo_q, **xq).fit(X_train, y_train).predict(X_test)
    xg_hi = XGBRegressor(quantile_alpha=hi_q, **xq).fit(X_train, y_train).predict(X_test)

    actual = y_test.to_numpy()
    gb_lower, gb_upper = np.minimum(gb_lo, gb_hi), np.maximum(gb_lo, gb_hi)
    xg_lower, xg_upper = np.minimum(xg_lo, xg_hi), np.maximum(xg_lo, xg_hi)

    def cov(lo, hi):
        return float(np.mean((actual >= lo) & (actual <= hi)))

    uncertainty = pd.DataFrame([
        {"Method": "Gradient Boosting quantile", "Nominal": level,
         "Observed coverage": cov(gb_lower, gb_upper),
         "Mean width": float(np.mean(gb_upper - gb_lower))},
        {"Method": "XGBoost quantile", "Nominal": level,
         "Observed coverage": cov(xg_lower, xg_upper),
         "Mean width": float(np.mean(xg_upper - xg_lower))},
    ])

    results = pd.DataFrame({
        "Date": d.loc[test, "target_date"].to_numpy(),
        "District": d.loc[test, DISTRICT_COL].to_numpy(),
        "Actual": actual,
        "Persistence": persistence,
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
        "GB lower": gb_lower, "GB upper": gb_upper,
        "XGB lower": xg_lower, "XGB upper": xg_upper,
    }).sort_values(["Date", "District"]).reset_index(drop=True)

    rf_imp = pd.DataFrame({"Feature": X_train.columns,
                           "Importance": rf.feature_importances_}
                          ).sort_values("Importance", ascending=False)
    xgb_imp = pd.DataFrame({"Feature": X_train.columns,
                            "Importance": xgb.feature_importances_}
                           ).sort_values("Importance", ascending=False)

    return {
        "metrics": metrics, "uncertainty": uncertainty, "results": results,
        "rf_imp": rf_imp, "xgb_imp": xgb_imp, "y_train": y_train.to_numpy(),
        "cutoff": pd.Timestamp(cutoff), "n_train": len(X_train), "n_test": len(X_test),
    }


def decide(lower, upper, threshold):
    if upper < threshold:
        return "Alert: verify need for irrigation"
    if lower > threshold:
        return "Wait: moisture predicted above threshold"
    return "Uncertain: check sensor and data, then reassess"


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("Settings")
upload = st.sidebar.file_uploader("Dataset (CSV)", type="csv")
st.sidebar.caption(
    "Leave empty to use sm_Uttarakhand_2020.csv from the app folder."
)

with st.sidebar.form("settings"):
    st.markdown("**Data split and uncertainty**")
    test_pct = st.slider("Test share (latest dates)", 10, 40, 20, step=5,
                         format="%d%%")
    level_pct = st.slider("Prediction interval", 50, 95, 80, step=5,
                          format="%d%%")

    st.markdown("**Random Forest**")
    rf_trees = st.slider("Trees", 50, 500, 200, step=50, key="rf_t")
    rf_depth = st.slider("Max depth", 2, 20, 10, key="rf_d")

    st.markdown("**XGBoost**")
    xgb_trees = st.slider("Boosting rounds", 50, 600, 300, step=50, key="x_t")
    xgb_depth = st.slider("Max depth", 2, 10, 4, key="x_d")
    xgb_lr = st.select_slider("Learning rate",
                              options=[0.01, 0.02, 0.05, 0.1, 0.2], value=0.05)

    st.form_submit_button("Train models", type="primary")

# ------------------------------------------------------------
# Load and run
# ------------------------------------------------------------

# ===== LOAD DATA BEFORE DISPLAYING THE HERO =====

# data-loading function and variables.
# If df is already created earlier in your app, do not load it again.
if "df" not in globals():
    df = load_clean(upload.getvalue() if upload is not None else None)

# ===== DYNAMIC SUBTITLE =====

DISTRICT_COL = "DistrictName"

if DISTRICT_COL in df.columns and not df[DISTRICT_COL].dropna().empty:
    districts = df[DISTRICT_COL].dropna().unique()
    district_count = len(districts)

    if district_count == 1:
        subtitle = f"Soil-moisture forecasting for {districts[0]}"
    else:
        subtitle = f"Soil-moisture forecasting across {district_count} districts"
else:
    subtitle = "Soil-moisture forecasting and predictive analytics"

# ===== HERO BANNER =====

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-tag">
        🌱 AI-POWERED AGRICULTURE • RESEARCH DASHBOARD
    </div>
    <h1>Smart Irrigation Intelligence</h1>
    <p>
        <strong>{subtitle}</strong><br>
        Compare machine-learning forecasts, inspect prediction uncertainty,
        and explore demonstration irrigation decisions from one workspace.
    </p>
</div>
""", unsafe_allow_html=True)


try:
    df = load_clean(upload.getvalue() if upload else None)
    out = run_pipeline(
        df, test_pct / 100, level_pct / 100, rf_trees, rf_depth,
        xgb_trees, xgb_depth, xgb_lr,
    )
except Exception as e:
    st.error(f"Could not run the analysis. {e}")
    st.stop()

metrics, unc, res = out["metrics"], out["uncertainty"], out["results"]
districts = sorted(df[DISTRICT_COL].unique())

tab_data, tab_models, tab_fc, tab_feat, tab_irr, tab_dl = st.tabs(
    ["Data", "Model comparison", "Forecast", "Feature importance",
     "Irrigation decisions", "Downloads"]
)

# ------------------------------------------------------------
# Data tab
# ------------------------------------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Districts", df[DISTRICT_COL].nunique())
    c3.metric("First date", df[DATE_COL].min().strftime("%d %b %Y"))
    c4.metric("Last date", df[DATE_COL].max().strftime("%d %b %Y"))

    left, right = st.columns(2)
    with left:
        st.subheader("Soil moisture distribution")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(df[TARGET], bins=30, edgecolor="black")
        ax.set_xlabel("Soil moisture (%)")
        ax.set_ylabel("Observations")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("Trend over time")
        choice = st.selectbox("District", [ALL] + districts, key="trend_d")
        sub = df if choice == ALL else df[df[DISTRICT_COL] == choice]
        series = sub.groupby(DATE_COL)[TARGET].mean().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(series.index, series.values, marker=".")
        ax.set_ylabel("Soil moisture (%)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

    with st.expander("Preview cleaned data"):
        st.dataframe(df.head(50), use_container_width=True)

# ------------------------------------------------------------
# Model comparison tab
# ------------------------------------------------------------
with tab_models:
    best = metrics.loc[metrics["RMSE"].idxmin()]
    st.success(
        f"Lowest error: {best['Model']} (RMSE {best['RMSE']:.3f}). "
        f"Trained on {out['n_train']:,} rows, tested on {out['n_test']:,} rows "
        f"from {out['cutoff'].strftime('%d %b %Y')} onward."
    )
    st.subheader("Accuracy on the test period")
    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
    st.bar_chart(metrics.set_index("Model")[["RMSE", "MAE"]])
    st.caption(
        "Persistence predicts that the next reading equals the current one. "
        "A model is only useful if it beats this baseline."
    )

    st.subheader("Prediction interval quality")
    st.dataframe(unc.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Observed coverage should be close to the nominal level. "
        "Much lower means the intervals are overconfident."
    )

# ------------------------------------------------------------
# Forecast tab
# ------------------------------------------------------------
with tab_fc:
    f1, f2, f3 = st.columns([2, 2, 3])
    district = f1.selectbox("District", [ALL] + districts, key="fc_d")
    interval = f2.radio("Interval from", ["Gradient Boosting", "XGBoost"],
                        horizontal=True)
    shown = f3.multiselect("Models to show", ["Random Forest", "XGBoost",
                                              "Persistence"],
                           default=["Random Forest", "XGBoost"])

    lo_col, hi_col = (("GB lower", "GB upper") if interval == "Gradient Boosting"
                      else ("XGB lower", "XGB upper"))

    sub = res if district == ALL else res[res["District"] == district]
    sub = sub.drop(columns="District").groupby("Date", as_index=False).mean()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(sub["Date"], sub["Actual"], label="Actual", color="black",
            marker=".", linewidth=2)
    for name in shown:
        ax.plot(sub["Date"], sub[name], label=name, marker=".", alpha=0.85)
    ax.fill_between(sub["Date"], sub[lo_col], sub[hi_col], alpha=0.2,
                    label=f"{level_pct}% interval ({interval})")
    ax.set_xlabel("Target observation date")
    ax.set_ylabel("Soil moisture (%)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

    inside = float(((sub["Actual"] >= sub[lo_col]) &
                    (sub["Actual"] <= sub[hi_col])).mean())
    st.metric("Actual values inside the interval (this view)", f"{inside:.1%}")

# ------------------------------------------------------------
# Feature importance tab
# ------------------------------------------------------------
with tab_feat:
    top_n = st.slider("Features to show", 5, 20, 10)
    a, b = st.columns(2)
    for col, title, imp in ((a, "Random Forest", out["rf_imp"]),
                            (b, "XGBoost", out["xgb_imp"])):
        with col:
            st.subheader(title)
            top = imp.head(top_n).sort_values("Importance")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(top["Feature"], top["Importance"])
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)

# ------------------------------------------------------------
# Irrigation decisions tab
# ------------------------------------------------------------
with tab_irr:
    st.warning(
        "Demonstration only. The threshold below is a percentile of training "
        "data, not a validated crop irrigation threshold. Real thresholds need "
        "agronomic evidence, soil type, crop stage and calibrated sensors."
    )
    d1, d2 = st.columns(2)
    pct = d1.slider("Threshold: percentile of training moisture", 5, 50, 25,
                    format="%d%%")
    src = d2.radio("Decide using interval from",
                   ["Gradient Boosting", "XGBoost"], horizontal=True,
                   key="dec_src")
    threshold = float(np.percentile(out["y_train"], pct))
    st.metric("Threshold (soil moisture %)", f"{threshold:.2f}")

    lo_c, hi_c = (("GB lower", "GB upper") if src == "Gradient Boosting"
                  else ("XGB lower", "XGB upper"))
    dec = res.copy()
    dec["Decision"] = [decide(lo, hi, threshold)
                       for lo, hi in zip(dec[lo_c], dec[hi_c])]

    counts = dec["Decision"].value_counts()
    cols = st.columns(3)
    for col, key in zip(cols, ["Alert", "Uncertain", "Wait"]):
        n = int(counts[[i for i in counts.index if i.startswith(key)]].sum())
        col.metric(key, n)

    pick = st.multiselect(
        "Show decisions", sorted(dec["Decision"].unique()),
        default=sorted(dec["Decision"].unique()),
    )
    dist_pick = st.selectbox("District", ["All"] + districts, key="dec_d")
    view = dec[dec["Decision"].isin(pick)]
    if dist_pick != "All":
        view = view[view["District"] == dist_pick]
    st.dataframe(
        view[["Date", "District", "Actual", "XGBoost", lo_c, hi_c, "Decision"]]
        .round(3),
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------
# Downloads tab
# ------------------------------------------------------------
with tab_dl:
    st.write("Save the results for your report.")
    files = {
        "test_predictions_with_decisions.csv": dec,
        "model_metrics.csv": metrics,
        "uncertainty_comparison.csv": unc,
        "feature_importance.csv": out["rf_imp"],
        "xgb_feature_importance.csv": out["xgb_imp"],
        "cleaned_data.csv": df,
    }
    for name, frame in files.items():
        st.download_button(
            f"Download {name}", frame.to_csv(index=False).encode("utf-8"),
            file_name=name, mime="text/csv", key=name,
        )
