"""
CLP/USD Multi-Horizon Scenario Analysis App
============================================
Four independently trained models: 3M / 6M / 9M / 12M.
Each horizon has its own tab with independent inputs.

Features per model (as of last training):
  3M : Precio_Cobre, Dolar_index, dif_CDS, USA CPI YoY, Chile CPI ex-Vol, USA Mortgage Delinquencies, VIX
  6M : Precio_Cobre, Dolar_index, dif_CDS, USA Core CPI YoY, Chile Bond 2Y, USA Inf Swap 2Y, VIX
  9M : Precio_Cobre, Dolar_index, VIX, dif_Swap_2Y, USA Core CPI YoY
  12M: Precio_Cobre, Dolar_index, dif_CDS, Chile CPI ex-Vol, USA Inf Swap 2Y, VIX

Run with:
    streamlit run "C:/Users/fsalgado/Python Data/app_clpusd_multi.py"
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# ══════════════════════════════════════════════════════════════════════════════
# Custom transformer classes — must be here at module level for pickle to work
# ══════════════════════════════════════════════════════════════════════════════

class LogReturnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, periods=1):
        self.periods = periods

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.last_values_ = X_arr[-self.periods:].copy()
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        n_rows, _ = X_arr.shape
        log_ret = np.full_like(X_arr, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret[self.periods:, :] = np.log(
                X_arr[self.periods:, :] / X_arr[:-self.periods, :]
            )
        if hasattr(self, "last_values_") and n_rows <= self.periods * 3:
            X_with_history = np.vstack([self.last_values_, X_arr])
            with np.errstate(divide="ignore", invalid="ignore"):
                for i in range(min(self.periods, n_rows)):
                    log_ret[i, :] = np.log(X_arr[i, :] / X_with_history[i, :])
        return log_ret

    def get_feature_names_out(self, feature_names_in=None):
        return feature_names_in


class DifferencesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, periods=1):
        self.periods = periods

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.last_values_ = X_arr[-self.periods:].copy()
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        n_rows, _ = X_arr.shape
        diff = np.full_like(X_arr, np.nan, dtype=float)
        diff[self.periods:, :] = X_arr[self.periods:, :] - X_arr[:-self.periods, :]
        if hasattr(self, "last_values_") and n_rows <= self.periods * 3:
            X_with_history = np.vstack([self.last_values_, X_arr])
            for i in range(min(self.periods, n_rows)):
                diff[i, :] = X_arr[i, :] - X_with_history[i, :]
        return diff

    def get_feature_names_out(self, feature_names_in=None):
        return feature_names_in


class ClipInfTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[np.isinf(X)] = np.nan
        return X


class ForwardFillTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.ffill()
        return pd.DataFrame(X).ffill().values


class TargetPreprocessorDiff(BaseEstimator, TransformerMixin):
    def __init__(self, periods=1):
        self.periods = periods
        self.first_valid_value_ = None
        self.first_prices_ = None
        self.last_prices_ = None

    def fit(self, y, X=None):
        y_clean = pd.Series(y).ffill().bfill()
        self.first_valid_value_ = y_clean.iloc[0]
        self.first_prices_ = y_clean.iloc[: self.periods].values
        self.last_prices_ = y_clean.iloc[-self.periods:].values
        return self

    def transform(self, y, use_history=False):
        y_clean = pd.Series(y).ffill().bfill()
        if use_history and self.last_prices_ is not None:
            extended = pd.concat(
                [pd.Series(self.last_prices_), y_clean], ignore_index=True
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret = np.log(extended / extended.shift(self.periods))
            return log_ret.iloc[self.periods:].reset_index(drop=True).values.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.log(y_clean / y_clean.shift(self.periods))
        return log_ret.values.ravel()

    def fit_transform(self, y, X=None):
        self.fit(y, X)
        return self.transform(y, use_history=False)

    def inverse_transform(self, y_transformed):
        return np.array(y_transformed).ravel()


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = "."
HORIZONS = ["3M", "6M", "9M", "12M"]
HORIZON_DAYS = {"3M": 63, "6M": 126, "9M": 186, "12M": 252}

# Human-readable labels and slider ranges for known features.
# Add more entries here as needed.
FEATURE_CFG = {
    "Precio_Cobre":                             ("Copper (USc/lb)",           200.0,  700.0,  1.0),
    "Dolar_index":                              ("DXY – USD Index",            80.0,  125.0,  0.1),
    "VIX":                                      ("VIX",                         8.0,   85.0,  0.5),
    "Chile_CDS":                                ("Chile CDS 5Y (bps)",         30.0,  350.0,  1.0),
    "USA_CDS":                                  ("USA CDS 5Y (bps)",           10.0,  120.0,  1.0),
    "Chile_Swap_1Y":                            ("Chile Swap 1Y (%)",           0.5,   12.0,  0.05),
    "Chile_Swap_2Y":                            ("Chile Swap 2Y (%)",           0.5,   12.0,  0.05),
    "USA_Swap_1Y":                              ("USA Swap 1Y (%)",             0.5,   12.0,  0.05),
    "USA_Swap_2Y":                              ("USA Swap 2Y (%)",             0.5,   12.0,  0.05),
    "Chile_Bond_1Y":                            ("Chile Bond 1Y (%)",           0.5,   12.0,  0.05),
    "Chile_Bond_2Y":                            ("Chile Bond 2Y (%)",           0.5,   12.0,  0.05),
    "Chile_Bond_5Y":                            ("Chile Bond 5Y (%)",           0.5,   12.0,  0.05),
    "Chile_Bond_10Y":                           ("Chile Bond 10Y (%)",          0.5,   12.0,  0.05),
    "USA_Bond_1Y":                              ("USA Bond 1Y (%)",             0.5,   12.0,  0.05),
    "USA_Bond_2Y":                              ("USA Bond 2Y (%)",             0.5,   12.0,  0.05),
    "USA_Bond_5Y":                              ("USA Bond 5Y (%)",             0.5,   12.0,  0.05),
    "USA_Bond_10Y":                             ("USA Bond 10Y (%)",            0.5,   12.0,  0.05),
    "dif_Swap_1Y":                              ("Δ Swap 1Y (CL-US, %)",       -5.0,    5.0,  0.05),
    "dif_Swap_2Y":                              ("Δ Swap 2Y (CL-US, %)",       -5.0,    5.0,  0.05),
    "dif_Bond_1Y":                              ("Δ Bond 1Y (CL-US, %)",       -5.0,    5.0,  0.05),
    "dif_Bond_2Y":                              ("Δ Bond 2Y (CL-US, %)",       -5.0,    5.0,  0.05),
    "dif_Bond_5Y":                              ("Δ Bond 5Y (CL-US, %)",       -5.0,    5.0,  0.05),
    "dif_Bond_10Y":                             ("Δ Bond 10Y (CL-US, %)",      -5.0,    5.0,  0.05),
    "dif_CDS":                                  ("Δ CDS (CL-US, bps)",        -200.0,  200.0,  1.0),
    "Chile_Incertidumbre_Local":                ("Chile Local Uncertainty",      0.0,  100.0,  0.5),
    "Chile_IPC_Sin_Volatiles (yoy %)":          ("Chile CPI ex-Volatiles YoY",  0.0,   15.0,  0.1),
    "USA_Unemployment Rate (%)":                ("USA Unemployment (%)",         3.0,   15.0,  0.1),
    "USA_Consumer Price Index (yoy %)":         ("USA CPI YoY (%)",             -2.0,   15.0,  0.1),
    "USA_CPI ex-Food & Energy (yoy %)":         ("USA Core CPI YoY (%)",        -2.0,   12.0,  0.1),
    "USA_PCE Price Index (yoy %)":              ("USA PCE YoY (%)",             -2.0,   12.0,  0.1),
    "USA_Core PCE Index (yoy %)":               ("USA Core PCE YoY (%)",        -2.0,   12.0,  0.1),
    "USA_Mortgage Delinquencies (% of total loans, sa)":  ("USA Mortgage Delinquencies (%)",   1.0,   12.0,  0.1),
    "USA_Swap_Inf_2Y":                          ("USA Inflation Swap 2Y (%)",    0.5,   10.0,  0.05),
}


# ══════════════════════════════════════════════════════════════════════════════
# Load artifacts
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_all_artifacts():
    arts = {}
    missing = []
    for h in HORIZONS:
        try:
            arts[h] = joblib.load(f"{DATA_DIR}/model_CLPUSD_{h}.pkl")
        except FileNotFoundError:
            missing.append(h)
    return arts, missing


# ══════════════════════════════════════════════════════════════════════════════
# Prediction logic
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(art: dict, user_inputs: dict) -> dict:
    """Run one horizon's prediction given user inputs for that horizon."""
    model           = art["model"]
    feat_pipeline   = art["features_pipeline"]
    target_prep     = art["target_preprocessor"]
    sel_feats       = art["selected_features"]
    feats_no_suffix = art["features_no_suffix"]
    history_data    = art["history_data"].copy()
    fh              = art["horizon_days"]
    last_price      = art["last_known_price"]

    # Build future row
    future_row = history_data.iloc[[-1]].copy()
    future_row.index = [history_data.index[-1] + pd.tseries.offsets.BDay(fh)]
    for feat in feats_no_suffix:
        if feat in future_row.columns:
            future_row[feat] = user_inputs.get(feat, history_data[feat].iloc[-1])

    # Transform
    extended    = pd.concat([history_data, future_row])
    X_ext       = feat_pipeline.transform(extended)
    X_last      = X_ext[[-1]]

    # Rebuild feature names from the ColumnTransformer
    feature_names = []
    preprocessing = feat_pipeline.named_steps["preprocessing"]
    for name, _, columns in preprocessing.transformers_:
        if name == "remainder":
            continue
        if isinstance(columns, list):
            suffix = "_log_ret" if "log_ret" in name else "_diff"
            feature_names.extend([f"{c}{suffix}" for c in columns])

    row_df      = pd.DataFrame(X_last, columns=feature_names)
    row_prep    = row_df[sel_feats].values

    pred_diff   = model.predict(row_prep)[0]
    pred_diff   = target_prep.inverse_transform([pred_diff])[0]
    proj_price  = last_price + pred_diff
    pct_change  = (proj_price / last_price - 1) * 100

    return {
        "projected_price": proj_price,
        "pred_diff":        pred_diff,
        "pct_change":       pct_change,
        "last_price":       last_price,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helper: render input widgets for one horizon
# ══════════════════════════════════════════════════════════════════════════════

def render_inputs(h: str, art: dict, key_prefix: str) -> dict:
    """Render number_input widgets for each feature of a given horizon."""
    feats_no_sfx = art["features_no_suffix"]
    last_vals    = art["last_feature_values"]
    inputs       = {}

    for feat in feats_no_sfx:
        default  = float(last_vals.get(feat, 0.0))
        label, mn, mx, step = FEATURE_CFG.get(feat, (feat, default * 0.5, default * 2.0, abs(default) * 0.01 or 0.01))

        # Ensure default is within bounds
        mn  = min(mn, default - abs(step))
        mx  = max(mx, default + abs(step))

        inputs[feat] = st.number_input(
            label=f"{label}",
            min_value=float(mn),
            max_value=float(mx),
            value=round(default, 4),
            step=float(step),
            format="%.4f",
            help=f"Last observed: {default:.4g}",
            key=f"{key_prefix}_{feat}",
        )

    return inputs


# ══════════════════════════════════════════════════════════════════════════════
# App layout
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CLP/USD – Multi-Horizon Forecast",
    page_icon="💱",
    layout="wide",
)

st.title("💱 CLP/USD — Multi-Horizon Scenario Analysis")
st.markdown(
    "Set projected macro values **independently for each term** and "
    "compare the expected exchange rate evolution across 3M / 6M / 9M / 12M."
)
st.divider()

# ── Load ──────────────────────────────────────────────────────────────────────
artifacts, missing = load_all_artifacts()

if missing:
    st.warning(
        f"⚠️ Missing model files for: **{', '.join(missing)}**. "
        f"Train and save those horizons first."
    )

available = [h for h in HORIZONS if h in artifacts]
if not available:
    st.error("No model artifacts found. Check DATA_DIR path.")
    st.stop()

last_price = artifacts[available[0]]["last_known_price"]
last_date  = artifacts[available[0]].get("last_date", None)

col_px, col_dt = st.columns(2)
col_px.metric("Last known CLP/USD", f"{last_price:,.2f}")
if last_date is not None:
    col_dt.metric("Data as of", pd.Timestamp(last_date).strftime("%d %b %Y"))
st.divider()

# ── One tab per horizon for inputs ────────────────────────────────────────────
st.subheader("📊 Projected macro variables — by horizon")
st.caption(
    "Each model may use different features. "
    "Variables shared across terms appear in multiple tabs — "
    "set them independently for each horizon."
)

tabs         = st.tabs([f"📅 {h}" for h in available])
all_inputs   = {}   # {horizon: {feature: value}}

for tab, h in zip(tabs, available):
    art = artifacts[h]
    with tab:
        col_info, _ = st.columns([3, 1])
        with col_info:
            feats_used = art["features_no_suffix"]
            st.caption(
                f"**{h} model** uses {len(feats_used)} features: "
                + " · ".join(f"`{f}`" for f in feats_used)
            )

        cols = st.columns(2)
        feats_no_sfx = art["features_no_suffix"]
        last_vals    = art["last_feature_values"]

        inputs = {}
        for i, feat in enumerate(feats_no_sfx):
            default  = float(last_vals.get(feat, 0.0))
            label, mn, mx, step = FEATURE_CFG.get(
                feat, (feat, default * 0.5, default * 2.0, abs(default) * 0.01 or 0.01)
            )
            mn  = min(mn, default - abs(step))
            mx  = max(mx, default + abs(step))

            with cols[i % 2]:
                inputs[feat] = st.number_input(
                    label=label,
                    min_value=float(mn),
                    max_value=float(mx),
                    value=round(default, 4),
                    step=float(step),
                    format="%.4f",
                    help=f"Last observed: {default:.4g}",
                    key=f"{h}_{feat}",
                )

        all_inputs[h] = inputs

st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
run = st.button("🔮  Run all scenarios", type="primary", use_container_width=True)

st.divider()

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    results = {}
    errors  = {}

    with st.spinner("Running models…"):
        for h in available:
            try:
                results[h] = run_scenario(artifacts[h], all_inputs[h])
            except Exception as e:
                errors[h] = str(e)

    if errors:
        for h, err in errors.items():
            st.error(f"❌ Error in {h} model: {err}")

    if results:
        # ── Chart ─────────────────────────────────────────────────────────────
        labels  = ["Now"] + list(results.keys())
        x_days  = [0]    + [HORIZON_DAYS[h] for h in results]
        prices  = [last_price] + [r["projected_price"] for r in results.values()]
        pct_chg = [0.0]  + [r["pct_change"] for r in results.values()]

        marker_colors = ["#636EFA"] + [
            "#00CC96" if r["projected_price"] <= last_price else "#EF553B"
            for r in results.values()
        ]

        fig = go.Figure()
        fig.add_hline(
            y=last_price, line_dash="dot", line_color="#888",
            annotation_text=f"Today: {last_price:.2f}",
            annotation_position="bottom left",
        )
        fig.add_trace(go.Scatter(
            x=x_days,
            y=prices,
            mode="lines+markers+text",
            text=[f"{p:,.1f}" for p in prices],
            textposition="top center",
            textfont=dict(size=13),
            line=dict(width=3, color="#636EFA"),
            marker=dict(size=16, color=marker_colors, line=dict(width=2, color="white")),
            hovertemplate="<b>%{customdata}</b><br>CLP/USD: %{y:,.2f}<extra></extra>",
            customdata=labels,
        ))
        fig.update_layout(
            xaxis=dict(
                tickvals=x_days,
                ticktext=labels,
                title="Horizon",
                zeroline=False,
            ),
            yaxis_title="CLP/USD",
            height=420,
            margin=dict(t=30, b=10),
        )

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            st.subheader("📈 Projected CLP/USD")
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.subheader("📋 Horizon Summary")
            summary_rows = []
            for label, price, pct in zip(labels, prices, pct_chg):
                diff = price - last_price
                summary_rows.append({
                    "Horizon":    label,
                    "CLP/USD":    f"{price:,.2f}",
                    "Δ vs Today": f"{diff:+.2f}",
                    "%":          f"{pct:+.2f}%",
                })
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, hide_index=True, use_container_width=True)

            st.divider()
            # Metric cards
            for h, r in results.items():
                st.metric(
                    label=f"{h} Forecast",
                    value=f"{r['projected_price']:,.2f} CLP/USD",
                    delta=f"{r['pct_change']:+.2f}%",
                    delta_color="inverse",   # green = CLP appreciates (price ↓)
                )

        # ── Per-horizon feature breakdown ──────────────────────────────────────
        st.divider()
        st.subheader("🔍 Input detail by horizon")

        detail_tabs = st.tabs([f"📅 {h}" for h in results])
        for d_tab, (h, r) in zip(detail_tabs, results.items()):
            art  = artifacts[h]
            with d_tab:
                feat_rows = []
                for feat in art["features_no_suffix"]:
                    label = FEATURE_CFG.get(feat, (feat,))[0]
                    last  = art["last_feature_values"].get(feat, float("nan"))
                    proj  = all_inputs[h].get(feat, last)
                    feat_rows.append({
                        "Feature":   label,
                        "Last obs.": f"{last:.4g}",
                        "Projected": f"{proj:.4g}",
                        "Δ":         f"{proj - last:+.4g}",
                    })
                st.dataframe(
                    pd.DataFrame(feat_rows),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    f"Projected CLP/USD: **{r['projected_price']:,.2f}**  "
                    f"({r['pct_change']:+.2f}% vs today)"
                )
