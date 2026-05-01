# Purpose: Streamlit app for predicting purchase intent and visualizing model insights.

from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
DATA_PATH  = BASE_DIR / "data" / "online_shoppers_intention.csv"

st.set_page_config(
    page_title="Purchase Intent Predictor",
    page_icon="🛒",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants — must mirror preprocess.py exactly
# ---------------------------------------------------------------------------

# Columns label-encoded during training
_LABEL_ENCODE_COLS = ["Month", "VisitorType", "Weekend"]

# Columns one-hot encoded during training (order matters for column reconstruction)
_OHE_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]

# ---------------------------------------------------------------------------
# Label → numeric mappings for OHE sidebar dropdowns
# Keys are shown to the user; values are passed to the model.
# ---------------------------------------------------------------------------
_OS_MAP = {
    "Windows":      1,
    "Mac / iOS":    2,
    "Android":      3,
    "Linux":        4,
    "ChromeOS":     5,
    "Windows Phone":6,
    "Other":        7,
    "Unknown":      8,
}

_BROWSER_MAP = {
    "Chrome":           1,
    "Firefox":          2,
    "Internet Explorer":3,
    "Safari":           4,
    "Edge":             5,
    "Opera":            6,
    "Mobile Browser":   7,
    "Samsung Browser":  8,
    "UC Browser":       9,
    "Yandex":          10,
    "Other":           11,
    "Unknown":         12,
    "Bot / Crawler":   13,
}

_REGION_MAP = {
    "North America":  1,
    "Europe West":    2,
    "Europe East":    3,
    "South Asia":     4,
    "Southeast Asia": 5,
    "Middle East":    6,
    "South America":  7,
    "Africa":         8,
    "Oceania":        9,
}

_TRAFFIC_MAP = {
    "Direct (typed URL)": 1,
    "Google Search":      2,
    "Social Media":       3,
    "Email Campaign":     4,
    "Display Ad":         5,
    "Referral Link":      6,
    "Paid Search":        7,
    "Affiliate":          8,
    "SMS Campaign":       9,
    "Push Notification": 10,
    "YouTube":           11,
    "Facebook Ad":       12,
    "Instagram Ad":      13,
    "Twitter":           14,
    "LinkedIn":          15,
    "WhatsApp":          16,
    "Newsletter":        17,
    "Retargeting Ad":    18,
    "Influencer Link":   19,
    "Other":             20,
}

# Columns scaled with StandardScaler during training
_SCALE_COLS = [
    "Administrative", "Informational", "ProductRelated",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load ensemble model, scaler, and results dict."""
    ensemble_model = joblib.load(MODELS_DIR / "ensemble_model.pkl")

    # Scaler: try models/ first, then data/
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not scaler_path.exists():
        scaler_path = BASE_DIR / "data" / "scaler.pkl"
    scaler = joblib.load(scaler_path)

    results = joblib.load(MODELS_DIR / "results.pkl")
    return ensemble_model, scaler, results


@st.cache_resource
def load_rf_model():
    """Load tuned RF model, fall back to baseline."""
    for name in ("rf_tuned.pkl", "rf_model.pkl"):
        p = MODELS_DIR / name
        if p.exists():
            return joblib.load(p), name
    raise FileNotFoundError("No RF model found in models/.")


@st.cache_resource
def load_shap_explainer(rf_model):
    import shap
    return shap.TreeExplainer(rf_model)


@st.cache_resource
def load_label_encoders():
    """
    Fit label encoders from the dataset values so encoding matches training.
    Falls back to hard-coded class lists if the CSV is unavailable.
    """
    month_enc   = LabelEncoder()
    visitor_enc = LabelEncoder()
    weekend_enc = LabelEncoder()

    try:
        df = pd.read_csv(DATA_PATH)
        month_enc.fit(df["Month"].astype(str))
        visitor_enc.fit(df["VisitorType"].astype(str))
        weekend_enc.fit(df["Weekend"].astype(str))
    except Exception:
        month_enc.fit(["Apr","Aug","Dec","Feb","Jan","July","June",
                       "Mar","May","Nov","Oct","Sep"])
        visitor_enc.fit(["New_Visitor","Other","Returning_Visitor"])
        weekend_enc.fit(["False","True"])

    return month_enc, visitor_enc, weekend_enc


@st.cache_data
def get_feature_columns() -> list:
    """
    Reconstruct the exact 63-column feature list by replaying the same
    encoding steps as preprocess.py on the real CSV.
    This is the single source of truth — never use model.feature_names_in_.
    """
    df = pd.read_csv(DATA_PATH)

    # Normalise Revenue dtype (not needed for features but keeps df clean)
    if df["Revenue"].dtype == bool:
        df["Revenue"] = df["Revenue"].astype(int)
    elif df["Revenue"].dtype == object:
        df["Revenue"] = df["Revenue"].map(
            {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
        )

    # Fill missing values (same as training)
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=["number"]).columns:
        m = df[col].mode(dropna=True)
        if not m.empty:
            df[col] = df[col].fillna(m.iloc[0])

    # Label encode
    for col in _LABEL_ENCODE_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode
    df = pd.get_dummies(df, columns=_OHE_COLS, dtype=int).astype(float)

    # Return feature columns only (drop target)
    return [c for c in df.columns if c != "Revenue"]


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def safe_encode(enc: LabelEncoder, value) -> int:
    s = str(value)
    if s in set(enc.classes_):
        return int(enc.transform([s])[0])
    return int(enc.transform([str(enc.classes_[0])])[0])


def build_model_input(user_input: dict, feature_columns: list,
                      scaler, encoders) -> pd.DataFrame:
    """
    Reproduce the exact preprocessing pipeline from preprocess.py:
      1. Pass numeric columns through as-is
      2. Label-encode Month, VisitorType, Weekend
      3. One-hot encode OperatingSystems, Browser, Region, TrafficType
         by setting the matching dummy column to 1 (all others stay 0)
      4. Apply StandardScaler to the 7 numeric columns
      5. Return a single-row DataFrame in the exact column order used at training

    Prints feature vector shape to terminal for debugging.
    """
    month_enc, visitor_enc, weekend_enc = encoders

    # Start with every feature column set to 0
    row = {col: 0.0 for col in feature_columns}

    # ── Numeric pass-through columns ─────────────────────────────────────
    numeric_passthrough = [
        "Administrative", "Administrative_Duration",
        "Informational",  "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    ]
    for key in numeric_passthrough:
        if key in row and key in user_input:
            row[key] = float(user_input[key])

    # ── Label-encoded categoricals ────────────────────────────────────────
    if "Month"       in row:
        row["Month"]       = float(safe_encode(month_enc,   user_input["Month"]))
    if "VisitorType" in row:
        row["VisitorType"] = float(safe_encode(visitor_enc, user_input["VisitorType"]))
    if "Weekend"     in row:
        row["Weekend"]     = float(safe_encode(weekend_enc, str(user_input["Weekend"])))

    # ── One-hot encoded categoricals ──────────────────────────────────────
    # pandas get_dummies produces columns like "OperatingSystems_1",
    # "Browser_2", "Region_3", "TrafficType_4" — set the matching one to 1.
    for ohe_col in _OHE_COLS:
        val = user_input.get(ohe_col)
        if val is None:
            continue
        # Values from the CSV are integers; format as int to match column names
        dummy_col = f"{ohe_col}_{int(val)}"
        if dummy_col in row:
            row[dummy_col] = 1.0

    # ── Build DataFrame in exact training column order ────────────────────
    input_df = pd.DataFrame([row], columns=feature_columns).astype(float)

    # ── Scale the same 7 numeric columns scaled during training ──────────
    existing_scale = [c for c in _SCALE_COLS if c in input_df.columns]
    if existing_scale:
        input_df.loc[:, existing_scale] = scaler.transform(input_df[existing_scale])

    # Debug: print shape to terminal so mismatches are immediately visible
    print(f"Feature vector shape: {input_df.shape}")

    return input_df


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def segment_info(proba: float):
    """Return (label, action, st_alert_fn) for a given probability."""
    if proba < 0.3:
        return "🔴 Just Browsing",  "Show discount popup",               st.error
    elif proba < 0.7:
        return "🟡 Interested",     "Show product recommendations",      st.warning
    else:
        return "🟢 Ready to Buy",   "Show urgency / limited-time offer", st.success


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Bootstrap — load artifacts, derive feature columns
# ---------------------------------------------------------------------------

try:
    ensemble_model, scaler, results = load_artifacts()
    encoders = load_label_encoders()
except Exception as err:
    st.error(f"❌ Failed to load model artifacts: {err}")
    st.stop()

try:
    model_feature_columns = get_feature_columns()
except Exception as err:
    st.error(f"❌ Failed to reconstruct feature columns: {err}")
    st.stop()


# RF model (non-fatal — only needed for SHAP tab)
try:
    rf_model, rf_source = load_rf_model()
    rf_available = True
except Exception as _e:
    rf_model, rf_source, rf_available = None, str(_e), False

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🛒 Purchase Intent Predictor")
st.caption("Predict customer purchase intent and explore model insights.")

# ---------------------------------------------------------------------------
# Sidebar — ALL inputs, shared across every tab
# ---------------------------------------------------------------------------

st.sidebar.header("Enter User Behaviour")

# Behaviour metrics
administrative          = st.sidebar.slider("Administrative",           0,    30,    5)
administrative_duration = st.sidebar.slider("Administrative Duration",  0.0, 3000.0, 80.0)
informational           = st.sidebar.slider("Informational",            0,    30,    2)
informational_duration  = st.sidebar.slider("Informational Duration",   0.0, 2500.0, 30.0)
product_related         = st.sidebar.slider("ProductRelated",           0,    50,   10)
product_related_duration= st.sidebar.slider("ProductRelated Duration",  0.0, 9000.0, 500.0)
bounce_rates            = st.sidebar.slider("BounceRates",              0.0,  1.0,   0.2,  step=0.01)
exit_rates              = st.sidebar.slider("ExitRates",                0.0,  1.0,   0.2,  step=0.01)
page_values             = st.sidebar.slider("PageValues",               0.0, 500.0,  50.0)
special_day             = st.sidebar.slider("SpecialDay",               0.0,  1.0,   0.0,  step=0.1)

st.sidebar.markdown("---")

# Categorical — label encoded
month        = st.sidebar.selectbox("Month", [
    "Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"
])
visitor_type = st.sidebar.selectbox("VisitorType", [
    "New_Visitor", "Returning_Visitor", "Other"
])
weekend      = st.sidebar.checkbox("Is it Weekend?")

st.sidebar.markdown("---")

# Categorical — one-hot encoded (label shown to user, number passed to model)
os_label      = st.sidebar.selectbox("Operating System",  list(_OS_MAP.keys()))
browser_label = st.sidebar.selectbox("Browser",           list(_BROWSER_MAP.keys()))
region_label  = st.sidebar.selectbox("Region",            list(_REGION_MAP.keys()))
traffic_label = st.sidebar.selectbox("Traffic Type",      list(_TRAFFIC_MAP.keys()))

operating_systems = _OS_MAP[os_label]
browser           = _BROWSER_MAP[browser_label]
region            = _REGION_MAP[region_label]
traffic_type      = _TRAFFIC_MAP[traffic_label]

# Unified input dict — consumed by all tabs
user_input = {
    "Administrative":           administrative,
    "Administrative_Duration":  administrative_duration,
    "Informational":            informational,
    "Informational_Duration":   informational_duration,
    "ProductRelated":           product_related,
    "ProductRelated_Duration":  product_related_duration,
    "BounceRates":              bounce_rates,
    "ExitRates":                exit_rates,
    "PageValues":               page_values,
    "SpecialDay":               special_day,
    "Month":                    month,
    "VisitorType":              visitor_type,
    "Weekend":                  weekend,
    "OperatingSystems":         operating_systems,
    "Browser":                  browser,
    "Region":                   region,
    "TrafficType":              traffic_type,
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_predict, tab_models, tab_features, tab_shap = st.tabs([
    "Predict", "Model Comparison", "Feature Importance", "Explainability (SHAP)"
])

# ===========================================================================
# TAB 1 — Predict
# ===========================================================================
with tab_predict:

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left column: input summary ────────────────────────────────────────
    with col_left:
        st.subheader("Current Input Summary")

        # Reverse maps — number → label, used only for display
        _os_rev      = {v: k for k, v in _OS_MAP.items()}
        _browser_rev = {v: k for k, v in _BROWSER_MAP.items()}
        _region_rev  = {v: k for k, v in _REGION_MAP.items()}
        _traffic_rev = {v: k for k, v in _TRAFFIC_MAP.items()}

        # Build a display copy — model still receives numbers via user_input
        display_input = {
            **user_input,
            "OperatingSystems": _os_rev.get(user_input["OperatingSystems"], user_input["OperatingSystems"]),
            "Browser":          _browser_rev.get(user_input["Browser"],          user_input["Browser"]),
            "Region":           _region_rev.get(user_input["Region"],            user_input["Region"]),
            "TrafficType":      _traffic_rev.get(user_input["TrafficType"],      user_input["TrafficType"]),
        }

        # Convert all values to str so the Value column is uniform —
        # prevents Arrow serialisation errors from mixed types (str + int + float + bool).
        summary_df = pd.DataFrame({
            "Feature": list(display_input.keys()),
            "Value":   [str(v) for v in display_input.values()],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Right column: prediction output ──────────────────────────────────
    with col_right:
        st.subheader("Prediction")

        if st.button("🔍 Predict", type="primary", use_container_width=True):
            try:
                input_df = build_model_input(
                    user_input, model_feature_columns, scaler, encoders
                )

                # Validate shape before calling the model
                expected_n = len(model_feature_columns)
                actual_n   = input_df.shape[1]
                if actual_n != expected_n:
                    raise ValueError(
                        f"Feature count mismatch: model expects {expected_n} "
                        f"columns but input has {actual_n}."
                    )

                # Run ensemble model prediction
                proba = float(ensemble_model.predict_proba(input_df)[0][1])

                # Persist result so it survives Streamlit reruns
                st.session_state["pred_proba"]    = proba
                st.session_state["pred_input_df"] = input_df

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.pop("pred_proba", None)

        # ── Render results from session state (survives reruns) ───────────
        if "pred_proba" in st.session_state:
            proba = st.session_state["pred_proba"]

            st.markdown("---")

            # 1. Purchase probability as percentage
            st.metric(label="Purchase Probability", value=f"{proba:.0%}")

            # 2. Confidence progress bar
            st.progress(float(proba), text=f"Confidence: {proba:.1%}")

            st.markdown("")

            # 3. Coloured segment badge
            label, action, alert_fn = segment_info(proba)
            st.markdown("**Customer Segment**")
            alert_fn(f"**{label}**")

            # 4. Recommended business action
            st.markdown("**Recommended Action**")
            st.info(f"💡 {action}")

            # 5. Final verdict
            st.markdown("**Final Verdict**")
            if proba >= 0.5:
                st.success("✅ Will Purchase")
            else:
                st.error("❌ Will Not Purchase")

# ===========================================================================
# TAB 2 — Model Comparison  (unchanged)
# ===========================================================================
with tab_models:
    st.subheader("Model Comparison")

    for img_file, caption in [
        ("comparison_chart.png", "Model Performance Comparison"),
        ("roc_curve.png",        "ROC Curve Comparison"),
    ]:
        p = STATIC_DIR / img_file
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)
        else:
            st.warning(f"{img_file} not found in static/.")

    st.subheader("Results Table")
    try:
        results_table = (
            pd.DataFrame(results).T
            .reset_index()
            .rename(columns={
                "index":     "Model",
                "accuracy":  "Accuracy",
                "precision": "Precision",
                "recall":    "Recall",
                "f1":        "F1-Score",
            })
        )
        st.dataframe(results_table, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render results table: {e}")

# ===========================================================================
# TAB 3 — Feature Importance  (unchanged)
# ===========================================================================
with tab_features:
    st.subheader("Feature Importance")

    for img_file, caption in [
        ("feature_importance.png",   "Top 10 Feature Importances"),
        ("segment_distribution.png", "Segment Distribution"),
    ]:
        p = STATIC_DIR / img_file
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)
        else:
            st.warning(f"{img_file} not found in static/.")

# ===========================================================================
# TAB 4 — Explainability (SHAP)  (shap.initjs() removed)
# ===========================================================================
with tab_shap:
    st.subheader("Explainability (SHAP)")
    st.markdown(
        """
        **What is SHAP?**  
        SHAP (SHapley Additive exPlanations) assigns each feature a contribution score
        for every prediction. Grounded in game theory, SHAP values reliably explain
        *why* the model made a decision — both globally across all customers and
        locally for a single prediction.
        """
    )

    if not rf_available:
        st.error(f"Random Forest model could not be loaded: {rf_source}")
    else:
        st.caption(f"Model in use: **{rf_source}**")

        # ── Global SHAP plots (pre-generated by 08_shap_explainability.py) ─
        st.markdown("---")
        st.markdown("### Global Feature Impact")
        col_bar, col_dot = st.columns(2)

        with col_bar:
            p = STATIC_DIR / "shap_summary_bar.png"
            if p.exists():
                st.image(str(p), caption="Mean |SHAP| — Overall Feature Importance",
                         use_container_width=True)
            else:
                st.warning("shap_summary_bar.png not found. "
                           "Run `python notebooks/08_shap_explainability.py`.")

        with col_dot:
            p = STATIC_DIR / "shap_summary_dot.png"
            if p.exists():
                st.image(str(p), caption="Beeswarm — Direction & Magnitude",
                         use_container_width=True)
            else:
                st.warning("shap_summary_dot.png not found. "
                           "Run `python notebooks/08_shap_explainability.py`.")

        # ── Real-time force plot for current sidebar input ────────────────
        st.markdown("---")
        st.markdown("### Why did the model predict this?")
        st.markdown(
            "Adjust the sidebar and click **Explain This Prediction** to see "
            "which features drove the model's decision for the current input."
        )

        if st.button("Explain This Prediction", type="primary"):
            try:
                import shap

                input_df = build_model_input(
                    user_input, model_feature_columns, scaler, encoders
                )

                # Align to RF's own feature order if it has one, else use training order
                if hasattr(rf_model, "feature_names_in_"):
                    input_rf = input_df.reindex(
                        columns=list(rf_model.feature_names_in_), fill_value=0.0
                    )
                else:
                    input_rf = input_df

                explainer   = load_shap_explainer(rf_model)
                shap_values = explainer.shap_values(input_rf)

                # sklearn RF returns list [class0_shap, class1_shap]
                if isinstance(shap_values, list):
                    sv1 = shap_values[1][0]
                else:
                    sv1 = shap_values[0]

                base_val = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )

                # Force plot — matplotlib mode, no JS / IPython required
                force_fig = shap.force_plot(
                    base_val, sv1, input_rf.iloc[0],
                    matplotlib=True, show=False,
                )
                st.image(
                    fig_to_bytes(force_fig),
                    caption="Force Plot — red pushes toward Buy, blue pushes away",
                    use_container_width=True,
                )
                plt.close("all")

                # Top-3 text explanation
                feat_names  = list(input_rf.columns)
                shap_series = pd.Series(sv1, index=feat_names)
                top3_pos    = shap_series.nlargest(3)
                top3_neg    = shap_series.nsmallest(3)

                rf_proba  = float(rf_model.predict_proba(input_rf)[0][1])
                direction = "Buy 🟢" if rf_proba >= 0.5 else "Not Buy 🔴"
                st.markdown(
                    f"**RF model leans toward: {direction}** "
                    f"(probability {rf_proba:.1%})"
                )

                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("**Top 3 pushing toward Buy ↑**")
                    for feat, val in top3_pos.items():
                        raw = user_input.get(feat, input_rf.iloc[0][feat])
                        st.markdown(f"- **{feat}** = `{raw}` *(SHAP: +{val:.4f})*")

                with col_neg:
                    st.markdown("**Top 3 pushing toward Not Buy ↓**")
                    for feat, val in top3_neg.items():
                        raw = user_input.get(feat, input_rf.iloc[0][feat])
                        st.markdown(f"- **{feat}** = `{raw}` *(SHAP: {val:.4f})*")

            except ImportError:
                st.error("SHAP not installed. Run `pip install shap` and restart.")
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")
