# ============================================================
# 08_shap_explainability.py
# Purpose: Explain the Random Forest model predictions using
#          SHAP (SHapley Additive exPlanations).
#
# ---------------------------------------------------------------
# WHAT IS SHAP AND WHY DOES IT MATTER FOR MODEL TRUST?
# ---------------------------------------------------------------
# SHAP is a game-theory-based framework that assigns each feature
# a "Shapley value" — the average marginal contribution of that
# feature across all possible subsets of features.
#
# Why it matters:
#   1. TRANSPARENCY  — Shows exactly which features pushed a
#      prediction higher or lower, making black-box models
#      interpretable to stakeholders and regulators.
#
#   2. CONSISTENCY   — Unlike permutation importance, SHAP values
#      satisfy mathematical fairness axioms (efficiency, symmetry,
#      dummy, additivity), so rankings are reliable and stable.
#
#   3. LOCAL + GLOBAL — SHAP explains individual predictions
#      (force plots) AND global model behaviour (summary plots),
#      giving both micro and macro insight.
#
#   4. TRUST & DEBUGGING — Reveals if the model is relying on
#      spurious correlations or the right business signals,
#      enabling data scientists to catch and fix bias early.
#
#   5. REGULATORY COMPLIANCE — Many industries (finance, health)
#      require explainable AI; SHAP provides auditable reasoning.
#
# Install: pip install shap
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import joblib
import shap

# ------------------------------------------------------------------
# 0. Path setup
# ------------------------------------------------------------------
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(NOTEBOOK_DIR)
DATA_DIR     = os.path.join(PROJECT_DIR, "data")
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
STATIC_DIR   = os.path.join(PROJECT_DIR, "static")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.makedirs(STATIC_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load X_train, X_test from data/
# ------------------------------------------------------------------
print("=" * 60)
print("Step 1 — Loading preprocessed data splits")
print("=" * 60)

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
X_TEST_PATH  = os.path.join(DATA_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
Y_TEST_PATH  = os.path.join(DATA_DIR, "y_test.npy")

npy_exist = all(
    os.path.exists(p)
    for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
)

if npy_exist:
    print("  Loading from .npy files...")
    X_train_raw = np.load(X_TRAIN_PATH)
    X_test_raw  = np.load(X_TEST_PATH)
    y_train     = np.load(Y_TRAIN_PATH)
    y_test      = np.load(Y_TEST_PATH)
    print(f"  X_train: {X_train_raw.shape}  |  X_test: {X_test_raw.shape}")
else:
    print("  .npy files not found — regenerating from CSV via preprocess.py ...")
    from preprocess import load_and_explore, preprocess_data

    csv_path = os.path.join(DATA_DIR, "online_shoppers_intention.csv")
    df = load_and_explore(dataset_path=csv_path, output_dir=STATIC_DIR)
    X_train_df, X_test_df, y_train, y_test = preprocess_data(df)

    X_train_raw = X_train_df.values
    X_test_raw  = X_test_df.values

    np.save(X_TRAIN_PATH, X_train_raw)
    np.save(X_TEST_PATH,  X_test_raw)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH,  y_test)
    print(f"  Saved .npy splits to {DATA_DIR}")

# ------------------------------------------------------------------
# 2. Load the Random Forest model
#    Prefer rf_tuned.pkl (from hyperparameter tuning) if available,
#    otherwise fall back to rf_model.pkl.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2 — Loading Random Forest model")
print("=" * 60)

rf_tuned_path    = os.path.join(MODELS_DIR, "rf_tuned.pkl")
rf_baseline_path = os.path.join(MODELS_DIR, "rf_model.pkl")

if os.path.exists(rf_tuned_path):
    rf_model = joblib.load(rf_tuned_path)
    print(f"  Loaded tuned model   → {rf_tuned_path}")
elif os.path.exists(rf_baseline_path):
    rf_model = joblib.load(rf_baseline_path)
    print(f"  Loaded baseline model → {rf_baseline_path}")
else:
    raise FileNotFoundError(
        "No Random Forest model found. Expected rf_tuned.pkl or rf_model.pkl in models/."
    )

# ------------------------------------------------------------------
# 3. Reconstruct feature names
#    Priority order:
#      a) feature_names_in_  — set by sklearn when fitted on a DataFrame
#      b) Replay the same encoding logic on the CSV to get column order
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3 — Reconstructing feature names")
print("=" * 60)

def reconstruct_feature_names(csv_path: str) -> list:
    """
    Replay the same encoding steps as preprocess.py to derive the
    exact post-encoding column list in the correct order.
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(csv_path)

    # Normalise Revenue dtype (not needed for feature names but keeps df clean)
    if df["Revenue"].dtype == bool:
        df["Revenue"] = df["Revenue"].astype(int)
    elif df["Revenue"].dtype == object:
        df["Revenue"] = df["Revenue"].map(
            {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
        )

    # Fill missing values
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(exclude=["number"]).columns:
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

    # Label encode
    for col in ["Month", "VisitorType", "Weekend"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode
    ohe_cols = [c for c in ["OperatingSystems", "Browser", "Region", "TrafficType"] if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, dtype=int).astype(float)

    # Drop target
    feature_cols = [c for c in df.columns if c != "Revenue"]
    return feature_cols


# Try model's own feature_names_in_ first
if hasattr(rf_model, "feature_names_in_"):
    feature_names = list(rf_model.feature_names_in_)
    print(f"  Source: model.feature_names_in_  ({len(feature_names)} features)")
else:
    csv_path = os.path.join(DATA_DIR, "online_shoppers_intention.csv")
    if os.path.exists(csv_path):
        feature_names = reconstruct_feature_names(csv_path)
        print(f"  Source: replayed encoding on CSV  ({len(feature_names)} features)")
    else:
        # Last resort: generic names
        n_features = X_test_raw.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]
        print(f"  Source: generic names  ({n_features} features)")

# Validate length matches data
assert len(feature_names) == X_test_raw.shape[1], (
    f"Feature name count ({len(feature_names)}) does not match "
    f"X_test columns ({X_test_raw.shape[1]}). "
    "Ensure preprocessing is consistent."
)

# Wrap arrays in DataFrames so SHAP plots show real column names
X_train = pd.DataFrame(X_train_raw, columns=feature_names)
X_test  = pd.DataFrame(X_test_raw,  columns=feature_names)

print(f"  Feature names validated — {len(feature_names)} columns confirmed.")

# ------------------------------------------------------------------
# 4. Compute SHAP values using TreeExplainer
#    TreeExplainer is optimised for tree-based models (RF, XGBoost,
#    LightGBM) and runs in O(TLD) time — much faster than KernelSHAP.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4 — Computing SHAP values (TreeExplainer)")
print("=" * 60)

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# For binary classifiers sklearn returns a list [class0_shap, class1_shap].
# We use class 1 (purchase = True) for all explanations.
if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]   # shape: (n_samples, n_features)
else:
    shap_vals_class1 = shap_values       # already 2-D for some versions

print(f"  SHAP values shape: {shap_vals_class1.shape}")
print(f"  Expected output (base value): {explainer.expected_value[1]:.4f}"
      if isinstance(explainer.expected_value, (list, np.ndarray))
      else f"  Expected output (base value): {explainer.expected_value:.4f}")

# ------------------------------------------------------------------
# 5a. shap_summary_bar.png — mean |SHAP| bar chart (global importance)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5a — Saving shap_summary_bar.png")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_vals_class1,
    X_test,
    plot_type="bar",
    show=False,
    max_display=20,
)
plt.title("SHAP Feature Importance — Mean |SHAP Value| (Class: Purchase)", pad=12)
plt.tight_layout()
bar_path = os.path.join(STATIC_DIR, "shap_summary_bar.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {bar_path}")

# ------------------------------------------------------------------
# 5b. shap_summary_dot.png — beeswarm plot (direction + magnitude)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5b — Saving shap_summary_dot.png")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_vals_class1,
    X_test,
    plot_type="dot",
    show=False,
    max_display=20,
)
plt.title(
    "SHAP Beeswarm — Feature Impact Direction & Magnitude (Class: Purchase)",
    pad=12,
)
plt.tight_layout()
dot_path = os.path.join(STATIC_DIR, "shap_summary_dot.png")
plt.savefig(dot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {dot_path}")

# ------------------------------------------------------------------
# 5c. shap_force_plot.png — force plot for the first test sample
#     Explains a single prediction: which features pushed the
#     model output above or below the base (expected) value.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5c — Saving shap_force_plot.png")
print("=" * 60)

# Resolve base value for class 1
base_value = (
    explainer.expected_value[1]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)

# matplotlib-based force plot (works without a browser / JS)
# Note: shap.initjs() is only for Jupyter notebooks — omitted here.
force_fig = shap.force_plot(
    base_value,
    shap_vals_class1[0],        # SHAP values for sample 0
    X_test.iloc[0],             # feature values for sample 0
    matplotlib=True,
    show=False,
)
force_path = os.path.join(STATIC_DIR, "shap_force_plot.png")
plt.savefig(force_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {force_path}")

# ------------------------------------------------------------------
# 6. Print top 5 most impactful features with mean |SHAP| values
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6 — Top 5 Most Impactful Features")
print("=" * 60)

mean_abs_shap = pd.Series(
    np.abs(shap_vals_class1).mean(axis=0),
    index=feature_names,
).sort_values(ascending=False)

print(f"\n  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
print("  " + "-" * 55)
for rank, (feat, val) in enumerate(mean_abs_shap.head(5).items(), start=1):
    print(f"  {rank:<6} {feat:<35} {val:>12.5f}")

# ------------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("SHAP Explainability — Complete")
print("=" * 60)
print(f"  Plots saved to : {STATIC_DIR}")
print(f"    • shap_summary_bar.png  — global feature importance (bar)")
print(f"    • shap_summary_dot.png  — beeswarm (direction + magnitude)")
print(f"    • shap_force_plot.png   — single-prediction force plot")
print(f"  Model explained : {rf_model.__class__.__name__}")
print(f"  Test samples    : {X_test.shape[0]}")
print(f"  Features        : {X_test.shape[1]}")
print("=" * 60)
