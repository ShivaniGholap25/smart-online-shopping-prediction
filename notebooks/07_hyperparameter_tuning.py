# ============================================================
# 07_hyperparameter_tuning.py
# Purpose: Hyperparameter tuning for Random Forest and ANN
#          using RandomizedSearchCV, with comparison charts.
# ============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------------
# 0. Path setup — make sure project root is importable
# ------------------------------------------------------------------
# This file lives in project/notebooks/, so we step up one level
# to reach project/ where preprocess.py and data/ live.
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(NOTEBOOK_DIR)
DATA_DIR     = os.path.join(PROJECT_DIR, "data")
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
STATIC_DIR   = os.path.join(PROJECT_DIR, "static")

# Add project root to sys.path so we can import preprocess.py
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load or generate X_train, X_test, y_train, y_test
# ------------------------------------------------------------------
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
X_TEST_PATH  = os.path.join(DATA_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
Y_TEST_PATH  = os.path.join(DATA_DIR, "y_test.npy")

npy_files_exist = all(
    os.path.exists(p)
    for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
)

if npy_files_exist:
    print("Loading preprocessed splits from .npy files...")
    X_train = np.load(X_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test  = np.load(Y_TEST_PATH)
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
else:
    print(".npy files not found — generating splits from CSV via preprocess.py ...")
    from preprocess import load_and_explore, preprocess_data

    csv_path = os.path.join(DATA_DIR, "online_shoppers_intention.csv")
    df = load_and_explore(
        dataset_path=csv_path,
        output_dir=STATIC_DIR,
    )
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Save for future runs
    np.save(X_TRAIN_PATH, X_train)
    np.save(X_TEST_PATH,  X_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH,  y_test)
    print(f"  Saved splits to {DATA_DIR}")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

# ------------------------------------------------------------------
# 2. Load untuned baseline models (for F1 comparison later)
# ------------------------------------------------------------------
print("\nLoading baseline (untuned) models...")

rf_baseline  = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
ann_baseline = joblib.load(os.path.join(MODELS_DIR, "ann_model.pkl"))

rf_baseline_f1  = f1_score(y_test, rf_baseline.predict(X_test),  average="weighted")
ann_baseline_f1 = f1_score(y_test, ann_baseline.predict(X_test), average="weighted")

print(f"  Baseline RF  F1 (weighted): {rf_baseline_f1:.4f}")
print(f"  Baseline ANN F1 (weighted): {ann_baseline_f1:.4f}")

# ------------------------------------------------------------------
# 3. RandomizedSearchCV — Random Forest
# ------------------------------------------------------------------
print("\n" + "="*60)
print("RandomizedSearchCV — Random Forest")
print("="*60)

rf_param_dist = {
    "n_estimators":     [50, 100, 200, 300, 500],
    "max_depth":        [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 15, 20],
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=5,
    scoring="f1",          # binary F1 for the positive class
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

rf_search.fit(X_train, y_train)

print(f"\nBest params  (RF): {rf_search.best_params_}")
print(f"Best CV F1   (RF): {rf_search.best_score_:.4f}")

# ------------------------------------------------------------------
# 4. RandomizedSearchCV — ANN (MLPClassifier)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("RandomizedSearchCV — ANN (MLPClassifier)")
print("="*60)

ann_param_dist = {
    "hidden_layer_sizes": [
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128),
        (256, 128, 64),
    ],
    "alpha":              [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init": [0.0001, 0.001, 0.005, 0.01],
}

ann_search = RandomizedSearchCV(
    estimator=MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
    ),
    param_distributions=ann_param_dist,
    n_iter=20,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

ann_search.fit(X_train, y_train)

print(f"\nBest params  (ANN): {ann_search.best_params_}")
print(f"Best CV F1   (ANN): {ann_search.best_score_:.4f}")

# ------------------------------------------------------------------
# 5. Retrain tuned models on full training set with best params
# ------------------------------------------------------------------
print("\n" + "="*60)
print("Retraining tuned models on full training set...")
print("="*60)

rf_tuned = RandomForestClassifier(
    **rf_search.best_params_,
    class_weight="balanced",
    random_state=42,
)
rf_tuned.fit(X_train, y_train)

ann_tuned = MLPClassifier(
    **ann_search.best_params_,
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
)
ann_tuned.fit(X_train, y_train)

print("  Tuned models retrained successfully.")

# ------------------------------------------------------------------
# 6. Evaluate tuned models
# ------------------------------------------------------------------
def evaluate(model, X, y, label):
    """Print and return Accuracy, Precision, Recall, F1 for a model."""
    y_pred = model.predict(X)
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y, y_pred, average="weighted", zero_division=0)
    print(f"\n  [{label}]")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    return acc, prec, rec, f1

print("\n" + "="*60)
print("Evaluation — Tuned Models on Test Set")
print("="*60)

rf_acc,  rf_prec,  rf_rec,  rf_f1  = evaluate(rf_tuned,  X_test, y_test, "RF  Tuned")
ann_acc, ann_prec, ann_rec, ann_f1 = evaluate(ann_tuned, X_test, y_test, "ANN Tuned")

# ------------------------------------------------------------------
# 7. Side-by-side bar chart: tuned vs untuned F1
# ------------------------------------------------------------------
print("\nGenerating comparison chart...")

models      = ["Random Forest", "ANN"]
f1_untuned  = [rf_baseline_f1,  ann_baseline_f1]
f1_tuned    = [rf_f1,           ann_f1]

x      = np.arange(len(models))
width  = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars_untuned = ax.bar(x - width / 2, f1_untuned, width, label="Untuned", color="steelblue")
bars_tuned   = ax.bar(x + width / 2, f1_tuned,   width, label="Tuned",   color="darkorange")

# Annotate bar heights
for bar in bars_untuned:
    ax.annotate(
        f"{bar.get_height():.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars_tuned:
    ax.annotate(
        f"{bar.get_height():.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xlabel("Model")
ax.set_ylabel("F1-Score (weighted)")
ax.set_title("Tuned vs Untuned F1-Score Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.05)
ax.legend(title="Version")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
chart_path = os.path.join(STATIC_DIR, "tuned_vs_untuned_f1.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"  Chart saved → {chart_path}")

# ------------------------------------------------------------------
# 8. Save tuned models
# ------------------------------------------------------------------
rf_tuned_path  = os.path.join(MODELS_DIR, "rf_tuned.pkl")
ann_tuned_path = os.path.join(MODELS_DIR, "ann_tuned.pkl")

joblib.dump(rf_tuned,  rf_tuned_path)
joblib.dump(ann_tuned, ann_tuned_path)

print(f"\n  Saved → {rf_tuned_path}")
print(f"  Saved → {ann_tuned_path}")

# ------------------------------------------------------------------
# 9. Final summary
# ------------------------------------------------------------------
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'Model':<20} {'Untuned F1':>12} {'Tuned F1':>10} {'Delta':>8}")
print("-"*55)
print(f"{'Random Forest':<20} {rf_baseline_f1:>12.4f} {rf_f1:>10.4f} {rf_f1 - rf_baseline_f1:>+8.4f}")
print(f"{'ANN':<20} {ann_baseline_f1:>12.4f} {ann_f1:>10.4f} {ann_f1 - ann_baseline_f1:>+8.4f}")
print("="*60)
print("\nHyperparameter tuning complete.")
