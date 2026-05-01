# Purpose: Evaluate trained models with charts, ROC analysis, and final summary insights.

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Absolute project root — anchored to this file's location.
# All paths below are built with os.path.join(PROJECT_ROOT, ...) so the
# script works correctly regardless of the current working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Canonical directory paths
_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
_STATIC_DIR = os.path.join(PROJECT_ROOT, "static")


def evaluate_models(results, rf_model, all_models, X_test, y_test):
    """Generate comparison charts, ROC curves, and feature importance plots."""
    try:
        sns.set_style("whitegrid")
        os.makedirs(_STATIC_DIR, exist_ok=True)

        # Build comparison table from results dict.
        comparison_rows = []
        for model_name, metric_values in results.items():
            comparison_rows.append({
                "Model":     model_name,
                "Accuracy":  metric_values.get("accuracy",  0.0),
                "Precision": metric_values.get("precision", 0.0),
                "Recall":    metric_values.get("recall",    0.0),
                "F1-Score":  metric_values.get("f1",        0.0),
            })

        comparison_df = pd.DataFrame(comparison_rows)[
            ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        ]

        print("\nModel Comparison Table:")
        if not comparison_df.empty:
            print(comparison_df.round(4).to_string(index=False))
        else:
            print("No model results available to compare.")

        # Grouped bar chart.
        long_df = comparison_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
            var_name="Metric",
            value_name="Score",
        )

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=long_df, x="Metric", y="Score", hue="Model")
        plt.title("Model Performance Comparison")
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.legend(title="Model")

        for patch in ax.patches:
            height = patch.get_height()
            if pd.notna(height):
                ax.annotate(
                    f"{height:.2f}",
                    (patch.get_x() + patch.get_width() / 2.0, height),
                    ha="center", va="bottom", fontsize=8,
                    xytext=(0, 3), textcoords="offset points",
                )

        plt.tight_layout()
        plt.savefig(os.path.join(_STATIC_DIR, "comparison_chart.png"), dpi=300)
        plt.close()

        # Feature importance chart.
        feature_importances = pd.Series(
            rf_model.feature_importances_, index=X_test.columns
        )
        top_10 = feature_importances.sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=top_10.values, y=top_10.index, color="steelblue", label="Importance")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.legend(title="Legend")
        plt.tight_layout()
        plt.savefig(os.path.join(_STATIC_DIR, "feature_importance.png"), dpi=300)
        plt.close()

        # ROC curves.
        plt.figure(figsize=(10, 7))
        for model_name, model in all_models.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
        plt.title("ROC Curve Comparison")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right", title="Models")
        plt.tight_layout()
        plt.savefig(os.path.join(_STATIC_DIR, "roc_curve.png"), dpi=300)
        plt.close()

        # Best model summary.
        ranked_df = comparison_df.sort_values(by=["F1-Score", "Accuracy"], ascending=False)
        best_model_name = ranked_df.iloc[0]["Model"] if not ranked_df.empty else "N/A"

        print("\nFinal Summary:")
        print(f"Best performing model (by F1-Score): {best_model_name}")

        if "Ensemble" in comparison_df["Model"].values:
            ensemble_f1      = comparison_df.loc[comparison_df["Model"] == "Ensemble", "F1-Score"].iloc[0]
            best_individual  = comparison_df.loc[comparison_df["Model"] != "Ensemble", "F1-Score"].max()
            if ensemble_f1 >= best_individual:
                print("Ensemble advantage: combines diverse model decisions, reducing variance.")
            else:
                print("Ensemble advantage: improves stability by blending complementary strengths.")

        return comparison_df

    except Exception as error:
        print(f"Error in evaluate_models: {error}")
        raise


def rebuild_and_save_results(all_models, X_test, y_test, output_path=None):
    """Recompute and save evaluation metrics for all models.

    Parameters
    ----------
    output_path : str, optional
        Absolute path for results.pkl.  Defaults to <project_root>/models/results.pkl.
    """
    if output_path is None:
        output_path = os.path.join(_MODELS_DIR, "results.pkl")

    try:
        import joblib

        results = {}
        for name, model in all_models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }

        joblib.dump(results, output_path)
        return results

    except Exception as error:
        print(f"Error in rebuild_and_save_results: {error}")
        raise


def business_insights(ensemble_model, X_test, y_test):
    """Segment customers by purchase probability and save a pie chart."""
    try:
        sns.set_style("whitegrid")
        os.makedirs(_STATIC_DIR, exist_ok=True)

        proba     = ensemble_model.predict_proba(X_test)[:, 1]
        predicted = ensemble_model.predict(X_test)

        result_df = pd.DataFrame({
            "Probability": proba,
            "Predicted":   predicted,
            "Actual":      y_test.values if hasattr(y_test, "values") else y_test,
        })

        result_df["Segment"] = pd.cut(
            result_df["Probability"],
            bins=[0.0, 0.3, 0.7, 1.0],
            labels=["Browsing", "Interested", "Ready-to-Buy"],
            include_lowest=True,
            right=True,
        )

        action_map = {
            "Browsing":     "Show Discount Offers",
            "Interested":   "Show Product Recommendations",
            "Ready-to-Buy": "Show Urgency Limited Time Offers",
        }
        result_df["Action"] = result_df["Segment"].astype(str).map(action_map)

        print("\nBusiness Insight Preview (First 10 Rows):")
        print(result_df.head(10).to_string(index=False))

        segment_counts = result_df["Segment"].value_counts(dropna=False)
        print("\nUser Count by Segment:")
        print(segment_counts.to_string())

        # Pie chart — saved to absolute path.
        plt.figure(figsize=(7, 7))
        pie_counts = result_df["Segment"].value_counts()
        plt.pie(
            pie_counts.values,
            labels=pie_counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Segment Distribution")
        plt.xlabel("Customer Segments")
        plt.ylabel("Proportion")
        plt.legend(title="Segments", loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(_STATIC_DIR, "segment_distribution.png"), dpi=300)
        plt.close()

        print("\nBusiness Interpretation:")
        print("Browsing:     Low intent — use discounts to increase engagement.")
        print("Interested:   Medium intent — use recommendations to nudge decisions.")
        print("Ready-to-Buy: High intent — use urgency offers to maximise conversions.")

        return result_df

    except Exception as error:
        print(f"Error in business_insights: {error}")
        raise


if __name__ == "__main__":
    import joblib
    from preprocess import load_and_explore, preprocess_data

    # Guard: models must exist before evaluation can run.
    required = [
        os.path.join(_MODELS_DIR, f)
        for f in ("logistic_model.pkl", "rf_model.pkl", "ann_model.pkl", "ensemble_model.pkl")
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: The following model files are missing:")
        for f in missing:
            print(f"  {f}")
        print("Run 'python train.py' first to generate them.")
        raise SystemExit(1)

    # Load data and preprocess.
    df = load_and_explore()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load all saved models using absolute paths.
    logistic_model = joblib.load(os.path.join(_MODELS_DIR, "logistic_model.pkl"))
    rf_model       = joblib.load(os.path.join(_MODELS_DIR, "rf_model.pkl"))
    ann_model      = joblib.load(os.path.join(_MODELS_DIR, "ann_model.pkl"))
    ensemble_model = joblib.load(os.path.join(_MODELS_DIR, "ensemble_model.pkl"))

    all_models = {
        "Logistic":     logistic_model,
        "RandomForest": rf_model,
        "ANN":          ann_model,
        "Ensemble":     ensemble_model,
    }

    results = rebuild_and_save_results(all_models, X_test, y_test)
    evaluate_models(results, rf_model, all_models, X_test, y_test)
    business_insights(ensemble_model, X_test, y_test)
