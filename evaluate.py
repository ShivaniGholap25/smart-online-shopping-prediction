# Purpose: Evaluate trained models with charts, ROC analysis, and final summary insights.

# Import built-in module for creating required output directories.
import os
import sys

# Resolve the project root as the directory containing this script.
# This ensures all relative paths work correctly regardless of where
# Python is invoked from (e.g. parent folder, IDE, or command line).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Import pandas for tabular comparison output.
import pandas as pd

# Import matplotlib and seaborn for plotting evaluation charts.
import matplotlib.pyplot as plt
import seaborn as sns

# Import ROC/AUC utilities for classifier curve analysis.
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_models(results, rf_model, all_models, X_test, y_test):
    # Wrap evaluation workflow in try/except for robust error handling.
    try:
        # Set a consistent seaborn style for all plots.
        sns.set_style("whitegrid")

        # Ensure the static folder exists before saving any images.
        os.makedirs("static", exist_ok=True)

        # Build a comparison table from the results dictionary.
        comparison_rows = []
        for model_name, metric_values in results.items():
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Accuracy": metric_values.get("accuracy", 0.0),
                    "Precision": metric_values.get("precision", 0.0),
                    "Recall": metric_values.get("recall", 0.0),
                    "F1-Score": metric_values.get("f1", 0.0),
                }
            )

        # Create a DataFrame for clear metric comparison across models.
        comparison_df = pd.DataFrame(comparison_rows)[
            ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        ]

        # Print the comparison DataFrame in a clean aligned view.
        print("\nModel Comparison Table:")
        if not comparison_df.empty:
            print(comparison_df.round(4).to_string(index=False))
        else:
            print("No model results available to compare.")

        # Prepare data for grouped bar chart by reshaping wide metrics into long format.
        long_df = comparison_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
            var_name="Metric",
            value_name="Score",
        )

        # Plot grouped bar chart to compare metrics across models.
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=long_df, x="Metric", y="Score", hue="Model")
        plt.title("Model Performance Comparison")
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.legend(title="Model")

        # Add value labels on top of each bar for exact score readability.
        for patch in ax.patches:
            height = patch.get_height()
            if pd.notna(height):
                ax.annotate(
                    f"{height:.2f}",
                    (patch.get_x() + patch.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        # Save grouped comparison chart to static folder.
        plt.tight_layout()
        plt.savefig("static/comparison_chart.png", dpi=300)
        plt.close()

        # Extract Random Forest feature importances and align with feature names.
        feature_importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)

        # Select top 10 most important features in descending order.
        top_10_importances = feature_importances.sort_values(ascending=False).head(10)

        # Plot horizontal bar chart for top 10 feature importances.
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x=top_10_importances.values,
            y=top_10_importances.index,
            color="steelblue",
            label="Importance",
        )
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.legend(title="Legend")

        # Save feature importance chart to static folder.
        plt.tight_layout()
        plt.savefig("static/feature_importance.png", dpi=300)
        plt.close()

        # Plot ROC curves for all models and display each model's AUC in legend.
        plt.figure(figsize=(10, 7))

        # Loop through all models to compute ROC points and AUC scores.
        for model_name, model in all_models.items():
            # Use predicted probabilities for positive class when available.
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                # Fall back to decision_function if probabilities are unavailable.
                y_score = model.decision_function(X_test)
            else:
                # Skip models that cannot provide a ranking score for ROC.
                continue

            # Compute ROC curve coordinates and area under curve.
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            # Draw each ROC curve with AUC in the legend label.
            plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

        # Plot random baseline line for reference.
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")

        # Set chart title, axis labels, and legend.
        plt.title("ROC Curve Comparison")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right", title="Models")

        # Save ROC curve chart to static folder.
        plt.tight_layout()
        plt.savefig("static/roc_curve.png", dpi=300)
        plt.close()

        # Select best model based on highest F1-Score, then Accuracy as tie-breaker.
        ranked_df = comparison_df.sort_values(
            by=["F1-Score", "Accuracy"],
            ascending=False,
        )
        best_model_name = ranked_df.iloc[0]["Model"] if not ranked_df.empty else "N/A"

        # Print final summary indicating best model and why ensembles can outperform single models.
        print("\nFinal Summary:")
        print(f"Best performing model (by F1-Score): {best_model_name}")

        # Compare ensemble score against the best individual model for interpretability.
        if "Ensemble" in comparison_df["Model"].values:
            ensemble_f1 = comparison_df.loc[
                comparison_df["Model"] == "Ensemble", "F1-Score"
            ].iloc[0]
            best_individual_f1 = comparison_df.loc[
                comparison_df["Model"] != "Ensemble", "F1-Score"
            ].max()

            if ensemble_f1 >= best_individual_f1:
                print(
                    "Ensemble advantage: It combines diverse model decisions, "
                    "which reduces variance and improves robustness compared to individual models."
                )
            else:
                print(
                    "Ensemble advantage: Even when not top in this split, ensembles generally improve "
                    "stability by blending complementary strengths, reducing variance, and mitigating "
                    "single-model blind spots."
                )
        else:
            print(
                "Ensemble advantage: Ensembles typically beat individual models by combining "
                "complementary patterns and reducing overfitting risk."
            )

        # Return the comparison DataFrame for optional downstream use.
        return comparison_df

    except Exception as error:
        # Print and re-raise errors so callers can handle failures explicitly.
        print(f"Error in evaluate_models: {error}")
        raise


def rebuild_and_save_results(all_models, X_test, y_test, output_path="models/results.pkl"):
    # Wrap results rebuilding in try/except for robust error handling.
    try:
        # Compute fresh metrics for all models to avoid stale/incorrect stored values.
        results = {
            "Logistic": {
                "accuracy": accuracy_score(y_test, all_models["Logistic"].predict(X_test)),
                "precision": precision_score(y_test, all_models["Logistic"].predict(X_test), average="weighted"),
                "recall": recall_score(y_test, all_models["Logistic"].predict(X_test), average="weighted"),
                "f1": f1_score(y_test, all_models["Logistic"].predict(X_test), average="weighted"),
            },
            "RandomForest": {
                "accuracy": accuracy_score(y_test, all_models["RandomForest"].predict(X_test)),
                "precision": precision_score(y_test, all_models["RandomForest"].predict(X_test), average="weighted"),
                "recall": recall_score(y_test, all_models["RandomForest"].predict(X_test), average="weighted"),
                "f1": f1_score(y_test, all_models["RandomForest"].predict(X_test), average="weighted"),
            },
            "ANN": {
                "accuracy": accuracy_score(y_test, all_models["ANN"].predict(X_test)),
                "precision": precision_score(y_test, all_models["ANN"].predict(X_test), average="weighted"),
                "recall": recall_score(y_test, all_models["ANN"].predict(X_test), average="weighted"),
                "f1": f1_score(y_test, all_models["ANN"].predict(X_test), average="weighted"),
            },
            "Ensemble": {
                "accuracy": accuracy_score(y_test, all_models["Ensemble"].predict(X_test)),
                "precision": precision_score(y_test, all_models["Ensemble"].predict(X_test), average="weighted"),
                "recall": recall_score(y_test, all_models["Ensemble"].predict(X_test), average="weighted"),
                "f1": f1_score(y_test, all_models["Ensemble"].predict(X_test), average="weighted"),
            },
        }

        # Save the corrected results dictionary for app/evaluation reuse.
        joblib.dump(results, output_path)

        # Return corrected results to the caller.
        return results

    except Exception as error:
        # Print and re-raise errors so callers can handle failures explicitly.
        print(f"Error in rebuild_and_save_results: {error}")
        raise


def business_insights(ensemble_model, X_test, y_test):
    # Wrap business insight generation in try/except for robust error handling.
    try:
        # Set consistent seaborn style for plotting.
        sns.set_style("whitegrid")

        # Ensure static output directory exists before saving charts.
        os.makedirs("static", exist_ok=True)

        # Get positive-class probabilities from the ensemble model.
        proba = ensemble_model.predict_proba(X_test)[:, 1]

        # Generate binary predictions using the model default decision threshold.
        predicted = ensemble_model.predict(X_test)

        # Create result DataFrame with requested columns.
        result_df = pd.DataFrame(
            {
                "Probability": proba,
                "Predicted": predicted,
                "Actual": y_test.values if hasattr(y_test, "values") else y_test,
            }
        )

        # Create customer segment labels based on probability ranges.
        result_df["Segment"] = pd.cut(
            result_df["Probability"],
            bins=[0.0, 0.3, 0.7, 1.0],
            labels=["Browsing", "Interested", "Ready-to-Buy"],
            include_lowest=True,
            right=True,
        )

        # Map each segment to a recommended business action.
        action_map = {
            "Browsing": "Show Discount Offers",
            "Interested": "Show Product Recommendations",
            "Ready-to-Buy": "Show Urgency Limited Time Offers",
        }
        result_df["Action"] = result_df["Segment"].astype(str).map(action_map)

        # Print first 10 rows for quick inspection.
        print("\nBusiness Insight Preview (First 10 Rows):")
        print(result_df.head(10).to_string(index=False))

        # Compute and print user counts per segment.
        segment_counts = result_df["Segment"].value_counts(dropna=False)
        print("\nUser Count by Segment:")
        print(segment_counts.to_string())

        # Plot and save pie chart for segment distribution.
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
        plt.savefig("static/segment_distribution.png", dpi=300)
        plt.close()

        # Print business interpretation for each segment.
        print("\nBusiness Interpretation:")
        print(
            "Browsing: Low purchase intent users; use discounts to increase engagement and move them toward conversion."
        )
        print(
            "Interested: Medium intent users; use personalized recommendations to nudge decision-making."
        )
        print(
            "Ready-to-Buy: High intent users; use urgency-driven limited-time offers to maximize immediate conversions."
        )

        # Return the result DataFrame for optional downstream use.
        return result_df

    except Exception as error:
        # Print and re-raise errors so callers can handle failures explicitly.
        print(f"Error in business_insights: {error}")
        raise


if __name__ == "__main__":
    import joblib
    from preprocess import load_and_explore, preprocess_data

    # Ensure models directory exists before attempting to load artifacts.
    if not os.path.isdir("models"):
        print("ERROR: 'models/' directory not found.")
        print("Run 'python train.py' first to generate all model files.")
        raise SystemExit(1)

    required_models = [
        "models/logistic_model.pkl",
        "models/rf_model.pkl",
        "models/ann_model.pkl",
        "models/ensemble_model.pkl",
    ]
    missing = [f for f in required_models if not os.path.exists(f)]
    if missing:
        print("ERROR: The following model files are missing:")
        for f in missing:
            print(f"  {f}")
        print("Run 'python train.py' first to generate them.")
        raise SystemExit(1)

    # Load data and preprocess
    df = load_and_explore()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load all saved models
    logistic_model = joblib.load("models/logistic_model.pkl")
    rf_model = joblib.load("models/rf_model.pkl")
    ann_model = joblib.load("models/ann_model.pkl")
    ensemble_model = joblib.load("models/ensemble_model.pkl")
    all_models = {
        "Logistic": logistic_model,
        "RandomForest": rf_model,
        "ANN": ann_model,
        "Ensemble": ensemble_model
    }

    # Rebuild and save corrected metrics for all 4 models (including Ensemble).
    results = rebuild_and_save_results(all_models, X_test, y_test, output_path="models/results.pkl")

    # Run evaluation
    evaluate_models(results, rf_model, all_models, X_test, y_test)
    business_insights(ensemble_model, X_test, y_test)
