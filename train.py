"""Purpose: Train and evaluate multiple ML models for revenue prediction."""

import os
import traceback

import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocess import preprocess_data, load_and_explore

# ---------------------------------------------------------------------------
# Absolute project root — anchored to this file's location.
# All paths below are built with os.path.join(PROJECT_ROOT, ...) so the
# script works correctly regardless of the current working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Canonical directory paths
_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
_STATIC_DIR = os.path.join(PROJECT_ROOT, "static")


def train_all_models(X_train, X_test, y_train, y_test):
    """Train, evaluate, plot, and persist 4 models, then return results and models."""
    try:
        # Ensure output directories exist before saving artifacts.
        os.makedirs(_MODELS_DIR, exist_ok=True)
        os.makedirs(_STATIC_DIR, exist_ok=True)

        results = {}

        # ── MODEL 1: Logistic Regression ─────────────────────────────────
        logistic_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="saga",
        )

        # ── MODEL 2: Random Forest ────────────────────────────────────────
        rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )

        # ── MODEL 3: ANN (MLPClassifier) ──────────────────────────────────
        ann_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
        )

        # Fit the three base models.
        logistic_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        ann_model.fit(X_train, y_train)

        # Save base models using absolute paths.
        joblib.dump(logistic_model, os.path.join(_MODELS_DIR, "logistic_model.pkl"))
        joblib.dump(rf_model,       os.path.join(_MODELS_DIR, "rf_model.pkl"))
        joblib.dump(ann_model,      os.path.join(_MODELS_DIR, "ann_model.pkl"))

        # ── MODEL 4: Soft-voting Ensemble ─────────────────────────────────
        ensemble_model = VotingClassifier(
            estimators=[
                ("logistic", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    solver="saga",
                )),
                ("random_forest", RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                )),
                ("ann", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    random_state=42,
                )),
            ],
            voting="soft",
        )

        ensemble_model.fit(X_train, y_train)
        joblib.dump(ensemble_model, os.path.join(_MODELS_DIR, "ensemble_model.pkl"))

        # ── Evaluate all models ───────────────────────────────────────────
        trained_models = {
            "Logistic":     logistic_model,
            "RandomForest": rf_model,
            "ANN":          ann_model,
            "Ensemble":     ensemble_model,
        }

        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test)

            accuracy    = accuracy_score(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_text = classification_report(y_test, y_pred, zero_division=0)

            print(f"\n===== {model_name} =====")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report_text)

            # Confusion matrix — saved to absolute path.
            cm   = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")
            plt.title(f"{model_name} Confusion Matrix")
            plt.tight_layout()
            plt.savefig(
                os.path.join(_STATIC_DIR, f"{model_name.lower()}_confusion_matrix.png"),
                dpi=300,
            )
            plt.close()

            results[model_name] = {
                "accuracy":  accuracy,
                "precision": report_dict["weighted avg"]["precision"],
                "recall":    report_dict["weighted avg"]["recall"],
                "f1":        report_dict["weighted avg"]["f1-score"],
            }

        # Save results dict using absolute path.
        joblib.dump(results, os.path.join(_MODELS_DIR, "results.pkl"))

        return results, logistic_model, rf_model, ann_model, ensemble_model

    except Exception as error:
        print(f"Error in train_all_models: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        df = load_and_explore()
        X_train, X_test, y_train, y_test = preprocess_data(df)
        all_results, _, _, _, _ = train_all_models(X_train, X_test, y_train, y_test)

        print("\nTraining completed successfully. Results summary:")
        for model_name, metrics in all_results.items():
            print(f"{model_name}: {metrics}")

    except Exception as error:
        print(f"Error while running training pipeline: {error}")
