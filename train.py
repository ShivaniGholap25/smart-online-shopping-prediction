"""Purpose: Train and evaluate multiple ML models for revenue prediction."""

# Import standard library modules for filesystem handling and traceback output.
import os
import traceback

# Import plotting library for confusion matrix visualizations.
import matplotlib.pyplot as plt

# Import joblib to save trained models and evaluation results.
import joblib

# Import required scikit-learn models.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# Import evaluation utilities.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Import preprocessing functions from preprocess.py.
from preprocess import preprocess_data, load_and_explore


def train_all_models(X_train, X_test, y_train, y_test):
    """Train, evaluate, plot, and persist 4 models, then return results and models."""
    # Wrap the entire training workflow in try/except for robust error handling.
    try:
        # Ensure output directories exist before saving artifacts.
        os.makedirs("models", exist_ok=True)
        os.makedirs("static", exist_ok=True)

        # Initialize a dictionary to collect summary metrics for all models.
        results = {}

        # Create MODEL 1: Logistic Regression with requested hyperparameters.
        logistic_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="saga",
        )

        # Create MODEL 2: Random Forest with requested hyperparameters.
        rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )

        # Create MODEL 3: ANN using MLPClassifier with requested hyperparameters.
        ann_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
        )

        # Fit the three base models before individual evaluation and saving.
        logistic_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        ann_model.fit(X_train, y_train)

        # Save trained MODEL 1 to the requested path.
        joblib.dump(logistic_model, "models/logistic_model.pkl")

        # Save trained MODEL 2 to the requested path.
        joblib.dump(rf_model, "models/rf_model.pkl")

        # Save trained MODEL 3 to the requested path.
        joblib.dump(ann_model, "models/ann_model.pkl")

        # Create MODEL 4: soft-voting ensemble using the three base models.
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

        # Fit the ensemble model on the training data.
        ensemble_model.fit(X_train, y_train)

        # Save trained MODEL 4 to the requested path.
        joblib.dump(ensemble_model, "models/ensemble_model.pkl")

        # Group all trained models for unified evaluation logic.
        trained_models = {
            "Logistic": logistic_model,
            "RandomForest": rf_model,
            "ANN": ann_model,
            "Ensemble": ensemble_model,
        }

        # Iterate through each model and compute metrics, reports, and confusion matrices.
        for model_name, model in trained_models.items():
            # Predict labels on the test set.
            y_pred = model.predict(X_test)

            # Compute accuracy score.
            accuracy = accuracy_score(y_test, y_pred)

            # Generate classification report as dict for extracting metrics.
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Generate text report for console visibility.
            report_text = classification_report(y_test, y_pred, zero_division=0)

            # Print model name and primary metrics for quick monitoring.
            print(f"\n===== {model_name} =====")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report_text)

            # Build confusion matrix from true vs predicted labels.
            cm = confusion_matrix(y_test, y_pred)

            # Plot confusion matrix and save to static/ as requested.
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")
            plt.title(f"{model_name} Confusion Matrix")
            plt.tight_layout()
            plt.savefig(f"static/{model_name.lower()}_confusion_matrix.png", dpi=300)
            plt.close()

            # Store selected summary metrics using weighted averages for class imbalance robustness.
            results[model_name] = {
                "accuracy": accuracy,
                "precision": report_dict["weighted avg"]["precision"],
                "recall": report_dict["weighted avg"]["recall"],
                "f1": report_dict["weighted avg"]["f1-score"],
            }

        # Save the complete results dictionary for later use.
        joblib.dump(results, "models/results.pkl")

        # Return results and all 4 trained models in the requested order.
        return results, logistic_model, rf_model, ann_model, ensemble_model

    except Exception as error:
        # Print a readable error message for quick debugging.
        print(f"Error in train_all_models: {error}")

        # Print full traceback for deeper debugging context.
        traceback.print_exc()

        # Re-raise the exception so calling code can handle failure explicitly.
        raise


if __name__ == "__main__":
    # Wrap script execution in try/except for safer standalone usage.
    try:
        # Load and explore the dataset using preprocessing module utilities.
        df = load_and_explore()

        # Preprocess data and split into train/test sets.
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # Train all models and collect returned artifacts.
        all_results, _, _, _, _ = train_all_models(X_train, X_test, y_train, y_test)

        # Print a short summary of stored model metrics.
        print("\nTraining completed successfully. Results summary:")
        for model_name, metrics in all_results.items():
            print(f"{model_name}: {metrics}")

    except Exception as error:
        # Print a high-level error for script-level failures.
        print(f"Error while running training pipeline: {error}")
