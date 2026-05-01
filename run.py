# Purpose: Run the full purchase intent pipeline from data loading to business insights.

import os
import time

# ---------------------------------------------------------------------------
# Absolute project root — anchored to this file's location.
# Importing preprocess/train/evaluate works from any working directory
# because we add PROJECT_ROOT to sys.path before the imports.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocess import load_and_explore, preprocess_data
from train      import train_all_models
from evaluate   import evaluate_models, business_insights


def run_step(step_name, step_function, *args, **kwargs):
    """Execute one pipeline step, print timing, and re-raise on failure."""
    start_time = time.perf_counter()
    try:
        result  = step_function(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[DONE] {step_name} completed in {elapsed:.2f} seconds")
        return result
    except Exception as error:
        elapsed = time.perf_counter() - start_time
        print(f"[ERROR] {step_name} failed after {elapsed:.2f} seconds")
        print(f"Reason: {error}")
        raise


def main():
    try:
        print("Starting pipeline execution...\n")

        # Step 1: Load dataset and run EDA.
        df = run_step("Step 1 - load_and_explore", load_and_explore)

        # Step 2: Preprocess data and create train/test splits.
        X_train, X_test, y_train, y_test = run_step(
            "Step 2 - preprocess_data", preprocess_data, df,
        )

        # Step 3: Train all models.
        results, logistic_model, rf_model, ann_model, ensemble_model = run_step(
            "Step 3 - train_all_models",
            train_all_models, X_train, X_test, y_train, y_test,
        )

        all_models = {
            "Logistic":     logistic_model,
            "RandomForest": rf_model,
            "ANN":          ann_model,
            "Ensemble":     ensemble_model,
        }

        # Step 4: Evaluate models.
        run_step(
            "Step 4 - evaluate_models",
            evaluate_models, results, rf_model, all_models, X_test, y_test,
        )

        # Step 5: Business insights.
        run_step(
            "Step 5 - business_insights",
            business_insights, ensemble_model, X_test, y_test,
        )

        print("\nPipeline execution completed successfully.")

    except Exception:
        print("\nPipeline execution stopped due to an error.")


if __name__ == "__main__":
    main()
