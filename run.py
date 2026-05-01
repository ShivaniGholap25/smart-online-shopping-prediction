# Purpose: Run the full purchase intent pipeline from data loading to business insights.

# Import standard library modules.
import os
import time

# Resolve the project root as the directory containing this script.
# This ensures all relative paths work correctly regardless of where
# Python is invoked from (e.g. parent folder, IDE, or command line).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Import data loading and preprocessing functions.
from preprocess import load_and_explore, preprocess_data

# Import model training function.
from train import train_all_models

# Import evaluation and business insight functions.
from evaluate import evaluate_models, business_insights


def run_step(step_name, step_function, *args, **kwargs):
    # Record step start time for duration reporting.
    start_time = time.perf_counter()

    # Execute the step and handle errors with a clear message.
    try:
        result = step_function(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[DONE] {step_name} completed in {elapsed:.2f} seconds")
        return result
    except Exception as error:
        elapsed = time.perf_counter() - start_time
        print(f"[ERROR] {step_name} failed after {elapsed:.2f} seconds")
        print(f"Reason: {error}")
        raise


def main():
    # Wrap the full pipeline in try/except for top-level failure clarity.
    try:
        print("Starting pipeline execution...\n")

        # Step 1: Load dataset and run exploratory analysis with saved plots.
        df = run_step("Step 1 - load_and_explore", load_and_explore)

        # Step 2: Preprocess data and create train/test splits.
        X_train, X_test, y_train, y_test = run_step(
            "Step 2 - preprocess_data",
            preprocess_data,
            df,
        )

        # Step 3: Train all models and collect results and model objects.
        results, logistic_model, rf_model, ann_model, ensemble_model = run_step(
            "Step 3 - train_all_models",
            train_all_models,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # Prepare a model dictionary for multi-model evaluation plots.
        all_models = {
            "Logistic": logistic_model,
            "RandomForest": rf_model,
            "ANN": ann_model,
            "Ensemble": ensemble_model,
        }

        # Step 4: Evaluate models and generate comparison/ROC/importance outputs.
        _ = run_step(
            "Step 4 - evaluate_models",
            evaluate_models,
            results,
            rf_model,
            all_models,
            X_test,
            y_test,
        )

        # Step 5: Generate segment-level business insights from ensemble probabilities.
        _ = run_step(
            "Step 5 - business_insights",
            business_insights,
            ensemble_model,
            X_test,
            y_test,
        )

        # Print a clear success message when all steps complete.
        print("\nPipeline execution completed successfully.")

    except Exception:
        # Print a final failure marker for quick debugging in terminal logs.
        print("\nPipeline execution stopped due to an error.")


if __name__ == "__main__":
    # Run the full pipeline when this script is executed directly.
    main()
