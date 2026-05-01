# Purpose: Load the online shoppers dataset, run EDA, and save key plots.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Absolute project root — anchored to this file's location.
# All paths below are built with os.path.join(PROJECT_ROOT, ...) so the
# script works correctly regardless of the current working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Canonical directory paths
_DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
_STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_and_explore(
    dataset_path: str = None,
    output_dir:   str = None,
):
    """Load the dataset, print EDA summary, and save visualisation plots.

    Parameters
    ----------
    dataset_path : str, optional
        Path to the CSV file.  Defaults to <project_root>/data/online_shoppers_intention.csv.
    output_dir : str, optional
        Directory where plots are saved.  Defaults to <project_root>/static/.
    """
    # Apply defaults anchored to the project root.
    if dataset_path is None:
        dataset_path = os.path.join(_DATA_DIR, "online_shoppers_intention.csv")
    if output_dir is None:
        output_dir = _STATIC_DIR

    # Set seaborn style for consistent plot formatting.
    sns.set_style("whitegrid")

    # Create the output directory if it does not already exist.
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset into a pandas DataFrame.
    df = pd.read_csv(dataset_path)

    # Convert Revenue to 1/0 if it is stored as boolean values.
    if df["Revenue"].dtype == bool:
        df["Revenue"] = df["Revenue"].astype(int)
    elif df["Revenue"].dtype == object:
        df["Revenue"] = df["Revenue"].map(
            {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
        )

    # Print dataset shape to show rows and columns.
    print("Shape:", df.shape)

    # Print each column data type for schema inspection.
    print("\nData Types:")
    print(df.dtypes)

    # Print missing value counts for every column.
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Print class distribution for target column Revenue.
    print("\nRevenue Class Distribution (Counts):")
    print(df["Revenue"].value_counts(dropna=False))

    # Print class distribution percentages for target column Revenue.
    print("\nRevenue Class Distribution (Percent):")
    print(df["Revenue"].value_counts(normalize=True, dropna=False) * 100)

    # Plot and save countplot of Revenue (0 vs 1).
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Revenue", hue="Revenue", dodge=False, legend=False)
    plt.title("Countplot of Revenue (0 vs 1)")
    plt.xlabel("Revenue")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "revenue_countplot.png"), dpi=300)
    plt.close()

    # Select numeric columns and compute correlation matrix.
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()

    # Plot and save correlation heatmap for numeric columns.
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Columns")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # Plot and save distribution of PageValues.
    plt.figure(figsize=(8, 4))
    sns.histplot(df["PageValues"], kde=True, bins=30, color="teal")
    plt.title("Distribution of PageValues")
    plt.xlabel("PageValues")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pagevalues_distribution.png"), dpi=300)
    plt.close()

    # Plot and save distribution of BounceRates.
    plt.figure(figsize=(8, 4))
    sns.histplot(df["BounceRates"], kde=True, bins=30, color="orange")
    plt.title("Distribution of BounceRates")
    plt.xlabel("BounceRates")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bouncerates_distribution.png"), dpi=300)
    plt.close()

    # Plot and save barplot of VisitorType vs Revenue (mean conversion rate).
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="VisitorType", y="Revenue", estimator="mean", errorbar=None)
    plt.title("VisitorType vs Revenue")
    plt.xlabel("VisitorType")
    plt.ylabel("Average Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visitortype_vs_revenue.png"), dpi=300)
    plt.close()

    # Return the loaded DataFrame for downstream usage.
    return df


def preprocess_data(df):
    """Clean, encode, scale, and split the dataset.

    Saves models/scaler.pkl using an absolute path anchored to PROJECT_ROOT.
    Returns X_train, X_test, y_train, y_test as float64 DataFrames/Series.
    """
    try:
        # Create a working copy so the original input DataFrame is not modified in place.
        data = df.copy()

        # Convert Revenue values from True/False (or string variants) to 1/0.
        data["Revenue"] = data["Revenue"].replace(
            {True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
        )

        # Fill missing values in numeric columns with the median of each column.
        numeric_cols = data.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())

        # Fill missing values in categorical columns with the mode of each column.
        categorical_cols = data.select_dtypes(exclude=["number"]).columns
        for col in categorical_cols:
            mode_value = data[col].mode(dropna=True)
            if not mode_value.empty:
                data[col] = data[col].fillna(mode_value.iloc[0])

        # Label encode Month, VisitorType, and Weekend columns.
        label_encode_cols = ["Month", "VisitorType", "Weekend"]
        for col in label_encode_cols:
            if col in data.columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col].astype(str))

        # One-hot encode OperatingSystems, Browser, Region, and TrafficType columns.
        one_hot_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]
        existing_one_hot_cols = [col for col in one_hot_cols if col in data.columns]
        data = pd.get_dummies(data, columns=existing_one_hot_cols, dtype=int).astype(float)

        # Separate features (X) and target (y) using Revenue as target.
        X = data.drop(columns=["Revenue"])
        y = data["Revenue"].astype(int)

        # Define numeric columns to scale as requested.
        scale_cols = [
            "Administrative", "Informational", "ProductRelated",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        ]

        # Keep only columns that exist in X to avoid key errors.
        existing_scale_cols = [col for col in scale_cols if col in X.columns]

        # Fit StandardScaler on selected numeric columns and transform them.
        scaler = StandardScaler()
        if existing_scale_cols:
            X.loc[:, existing_scale_cols] = X[existing_scale_cols].astype("float64")
            X.loc[:, existing_scale_cols] = scaler.fit_transform(X[existing_scale_cols])

        # Ensure the models directory exists before saving the scaler artifact.
        os.makedirs(_MODELS_DIR, exist_ok=True)

        # Save the fitted scaler using an absolute path.
        scaler_path = os.path.join(_MODELS_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        # Split into train/test sets with 80/20 ratio and stratified target split.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        # Ensure train/test feature matrices are float64 after splitting.
        X_train = X_train.astype(float)
        X_test  = X_test.astype(float)

        # Print resulting split shapes for quick verification.
        print("X_train shape:", X_train.shape)
        print("X_test shape:",  X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:",  y_test.shape)

        return X_train, X_test, y_train, y_test

    except Exception as error:
        print(f"Error in preprocess_data: {error}")
        raise


if __name__ == "__main__":
    load_and_explore()
