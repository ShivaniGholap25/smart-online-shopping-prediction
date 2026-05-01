# Purpose: Load the online shoppers dataset, run EDA, and save key plots.

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils.paths import get_path

# Canonical directory paths — derived from the single source of truth.
_DATA_DIR   = get_path("data")
_STATIC_DIR = get_path("static")
_MODELS_DIR = get_path("models")


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
    if dataset_path is None:
        dataset_path = get_path("data", "online_shoppers_intention.csv")
    if output_dir is None:
        output_dir = _STATIC_DIR

    sns.set_style("whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)

    # Convert Revenue to 1/0 if it is stored as boolean values.
    if df["Revenue"].dtype == bool:
        df["Revenue"] = df["Revenue"].astype(int)
    elif df["Revenue"].dtype == object:
        df["Revenue"] = df["Revenue"].map(
            {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
        )

    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nRevenue Class Distribution (Counts):")
    print(df["Revenue"].value_counts(dropna=False))
    print("\nRevenue Class Distribution (Percent):")
    print(df["Revenue"].value_counts(normalize=True, dropna=False) * 100)

    # Countplot of Revenue.
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Revenue", hue="Revenue", dodge=False, legend=False)
    plt.title("Countplot of Revenue (0 vs 1)")
    plt.xlabel("Revenue")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "revenue_countplot.png"), dpi=300)
    plt.close()

    # Correlation heatmap.
    numeric_df   = df.select_dtypes(include=["number"])
    corr_matrix  = numeric_df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Columns")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # PageValues distribution.
    plt.figure(figsize=(8, 4))
    sns.histplot(df["PageValues"], kde=True, bins=30, color="teal")
    plt.title("Distribution of PageValues")
    plt.xlabel("PageValues")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pagevalues_distribution.png"), dpi=300)
    plt.close()

    # BounceRates distribution.
    plt.figure(figsize=(8, 4))
    sns.histplot(df["BounceRates"], kde=True, bins=30, color="orange")
    plt.title("Distribution of BounceRates")
    plt.xlabel("BounceRates")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bouncerates_distribution.png"), dpi=300)
    plt.close()

    # VisitorType vs Revenue.
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="VisitorType", y="Revenue", estimator="mean", errorbar=None)
    plt.title("VisitorType vs Revenue")
    plt.xlabel("VisitorType")
    plt.ylabel("Average Revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "visitortype_vs_revenue.png"), dpi=300)
    plt.close()

    return df


def preprocess_data(df):
    """Clean, encode, scale, and split the dataset.

    Saves models/scaler.pkl via get_path().
    Returns X_train, X_test, y_train, y_test as float64 DataFrames/Series.
    """
    try:
        data = df.copy()

        data["Revenue"] = data["Revenue"].replace(
            {True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0}
        )

        # Fill missing values.
        for col in data.select_dtypes(include=["number"]).columns:
            data[col] = data[col].fillna(data[col].median())
        for col in data.select_dtypes(exclude=["number"]).columns:
            mode_value = data[col].mode(dropna=True)
            if not mode_value.empty:
                data[col] = data[col].fillna(mode_value.iloc[0])

        # Label encode.
        for col in ["Month", "VisitorType", "Weekend"]:
            if col in data.columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col].astype(str))

        # One-hot encode.
        ohe_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]
        existing_ohe = [c for c in ohe_cols if c in data.columns]
        data = pd.get_dummies(data, columns=existing_ohe, dtype=int).astype(float)

        X = data.drop(columns=["Revenue"])
        y = data["Revenue"].astype(int)

        # Scale selected numeric columns.
        scale_cols = [
            "Administrative", "Informational", "ProductRelated",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        ]
        existing_scale = [c for c in scale_cols if c in X.columns]

        scaler = StandardScaler()
        if existing_scale:
            X.loc[:, existing_scale] = X[existing_scale].astype("float64")
            X.loc[:, existing_scale] = scaler.fit_transform(X[existing_scale])

        # Save scaler via get_path().
        os.makedirs(_MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, get_path("models", "scaler.pkl"))

        # Train/test split.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        X_train = X_train.astype(float)
        X_test  = X_test.astype(float)

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
