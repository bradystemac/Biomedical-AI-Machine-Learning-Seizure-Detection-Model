"""End-to-end epileptic seizure detection pipeline.

Script version of Biomed_AI_Seizure_Project.ipynb. It runs every analysis step
in the notebook and writes all tables to results/ and all plots to figures/.

Usage:
    python scripts/seizure_detection_pipeline.py --data data/archive.zip
    python scripts/seizure_detection_pipeline.py --data "data/Epileptic Seizure Recognition.csv"
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
SAMPLING_RATE_HZ = 173.61
ROOT = Path(__file__).resolve().parent.parent

PARAM_GRIDS = {
    "Logistic Regression": [
        {"penalty": ["l2"], "C": [0.1, 1.0, 10], "solver": ["lbfgs"], "max_iter": [1000]}
    ],
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
    },
    "Support Vector Machine": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf"],
    },
    "Decision Tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, None],
    },
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_features": ["sqrt"],
        "random_state": [RANDOM_STATE],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-5, 1.0],
    },
}


def make_base_estimators():
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }


# ------------------------------------------------------------
# Section 1: Data extraction and loading
# ------------------------------------------------------------
def load_data(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Download it from Kaggle "
            "(see README.md) and pass its location with --data."
        )
    compression = "zip" if data_path.suffix == ".zip" else "infer"
    df = pd.read_csv(data_path, compression=compression)
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    return df


# ------------------------------------------------------------
# Section 2: Preprocessing, scaling and train/test split
# ------------------------------------------------------------
def preprocess(df):
    # Class 1 is seizure (1); classes 2-5 are non-seizure (0)
    y = df["y"].apply(lambda val: 1 if val == 1 else 0).values
    feature_names = df.drop(columns=["Unnamed", "y"]).columns.tolist()
    X = df[feature_names].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")
    print(f"Seizure Cases in Training: {sum(y_train)} / {len(y_train)}")
    return X_train, X_test, y_train, y_test, feature_names


# ------------------------------------------------------------
# Sections 3 and 8: GridSearchCV tuning and holdout evaluation
# ------------------------------------------------------------
def tune_and_evaluate(X_train, X_test, y_train, y_test, label=""):
    results = {}
    best_estimators = {}

    for name, estimator in make_base_estimators().items():
        print(f"Running Grid Search for {name}{label}...")
        start_time = time.time()

        grid_search = GridSearchCV(
            estimator, PARAM_GRIDS[name], cv=5, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        elapsed = time.time() - start_time
        print(f"Finished {name}{label} in {elapsed:.1f}s. Best CV Score: {grid_search.best_score_:.4f}")

        best_clf = grid_search.best_estimator_
        best_estimators[name] = best_clf

        y_pred = best_clf.predict(X_test)
        y_proba = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        results[name] = {
            "Best Parameters": str(grid_search.best_params_),
            "Best CV Accuracy": grid_search.best_score_,
            "Test Accuracy": acc,
            "Test Precision": prec,
            "Test Recall": rec,
            "Test F1-Score": f1,
            "Test ROC-AUC": auc,
            "Search Time (s)": elapsed,
        }

    results_df = pd.DataFrame(results).T.sort_values(by="Test Accuracy", ascending=False)
    return results_df, best_estimators


# ------------------------------------------------------------
# Section 5: Feature importance of the best interpretable model
# ------------------------------------------------------------
def feature_importance(results_df, best_estimators, feature_names, fig_dir):
    interpretable = {
        name: clf
        for name, clf in best_estimators.items()
        if hasattr(clf, "feature_importances_") or hasattr(clf, "coef_")
    }
    best_name = max(interpretable, key=lambda k: results_df.loc[k, "Test Accuracy"])
    best_model = interpretable[best_name]
    print(
        f"Selected Best Tuned Model for Feature Importance: {best_name} "
        f"(Test Accuracy: {results_df.loc[best_name, 'Test Accuracy']:.4f})"
    )

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        imp_type = "Feature Importance Score"
    else:
        importances = np.abs(best_model.coef_[0])
        imp_type = "Absolute Coefficient Value"

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Value": importances})
        .sort_values(by="Value", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=importance_df, palette="viridis", hue="Feature", legend=False)
    plt.title(f"Top 15 Most Important Features in Tuned {best_name}")
    plt.xlabel(imp_type)
    plt.ylabel("Feature (EEG Timepoint)")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance_top15.png", dpi=150)
    plt.close()

    print("=== Top 3 Most Important Features ===")
    print(importance_df["Feature"].head(3).tolist())
    return importance_df, best_name


# ------------------------------------------------------------
# Section 6: Temporal mapping of key EEG timepoints
# ------------------------------------------------------------
def temporal_mapping(importance_df):
    top = importance_df.head(5).copy()
    top["Timepoint_Index"] = top["Feature"].str.extract(r"(\d+)", expand=False).astype(int)
    top["Timestamp (Seconds)"] = (top["Timepoint_Index"] / SAMPLING_RATE_HZ).round(4)
    top = top[["Feature", "Value", "Timestamp (Seconds)"]].rename(columns={"Value": "Importance Score"})
    print("=== Temporal Mapping of Top 5 Most Important Features ===")
    print(top.to_string(index=False))
    return top


# ------------------------------------------------------------
# Section 7: PCA dimensionality reduction
# ------------------------------------------------------------
def apply_pca(X_train, X_test, fig_dir):
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Reduced number of features (preserving 95% variance): {X_train_pca.shape[1]}")

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.axhline(y=0.95, color="r", linestyle=":", label="95% Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_cumulative_variance.png", dpi=150)
    plt.close()
    return X_train_pca, X_test_pca


# ------------------------------------------------------------
# Section 9: Original vs. PCA comparison
# ------------------------------------------------------------
def compare(original_df, pca_df, fig_dir):
    comparison = pd.DataFrame({
        "Original Test Accuracy": original_df["Test Accuracy"],
        "PCA Test Accuracy": pca_df["Test Accuracy"],
        "Original Test F1-Score": original_df["Test F1-Score"],
        "PCA Test F1-Score": pca_df["Test F1-Score"],
        "Original Best Parameters": original_df["Best Parameters"],
        "PCA Best Parameters": pca_df["Best Parameters"],
    }).sort_values(by="Original Test Accuracy", ascending=False)

    print("=== Performance Comparison: Original vs. PCA-Reduced Data ===")
    print(comparison.drop(columns=["Original Best Parameters", "PCA Best Parameters"]).to_string())

    plot_df = comparison[["Original Test Accuracy", "PCA Test Accuracy"]].astype(float)
    ax = plot_df.plot(kind="barh", figsize=(9, 5), color=["#2563eb", "#16a34a"])
    ax.set_xlabel("Test Accuracy")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Test Accuracy: Original 178 Features vs. PCA (95% Variance)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_dir / "original_vs_pca_accuracy.png", dpi=150)
    plt.close()
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=ROOT / "data" / "archive.zip",
                        help="Path to archive.zip or 'Epileptic Seizure Recognition.csv'")
    parser.add_argument("--results-dir", default=ROOT / "results", type=Path)
    parser.add_argument("--fig-dir", default=ROOT / "figures", type=Path)
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    original_df, best_estimators = tune_and_evaluate(X_train, X_test, y_train, y_test)
    print("=== Grid Search Tuned Models Performance Comparison ===")
    print(original_df.drop(columns=["Best Parameters"]).to_string())
    original_df.to_csv(args.results_dir / "tuned_results_original.csv")

    importance_df, _ = feature_importance(original_df, best_estimators, feature_names, args.fig_dir)
    importance_df.to_csv(args.results_dir / "feature_importance_top15.csv", index=False)
    temporal_mapping(importance_df).to_csv(args.results_dir / "temporal_mapping_top5.csv", index=False)

    X_train_pca, X_test_pca = apply_pca(X_train, X_test, args.fig_dir)
    pca_df, _ = tune_and_evaluate(X_train_pca, X_test_pca, y_train, y_test, label=" with PCA")
    pca_df.to_csv(args.results_dir / "tuned_results_pca.csv")

    compare(original_df, pca_df, args.fig_dir).to_csv(args.results_dir / "comparison_original_vs_pca.csv")

    print(f"\nResults saved to {args.results_dir}")
    print(f"Figures saved to {args.fig_dir}")


if __name__ == "__main__":
    main()
