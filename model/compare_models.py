"""
AutoML / Model Comparison Script — Predictive Maintenance Project.

Purpose
-------
Compare a set of candidate classifiers on the same train/test split used
for the production Random Forest. Results are tracked in MLflow under the
experiment 'Predictive_Maintenance_ModelComparison'.

This is an EXPERIMENTATION script only.
- It does NOT replace the existing production Random Forest.
- The application always loads machine_failure_model.pkl (Random Forest).
- Use this to explore whether another algorithm would outperform the RF.

Models compared
---------------
1. Random Forest (production baseline)
2. Gradient Boosting
3. Logistic Regression
4. Decision Tree
5. K-Nearest Neighbours

Run
---
    python model/compare_models.py

Requirements
------------
    pip install -r model/requirements.txt
    # MLflow server should be running on MLFLOW_TRACKING_URI
"""

import io
import os
import sys

# Fix: MLflow 3.x emits emoji in run-end message; Windows cp1252 cannot encode it.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 7
TEST_SIZE = 0.2
EXPERIMENT_NAME = "Predictive_Maintenance_ModelComparison"
ARTIFACT_DIR = "mlflow_artifacts"

# ── Candidate models ─────────────────────────────────────────────────────────
CANDIDATES = {
    "RandomForest_baseline": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE, learning_rate=0.1
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "DecisionTree": DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", max_depth=10
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}


def load_and_preprocess(dataset_path: str):
    """Load AI4I dataset and apply same preprocessing as production training."""
    df = pd.read_csv(dataset_path)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(["Machine failure", "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns)


def evaluate(model, X_test, y_test) -> dict:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan"),
    }


def plot_comparison(results: dict, save_path: str) -> None:
    """Bar chart comparing all models across key metrics."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — AI4I Predictive Maintenance")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Comparison chart saved: {save_path}")


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    dataset_path = os.path.join("dataset", "ai4i2020.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run model/train_model.py first."
        )

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(dataset_path)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # Scale features once — used by distance/linear models; tree models ignore it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment: '{EXPERIMENT_NAME}' at {tracking_uri}\n")

    all_results = {}

    for model_name, model in CANDIDATES.items():
        print(f"Training {model_name}...")

        # Distance/linear models benefit from scaling; tree models do not
        needs_scaling = model_name in {"LogisticRegression", "KNN"}
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test

        model.fit(X_tr, y_train)
        metrics = evaluate(model, X_te, y_test)
        all_results[model_name] = metrics

        # Log each candidate as a separate MLflow run
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({"model_type": model_name, "scaled": needs_scaling})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "experiment_type": "model_comparison",
                "dataset": "AI4I_2020",
            })

        print(
            f"  Accuracy={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_score']:.4f}  "
            f"ROC_AUC={metrics['roc_auc']:.4f}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Accuracy':>9} {'F1':>9} {'ROC-AUC':>9}")
    print("-" * 70)
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["f1_score"], reverse=True):
        print(f"{name:<30} {m['accuracy']:>9.4f} {m['f1_score']:>9.4f} {m['roc_auc']:>9.4f}")
    print("=" * 70)

    # ── Comparison chart → logged to MLflow general run ───────────────────────
    chart_path = os.path.join(ARTIFACT_DIR, "model_comparison.png")
    plot_comparison(all_results, chart_path)

    with mlflow.start_run(run_name="comparison_summary"):
        mlflow.log_artifact(chart_path, artifact_path="plots")
        # Log best model info
        best = max(all_results.items(), key=lambda x: x[1]["f1_score"])
        mlflow.log_params({"best_model_by_f1": best[0]})
        mlflow.log_metrics({f"best_{k}": v for k, v in best[1].items()})
        mlflow.set_tags({"note": "production model is RandomForest — see train_model.py"})

    print(f"\nBest model by F1: {best[0]} (F1={best[1]['f1_score']:.4f})")
    print("NOTE: The production application continues to use the trained RandomForest.")
    print("Model comparison complete.")


if __name__ == "__main__":
    main()
