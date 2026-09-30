"""
Model Training Script with Full MLflow Experiment Tracking.

This script trains a Random Forest Classifier on the AI4I 2020 Predictive
Maintenance Dataset and tracks the entire experiment using MLflow.

MLflow Tracking Responsibilities:
  - Parameters: hyperparameters and split settings
  - Metrics:    accuracy, precision, recall, F1, ROC AUC
  - Artifacts:  confusion matrix plot, feature importance plot, ROC curve plot
  - Model:      registered in the MLflow Model Registry for versioning

This is TRAINING experiment tracking. It is distinct from inference logging
(done in backend/services/prediction.py), which records live API predictions.
"""

import os
import io
import sys
import urllib.request

# Fix: MLflow 3.x prints a emoji (🏃) when closing a run.
# Windows terminals use cp1252 by default which cannot encode it.
# Reconfigure stdout to UTF-8 so the script exits cleanly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required in containers/scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters — centralised here so they match exactly what MLflow logs
# ──────────────────────────────────────────────────────────────────────────────
N_ESTIMATORS = 100
RANDOM_STATE = 7
CLASS_WEIGHT = "balanced"
TEST_SIZE = 0.2
STRATIFY = True


def _plot_confusion_matrix(cm, save_path: str) -> None:
    """Saves a confusion matrix heatmap to save_path."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    labels = ["No Failure", "Failure"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def _plot_feature_importance(model, feature_names, save_path: str) -> None:
    """Saves a horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_importance[::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def _plot_roc_curve(y_test, y_prob, auc_score: float, save_path: str) -> None:
    """Saves a ROC curve plot with AUC score annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def main():
    # ── 0. Directory Setup ─────────────────────────────────────────────────────
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("mlflow_artifacts", exist_ok=True)

    dataset_path = os.path.join("dataset", "ai4i2020.csv")

    # ── 1. Dataset Acquisition ─────────────────────────────────────────────────
    if not os.path.exists(dataset_path):
        print("Downloading AI4I 2020 Predictive Maintenance dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        urllib.request.urlretrieve(url, dataset_path)
        print("Download complete.")

    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── 2. Preprocessing ───────────────────────────────────────────────────────
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(["Machine failure", "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
    y = df["Machine failure"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None,
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── 3. MLflow Experiment Setup ─────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Predictive_Maintenance_Training")
    print(f"MLflow tracking URI: {tracking_uri}")

    # ── 4. Training + Full MLflow Run ──────────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest_Training_v1") as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        # ── 4a. Log Hyperparameters ────────────────────────────────────────────
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "class_weight": CLASS_WEIGHT,
            "test_size": TEST_SIZE,
            "stratify": STRATIFY,
            "n_features": len(feature_names),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        })

        # ── 4b. Train Model ────────────────────────────────────────────────────
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight=CLASS_WEIGHT,
        )
        model.fit(X_train, y_train)

        # ── 4c. Evaluate ───────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

        # ── 4d. Log Metrics ────────────────────────────────────────────────────
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        })

        # ── 4e. Log Tags ───────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type": "RandomForestClassifier",
            "dataset": "AI4I_2020_Predictive_Maintenance",
            "task": "binary_classification",
        })

        # ── 4f. Confusion Matrix Artifact ──────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join("mlflow_artifacts", "confusion_matrix.png")
        _plot_confusion_matrix(cm, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        print(f"Confusion matrix saved and logged.")

        # ── 4g. Feature Importance Artifact ───────────────────────────────────
        fi_path = os.path.join("mlflow_artifacts", "feature_importance.png")
        _plot_feature_importance(model, feature_names, fi_path)
        mlflow.log_artifact(fi_path, artifact_path="plots")
        print("Feature importance plot saved and logged.")

        # ── 4h. ROC Curve Artifact ─────────────────────────────────────────────
        roc_path = os.path.join("mlflow_artifacts", "roc_curve.png")
        _plot_roc_curve(y_test, y_prob, roc_auc, roc_path)
        mlflow.log_artifact(roc_path, artifact_path="plots")
        print("ROC curve saved and logged.")

        # ── 4i. Save and Log Pickle Artifacts ──────────────────────────────────
        model_pkl   = os.path.join("model", "machine_failure_model.pkl")
        encoder_pkl = os.path.join("model", "label_encoders.pkl")
        features_pkl = os.path.join("model", "feature_names.pkl")

        joblib.dump(model, model_pkl)
        joblib.dump(encoders, encoder_pkl)
        joblib.dump(feature_names, features_pkl)

        mlflow.log_artifact(model_pkl, artifact_path="model_files")
        mlflow.log_artifact(encoder_pkl, artifact_path="model_files")
        mlflow.log_artifact(features_pkl, artifact_path="model_files")
        print("Model artifact files logged to MLflow.")

        # ── 4j. Register Model in MLflow Model Registry ────────────────────────
        # This registers the sklearn model so it can be loaded via mlflow.sklearn.load_model()
        # and promoted through Staging → Production stages in the UI.
        model_uri = f"runs:/{run.info.run_id}/sklearn_model"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            registered_model_name="RandomForest_MachineFail",
        )
        print(f"Model registered in MLflow registry as 'RandomForest_MachineFail'.")

    print("\nAll artifacts generated successfully in model/ directory.")
    print("MLflow training run complete.")


if __name__ == "__main__":
    main()
