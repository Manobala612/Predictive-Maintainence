"""
SHAP Model Interpretability Script — Predictive Maintenance Project.

Purpose
-------
Generate SHAP (SHapley Additive exPlanations) values for the existing
production Random Forest to explain:
  - Which features drive machine failure predictions globally (feature importance)
  - How individual predictions are explained (local explanations)

This script uses the already-trained model artifacts from the model/ directory.
It does NOT retrain the model.

Outputs
-------
- mlflow_artifacts/shap_summary_bar.png   — global feature importance (bar)
- mlflow_artifacts/shap_summary_dot.png   — global feature importance (beeswarm)
- mlflow_artifacts/shap_waterfall.png     — single-prediction explanation
All artifacts are also logged to MLflow under Predictive_Maintenance_Training.

Run
---
    python model/shap_analysis.py

Requirements
------------
    pip install shap matplotlib
    (or: pip install -r model/requirements.txt)
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

try:
    import shap
except ImportError:
    sys.exit(
        "ERROR: shap is not installed. Run: pip install shap\n"
        "Or: pip install -r model/requirements.txt"
    )

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ── Constants ────────────────────────────────────────────────────────────────
ARTIFACT_DIR = "mlflow_artifacts"
MODEL_PATH = os.path.join("model", "machine_failure_model.pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders.pkl")
FEATURES_PATH = os.path.join("model", "feature_names.pkl")
DATASET_PATH = os.path.join("dataset", "ai4i2020.csv")
RANDOM_STATE = 7
TEST_SIZE = 0.2
# Number of test samples to compute SHAP for (full set is slow for tree ensembles)
SHAP_SAMPLE_SIZE = 200


def load_artifacts():
    """Load pre-trained model, encoders, and feature list from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun train_model.py first.")
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"Model loaded: {type(model).__name__}")
    print(f"Features: {feature_names}")
    return model, encoders, feature_names


def load_test_data(encoders, feature_names):
    """Re-create the test split used during training for SHAP analysis."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\nRun train_model.py first."
        )
    df = pd.read_csv(DATASET_PATH)
    for col in df.select_dtypes(include="object").columns:
        le = encoders.get(col)
        if le:
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    X = df.drop(["Machine failure", "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
    y = df["Machine failure"]

    _, X_test, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test[feature_names].reset_index(drop=True)


def run_shap(model, X_test_df: pd.DataFrame) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer (optimised for tree-based models).
    
    TreeExplainer is exact (not approximate) for Random Forest.
    It computes the contribution of each feature to each prediction.
    """
    print(f"Computing SHAP values for {min(SHAP_SAMPLE_SIZE, len(X_test_df))} samples...")
    X_sample = X_test_df.iloc[:SHAP_SAMPLE_SIZE]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    print("SHAP values computed.")
    return shap_values, X_sample


def plot_summary_bar(shap_values, feature_names: list, save_path: str) -> None:
    """
    Global feature importance: mean |SHAP| per feature across all samples.
    Shows which features matter most on average across the test set.
    """
    # shap_values[:, :, 1] selects class=1 (Failure) SHAP values
    sv_failure = shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values

    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs_shap = np.abs(sv_failure.values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx],
        color="steelblue"
    )
    ax.set_xlabel("Mean |SHAP Value| (average impact on failure probability)")
    ax.set_title("SHAP Feature Importance — Random Forest\n(Global, class=Failure)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"SHAP bar chart saved: {save_path}")


def plot_summary_beeswarm(shap_values, X_sample: pd.DataFrame, save_path: str) -> None:
    """
    Beeswarm plot: shows distribution of SHAP values per feature.
    High feature values = red dots; low = blue. Position = impact on prediction.
    """
    sv_failure = shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        sv_failure.values,
        X_sample,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP Beeswarm — Feature Impact on Failure Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"SHAP beeswarm saved: {save_path}")


def plot_waterfall(shap_values, X_sample: pd.DataFrame, sample_idx: int, save_path: str) -> None:
    """
    Waterfall plot for a single prediction: explains why the model predicted
    failure or no-failure for one specific data point.
    Each bar shows how much each feature pushed the prediction up or down.
    """
    sv_failure = shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(sv_failure[sample_idx], show=False)
    plt.title(f"SHAP Waterfall — Single Prediction Explanation (sample #{sample_idx})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"SHAP waterfall saved: {save_path}")


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ── Load artifacts ────────────────────────────────────────────────────────
    model, encoders, feature_names = load_artifacts()
    X_test_df = load_test_data(encoders, feature_names)
    print(f"Test set size: {len(X_test_df)}")

    # ── Compute SHAP values ───────────────────────────────────────────────────
    shap_values, X_sample = run_shap(model, X_test_df)

    # ── Save plots ────────────────────────────────────────────────────────────
    bar_path = os.path.join(ARTIFACT_DIR, "shap_summary_bar.png")
    bee_path = os.path.join(ARTIFACT_DIR, "shap_summary_dot.png")
    wf_path = os.path.join(ARTIFACT_DIR, "shap_waterfall.png")

    plot_summary_bar(shap_values, feature_names, bar_path)
    plot_summary_beeswarm(shap_values, X_sample, bee_path)
    # Use sample index 0 (first test sample) for waterfall
    plot_waterfall(shap_values, X_sample, 0, wf_path)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Predictive_Maintenance_Training")

    try:
        with mlflow.start_run(run_name="SHAP_Interpretability"):
            mlflow.log_artifact(bar_path, artifact_path="shap")
            mlflow.log_artifact(bee_path, artifact_path="shap")
            mlflow.log_artifact(wf_path, artifact_path="shap")
            mlflow.log_params({
                "shap_sample_size": SHAP_SAMPLE_SIZE,
                "explainer_type": "TreeExplainer",
                "model_type": type(model).__name__,
            })
            mlflow.set_tags({
                "task": "interpretability",
                "dataset": "AI4I_2020",
            })
        print("SHAP artifacts logged to MLflow under Predictive_Maintenance_Training.")
    except Exception as e:
        print(f"WARNING: MLflow logging failed — {e}")
        print("SHAP plots are still saved locally in mlflow_artifacts/")

    # ── Print top features ────────────────────────────────────────────────────
    sv_failure = shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values
    mean_abs = np.abs(sv_failure.values).mean(axis=0)
    print("\nTop features by mean |SHAP| (failure class):")
    for feat, score in sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<35} {score:.4f}")
    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    main()
