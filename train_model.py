"""
train_model.py — FastGuard Breaking-Change Classifier
=====================================================

End-to-end pipeline that trains, evaluates, and serialises a machine-learning
model to detect breaking API-contract changes in FastAPI applications.

Pipeline stages:
  1. Load pre-split train / test CSVs (NO in-memory re-splitting).
  2. Select an optimal decision threshold via 5-fold stratified OOF
     probabilities on the training set (maximising Macro F1).
  3. Train RandomForest (class_weight="balanced") and GradientBoosting
     (explicit sample_weight) on the full training set.
  4. Evaluate both models ONCE on the real holdout test set using the
     frozen threshold.
  5. Save feature-importance plot, confusion-matrix plot, and the best
     model (by PR-AUC) to disk.

Usage:
    python train_model.py

Outputs:
    - fastguard_model.pkl        (best model by PR-AUC)
    - feature_importance.png     (Random Forest AST feature importances)
    - confusion_matrix.png       (best model confusion matrix on test set)
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ============================================================================
# Constants
# ============================================================================

# Strictly structural AST features — intentionally excludes churn / complexity
# metrics (lines_added, lines_deleted, net_churn, code_complexity) to prevent
# data leakage from surface-level edit statistics.
AST_FEATURE_COLS: list[str] = [
    "fields_removed_count",
    "fields_added_count",
    "fields_type_changed_count",
    "fields_default_changed_count",
    "models_renamed_count",
    "routes_removed_count",
    "routes_added_count",
    "routes_param_changed_count",
    "routes_param_type_changed_count",
    "routes_param_default_changed_count",
]

LABEL_COL: str = "is_breaking_commit"

TRAIN_CSV: str = "train_dataset.csv"
TEST_CSV: str = "test_dataset.csv"

MODEL_OUTPUT: str = "fastguard_model.pkl"
FEATURE_IMPORTANCE_PNG: str = "feature_importance.png"
CONFUSION_MATRIX_PNG: str = "confusion_matrix.png"

# OOF threshold search grid
THRESHOLD_LOW: float = 0.01
THRESHOLD_HIGH: float = 0.50
THRESHOLD_STEPS: int = 100

# Cross-validation folds
CV_SPLITS: int = 5
RANDOM_STATE: int = 42


# ============================================================================
# 1. Dataset Ingestion & Strict Isolation
# ============================================================================

def load_datasets(
    train_path: str = TRAIN_CSV,
    test_path: str = TEST_CSV,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load pre-split train / test CSVs and extract AST-only features.

    Validates that every expected AST column is present in both files.
    Returns (X_train, y_train, X_test, y_test).
    """
    print("=" * 60)
    print("STEP 1: LOAD TRAIN & TEST DATASETS SEPARATELY")
    print("=" * 60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Validate that all expected AST features are present
    missing_train = set(AST_FEATURE_COLS) - set(train_df.columns)
    missing_test = set(AST_FEATURE_COLS) - set(test_df.columns)
    if missing_train:
        sys.exit(f"FATAL: Missing columns in {train_path}: {missing_train}")
    if missing_test:
        sys.exit(f"FATAL: Missing columns in {test_path}: {missing_test}")

    # Use only the 10 structural AST features (no churn / complexity leakage)
    feature_cols = [c for c in AST_FEATURE_COLS if c in train_df.columns]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[LABEL_COL].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[LABEL_COL].copy()

    print(f"Train Set : {X_train.shape[0]:,} samples | {X_train.shape[1]} AST features")
    print(f"  Class balance: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Test  Set : {X_test.shape[0]:,} samples  (REAL HOLDOUT — evaluated ONCE)")
    print(f"  Class balance: {y_test.value_counts(normalize=True).to_dict()}")
    print()

    return X_train, y_train, X_test, y_test


# ============================================================================
# 2. Out-of-Fold (OOF) Threshold Selection
# ============================================================================

def _oof_probabilities(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    *,
    needs_sample_weight: bool = False,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Generate out-of-fold predicted probabilities via cross_val_predict.

    Handles the scikit-learn API change where `fit_params` was renamed
    to `params` in newer releases (>=1.4).
    """
    sig = inspect.signature(cross_val_predict)
    use_params_kwarg = "params" in sig.parameters

    if needs_sample_weight and sample_weight is not None:
        weight_dict = {"sample_weight": sample_weight}
        if use_params_kwarg:
            oof = cross_val_predict(
                model, X, y, cv=cv,
                method="predict_proba",
                params=weight_dict,
                n_jobs=-1,
            )
        else:
            oof = cross_val_predict(
                model, X, y, cv=cv,
                method="predict_proba",
                fit_params=weight_dict,
                n_jobs=-1,
            )
    else:
        oof = cross_val_predict(
            model, X, y, cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )

    return oof[:, 1]  # probability of the positive (breaking) class


def select_threshold(
    oof_probs: np.ndarray,
    y_true: pd.Series,
    low: float = THRESHOLD_LOW,
    high: float = THRESHOLD_HIGH,
    steps: int = THRESHOLD_STEPS,
) -> tuple[float, float]:
    """Sweep decision thresholds over OOF probabilities and return the
    threshold that maximises the Macro F1 score on the training set.

    Returns (best_threshold, best_macro_f1).
    """
    best_thresh = 0.5
    best_f1 = -1.0

    for t in np.linspace(low, high, steps):
        preds = (oof_probs >= t).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    return best_thresh, best_f1


# ============================================================================
# 3. Model Training & Cost-Sensitive Weighting
# ============================================================================

def build_models() -> dict[str, Any]:
    """Construct the two candidate classifiers with their hyper-parameters."""
    return {
        "Random Forest (AST Only)": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting (AST Only)": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
    }


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """Compute inverse-frequency sample weights so the positive class
    (breaking) receives ``neg_count / pos_count`` weight per sample."""
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    return np.where(y == 1, pos_weight, 1.0)


# ============================================================================
# 4. Evaluation on Real Holdout Set
# ============================================================================

def evaluate_model(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a trained model ONCE on the real holdout test set using
    the pre-selected frozen threshold.

    Prints the full classification report & confusion matrix and returns
    a metrics dict.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Rank-based (threshold-independent) metrics
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val = auc(recalls, precisions)
    roc_auc_val = roc_auc_score(y_test, y_prob)

    # Threshold-dependent metrics
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    prec_break = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_break = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    # --- Console output ---
    print(f"\n{'—' * 50}")
    print(f"Classification Report — {name} (threshold={threshold:.4f})")
    print("—" * 50)
    print(classification_report(
        y_test, y_pred,
        target_names=["Non-Breaking", "Breaking"],
        zero_division=0,
    ))
    print(f"Confusion matrix ([[TN, FP], [FN, TP]]):\n{cm}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Precision(Breaking): {prec_break:.4f}")
    print(f"  Recall(Breaking)   : {rec_break:.4f}")
    print(f"  Macro F1           : {macro_f1:.4f}")
    print(f"  PR-AUC             : {pr_auc_val:.4f}")
    print(f"  ROC-AUC            : {roc_auc_val:.4f}")

    return {
        "Model": name,
        "Threshold": round(threshold, 4),
        "Accuracy": round(acc, 4),
        "Precision (Breaking)": round(prec_break, 4),
        "Recall (Breaking)": round(rec_break, 4),
        "Macro F1": round(macro_f1, 4),
        "PR-AUC": round(pr_auc_val, 4),
        "ROC-AUC": round(roc_auc_val, 4),
    }


# ============================================================================
# 5. Feature Importance & Artifact Saving
# ============================================================================

def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    save_path: str = FEATURE_IMPORTANCE_PNG,
) -> pd.DataFrame:
    """Calculate, print, and plot Random Forest feature importances.

    Returns the importance DataFrame sorted descending.
    """
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    print("\n" + "=" * 60)
    print("PURE AST FEATURE IMPORTANCES (Random Forest)")
    print("=" * 60)
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        y="Feature",
        x="Importance",
        hue="Feature",
        legend=False,
        palette="viridis",
    )
    plt.title("Pure AST Feature Importances — FastGuard Breaking-Change Detection")
    plt.xlabel("Gini Importance")
    plt.ylabel("AST Feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nSaved {save_path}")

    return importance_df


def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    model_name: str,
    save_path: str = CONFUSION_MATRIX_PNG,
) -> None:
    """Plot and save the confusion matrix for the best model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Non-Breaking", "Breaking"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix — {model_name} (thresh={threshold:.4f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def save_best_model(
    results: list[dict],
    trained_models: dict[str, Any],
    selected_thresholds: dict[str, float],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: str = MODEL_OUTPUT,
) -> None:
    """Select the model with highest PR-AUC, save it, and plot its
    confusion matrix."""
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("LEAKAGE-FREE PERFORMANCE COMPARISON (Real Holdout Set)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_idx = results_df["PR-AUC"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    best_model = trained_models[best_name]
    best_thresh = selected_thresholds[best_name]

    print(f"\n→ Best model by PR-AUC: {best_name} "
          f"(PR-AUC={results_df.loc[best_idx, 'PR-AUC']:.4f})")

    # Confusion matrix for best model
    plot_confusion_matrix(
        best_model, X_test, y_test,
        best_thresh, best_name,
    )

    # Serialise best model
    joblib.dump(best_model, model_path)
    print(f"Saved best trained model → {model_path}")


# ============================================================================
# Main Orchestrator
# ============================================================================

def main() -> None:
    """Execute the full train → evaluate → save pipeline."""

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    X_train, y_train, X_test, y_test = load_datasets()
    feature_cols = list(X_train.columns)

    # ------------------------------------------------------------------
    # 2. Prepare cost-sensitive weighting for Gradient Boosting
    # ------------------------------------------------------------------
    sample_weight_train = compute_sample_weights(y_train)

    # ------------------------------------------------------------------
    # 3. Build models & cross-validation splitter
    # ------------------------------------------------------------------
    models = build_models()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results: list[dict] = []
    trained_models: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 4. Per-model: OOF threshold → fit → evaluate
    # ------------------------------------------------------------------
    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"STEP 2–4: PROCESSING — {name}")
        print("=" * 60)

        # --- 2a. Generate Out-Of-Fold probabilities (train only) ---
        is_gb = "Gradient Boosting" in name
        print("Generating OOF predictions on Train Set for threshold selection …")

        oof_probs = _oof_probabilities(
            model, X_train, y_train, cv,
            needs_sample_weight=is_gb,
            sample_weight=sample_weight_train if is_gb else None,
        )

        # --- 2b. Sweep thresholds using OOF probabilities ---
        best_thresh, best_oof_f1 = select_threshold(oof_probs, y_train)
        print(f"Selected Threshold (OOF Macro F1): {best_thresh:.4f}  "
              f"(OOF Macro F1 = {best_oof_f1:.4f})")
        selected_thresholds[name] = best_thresh

        # --- 3. Fit model on the *complete* training set ---
        print(f"Fitting {name} on full training set ({X_train.shape[0]:,} samples) …")
        if is_gb:
            model.fit(X_train, y_train, sample_weight=sample_weight_train)
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model

        # --- 4. Evaluate ONCE on real holdout test set (frozen threshold) ---
        metrics = evaluate_model(name, model, X_test, y_test, best_thresh)
        results.append(metrics)

    # ------------------------------------------------------------------
    # 5. Feature importance (Random Forest only)
    # ------------------------------------------------------------------
    rf_model = trained_models["Random Forest (AST Only)"]
    plot_feature_importance(rf_model, feature_cols)

    # ------------------------------------------------------------------
    # 6. Select & save best model; plot confusion matrix
    # ------------------------------------------------------------------
    save_best_model(
        results, trained_models, selected_thresholds,
        X_test, y_test,
    )

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()