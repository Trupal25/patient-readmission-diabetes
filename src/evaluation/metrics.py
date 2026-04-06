import logging
from typing import Protocol, TypeAlias, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from src.features.pipeline import get_processed_data
from src.utils.config import MODELS_DIR, FIGURES_DIR, METRICS_DIR

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
MetricValues: TypeAlias = dict[str, float]
EvaluationMetrics: TypeAlias = dict[str, str | float]


class SupportsPredictProba(Protocol):
    def predict_proba(self, X: object) -> FloatArray: ...


def compute_optimal_threshold(
    y_true: IntArray,
    y_prob: FloatArray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> float:
    """
    Select a probability threshold by minimizing expected misclassification burden.

    The operating point is chosen from the observed score cutoffs plus a value
    strictly greater than `1.0`, which lets the rule select an "alert nobody"
    policy even when some predicted probabilities equal `1.0`.
    """
    candidate_thresholds = np.unique(np.asarray(y_prob, dtype=np.float64))
    candidate_thresholds = np.append(candidate_thresholds, np.nextafter(1.0, 2.0))
    best_threshold = float(candidate_thresholds[0])
    best_cost = float("inf")

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(np.int_)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        cost = false_negative_cost * fn + false_positive_cost * fp

        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(threshold)

    return best_threshold


def evaluate_predictions(
    y_true: IntArray, y_prob: FloatArray, model_name: str
) -> tuple[EvaluationMetrics, IntArray]:
    """Calculates AUROC, AUPRC, Brier, F1, Sens, Spec, PPV, NPV at optimal threshold."""
    # 1. Base metrics
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    # 2. Find the operating threshold using the same asymmetric clinical cost rule
    optimal_threshold = compute_optimal_threshold(y_true, y_prob)
    
    # Convert to binary predictions using the optimal threshold
    y_pred: IntArray = (y_prob >= optimal_threshold).astype(np.int_)
    
    # 3. Compute clinical metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    
    numeric_metrics: MetricValues = {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Brier": float(brier),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "Opt_Threshold": float(optimal_threshold),
    }
    metrics: EvaluationMetrics = {
        "Model": model_name,
        **numeric_metrics,
    }
    
    logger.info(f"--- {model_name} Test Metrics ---")
    for k, v in numeric_metrics.items():
        logger.info(f"{k}: {v:.4f}")
            
    return metrics, y_pred

def plot_roc_prc(y_true: IntArray, lr_prob: FloatArray, xgb_prob: FloatArray) -> None:
    """Generates comparative ROC and Precision-Recall curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC Curve
    for name, prob in [("Logistic Regression", lr_prob), ("XGBoost", xgb_prob)]:
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_val = roc_auc_score(y_true, prob)
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    
    ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    
    # PR Curve
    for name, prob in [("Logistic Regression", lr_prob), ("XGBoost", xgb_prob)]:
        precision, recall, _ = precision_recall_curve(y_true, prob)
        prc_val = average_precision_score(y_true, prob)
        ax2.plot(recall, precision, lw=2, label=f"{name} (AUC = {prc_val:.3f})")
        
    baseline_pr = y_true.mean()
    ax2.axhline(baseline_pr, color="gray", lw=1, linestyle="--", label=f"Random (Prev = {baseline_pr:.3f})")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax2.set_ylabel("Precision (PPV)", fontsize=12)
    ax2.set_title("Precision-Recall Curve (PRC)", fontsize=14, fontweight='bold')
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_prc_comparison.png", dpi=150)
    plt.close()

def plot_calibration(y_true: IntArray, lr_prob: FloatArray, xgb_prob: FloatArray) -> None:
    """Plots the calibration curve (reliability diagram)."""
    plt.figure(figsize=(8, 8))
    
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for name, prob in [("Logistic Regression", lr_prob), ("XGBoost", xgb_prob)]:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, prob, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)
        ax2.hist(prob, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
        
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration Plots (Reliability Curve)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=150)
    plt.close()

def run_evaluation_suite() -> None:
    logger.info("Running complete evaluation suite on the held-out TEST set...")
    
    # Load Models
    lr_model = cast(
        SupportsPredictProba, joblib.load(MODELS_DIR / "logistic_baseline.joblib")
    )
    # Actually, try loading optimized first, fallback to initial
    xgb_path = MODELS_DIR / "xgboost_optimized.joblib"
    if not xgb_path.exists():
        xgb_path = MODELS_DIR / "xgboost_initial.joblib"
    xgb_model = cast(SupportsPredictProba, joblib.load(xgb_path))
    
    # Load separate pipelines. Both pipelines will result in identical y_test because we use fixed seed
    # and identical splits before the data is transformed. Let's verify y_test matches.
    _, _, X_test_lr, _, _, y_test_lr, _, _ = get_processed_data("lr")
    _, _, X_test_xgb, _, _, y_test_xgb, _, _ = get_processed_data("xgb")
    
    assert np.array_equal(y_test_lr.to_numpy(), y_test_xgb.to_numpy()), (
        "Test sets for LR and XGB do not align!"
    )
    y_test: IntArray = y_test_lr.to_numpy(dtype=np.int_)
    
    # Get probabilities
    lr_prob: FloatArray = np.asarray(lr_model.predict_proba(X_test_lr)[:, 1], dtype=np.float64)
    xgb_prob: FloatArray = np.asarray(xgb_model.predict_proba(X_test_xgb)[:, 1], dtype=np.float64)
    
    # Evaluate
    logger.info("\nEvaluating Logistic Regression")
    lr_metrics, lr_pred = evaluate_predictions(y_test, lr_prob, "Logistic Regression")
    
    logger.info("\nEvaluating XGBoost")
    xgb_metrics, xgb_pred = evaluate_predictions(y_test, xgb_prob, "XGBoost (Tuned)")
    
    # Save comparison table
    df_metrics = pd.DataFrame([lr_metrics, xgb_metrics])
    df_metrics.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    logger.info(f"Saved metric comparison to {METRICS_DIR / 'model_comparison.csv'}")
    
    # Plotting
    logger.info("Generating ROC/PRC curves...")
    plot_roc_prc(y_test, lr_prob, xgb_prob)
    
    logger.info("Generating Calibration curve...")
    plot_calibration(y_test, lr_prob, xgb_prob)
    
    # Generate Confusion matrix plot for the better model (XGBoost)
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, xgb_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('XGBoost Confusion Matrix (Optimal Threshold)')
    plt.savefig(FIGURES_DIR / "xgboost_confusion_matrix.png", dpi=150)
    plt.close()
    
    logger.info("Evaluation complete. All artifacts saved to reports/")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_evaluation_suite()
