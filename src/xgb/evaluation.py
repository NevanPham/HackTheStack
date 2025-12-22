import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import warnings

warnings.filterwarnings("ignore")


def evaluate_model(
    model, X_train, X_test, y_train, y_test, train_sources, test_sources, threshold=0.5
):
    """
    Evaluate model overall and by source; return metrics and artifacts.

    Args:
        model: Trained XGBoost model.
        X_train (csr_matrix), X_test (csr_matrix)
        y_train (np.ndarray), y_test (np.ndarray)
        train_sources (array-like), test_sources (array-like)
        threshold (float): Decision threshold.

    Returns:
        dict: train/test metrics, per-source metrics, reports, predictions,
              probabilities, and confusion matrices.
    """
    print("Evaluating final XGBoost model...")
    print(f"Using threshold: {threshold:.4f}")

    # Prediction probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Predictions using custom threshold
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Calculate metrics
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1": f1_score(y_train, y_train_pred),
        "roc_auc": roc_auc_score(y_train, y_train_proba),
        "aucpr": average_precision_score(y_train, y_train_proba),
    }

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "aucpr": average_precision_score(y_test, y_test_proba),
    }

    # Print results
    print("\nTraining Set Performance:")
    for metric, value in train_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Performance by source - collect detailed metrics
    test_performance_by_source = {}
    print("\nTest Performance by Source:")
    for source in np.unique(test_sources):
        source_mask = test_sources == source
        source_y_true = y_test[source_mask]
        source_y_pred = y_test_pred[source_mask]
        source_y_proba = y_test_proba[source_mask]

        if len(source_y_true) > 0:
            source_metrics = {
                "accuracy": accuracy_score(source_y_true, source_y_pred),
                "precision": precision_score(source_y_true, source_y_pred),
                "recall": recall_score(source_y_true, source_y_pred),
                "f1": f1_score(source_y_true, source_y_pred),
                "roc_auc": roc_auc_score(source_y_true, source_y_proba),
                "sample_count": len(source_y_true),
                "spam_rate": float(np.mean(source_y_true)),
            }
            test_performance_by_source[source] = source_metrics

            print(
                f"  {source}: F1={source_metrics['f1']:.4f}, Accuracy={source_metrics['accuracy']:.4f} (n={source_metrics['sample_count']})"
            )

    # Training performance by source
    train_performance_by_source = {}
    print("\nTraining Performance by Source:")
    for source in np.unique(train_sources):
        source_mask = train_sources == source
        source_y_true = y_train[source_mask]
        source_y_pred = y_train_pred[source_mask]
        source_y_proba = y_train_proba[source_mask]

        if len(source_y_true) > 0:
            source_metrics = {
                "accuracy": accuracy_score(source_y_true, source_y_pred),
                "precision": precision_score(source_y_true, source_y_pred),
                "recall": recall_score(source_y_true, source_y_pred),
                "f1": f1_score(source_y_true, source_y_pred),
                "roc_auc": roc_auc_score(source_y_true, source_y_proba),
                "sample_count": len(source_y_true),
                "spam_rate": float(np.mean(source_y_true)),
            }
            train_performance_by_source[source] = source_metrics

            print(
                f"  {source}: F1={source_metrics['f1']:.4f}, Accuracy={source_metrics['accuracy']:.4f} (n={source_metrics['sample_count']})"
            )

    # Classification reports
    print("\nDetailed Test Set Classification Report:")
    test_classification_report = classification_report(
        y_test, y_test_pred, target_names=["Ham", "Spam"], output_dict=True
    )
    print(classification_report(y_test, y_test_pred, target_names=["Ham", "Spam"]))

    print("\nDetailed Training Set Classification Report:")
    train_classification_report = classification_report(
        y_train, y_train_pred, target_names=["Ham", "Spam"], output_dict=True
    )
    print(classification_report(y_train, y_train_pred, target_names=["Ham", "Spam"]))

    # Confusion matrices
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_performance_by_source": train_performance_by_source,
        "test_performance_by_source": test_performance_by_source,
        "train_classification_report": train_classification_report,
        "test_classification_report": test_classification_report,
        "train_predictions": y_train_pred,
        "test_predictions": y_test_pred,
        "train_probabilities": y_train_proba,
        "test_probabilities": y_test_proba,
        "train_confusion_matrix": train_cm,
        "test_confusion_matrix": test_cm,
    }


def analyze_feature_importance(model, metadata, top_n=30):
    """
    Return feature importance DataFrame (gain preferred; fallback to weight).

    Args:
        model: Trained XGBoost model.
        metadata (dict): Contains feature_names.
        top_n (int): Top features to print.

    Returns:
        pd.DataFrame: Columns ['feature','importance'] sorted desc.
    """
    print(f"Analyzing top {top_n} most important features...")

    # Get feature importances (XGBoost provides multiple importance types)
    importances_weight = model.feature_importances_  # Default: weight-based

    # Try to get gain-based importance (more meaningful)
    try:
        importances_gain = model.get_booster().get_score(importance_type="gain")
        feature_names = metadata["feature_names"]

        # Convert gain importance to array (matching feature order)
        gain_values = []
        for fname in feature_names:
            # XGBoost uses f0, f1, f2... as feature names internally
            xgb_fname = f"f{feature_names.index(fname)}"
            gain_values.append(importances_gain.get(xgb_fname, 0.0))

        importances_gain = np.array(gain_values)

        print("Using gain-based importance (preferred for XGBoost)")
        importances = importances_gain
    except:
        print("Using weight-based importance (fallback)")
        importances = importances_weight

    feature_names = metadata["feature_names"]

    # Create importance DataFrame
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # Display top features
    print(f"\nTop {top_n} most important features:")
    top_features = importance_df.head(top_n)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")

    # Analyze by feature type
    feature_type_importance = {}
    for feature_type in [
        "tfidf",
        "char_count",
        "word_count",
        "exclamation",
        "question",
        "upper",
        "url",
        "email",
        "phone",
        "money",
        "source",
    ]:
        mask = importance_df["feature"].str.contains(feature_type, case=False)
        if mask.any():
            total_importance = importance_df[mask]["importance"].sum()
            count = mask.sum()
            feature_type_importance[feature_type] = {
                "total_importance": total_importance,
                "avg_importance": total_importance / count,
                "count": count,
            }

    print(f"\nImportance by feature type:")
    for feature_type, stats in sorted(
        feature_type_importance.items(),
        key=lambda x: x[1]["total_importance"],
        reverse=True,
    ):
        print(
            f"  {feature_type}: total={stats['total_importance']:.6f}, "
            f"avg={stats['avg_importance']:.6f}, count={stats['count']}"
        )

    return importance_df


def create_evaluation_plots(
    evaluation_results, importance_df, y_test, output_dir="models/xgb"
):
    """
    Save figures: confusion matrix, top features, prob histogram, train-vs-test,
    type importance, and ROC curve.

    Args:
        evaluation_results (dict): From evaluate_model.
        importance_df (pd.DataFrame): Feature importance.
        y_test (np.ndarray): Test labels.
        output_dir (str): Directory to save.

    Returns:
        None

    Outputs:
        - models/xgb/xgb_evaluation_randomized.png
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("XGBoost Model Evaluation (Randomized Search)", fontsize=16)

    # 1. Confusion Matrix - Test Set
    ax1 = axes[0, 0]
    sns.heatmap(
        evaluation_results["test_confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    ax1.set_title("Test Set Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # 2. Feature Importance (Top 15)
    ax2 = axes[0, 1]
    top_15_features = importance_df.head(15)
    y_pos = np.arange(len(top_15_features))
    ax2.barh(y_pos, top_15_features["importance"])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_15_features["feature"], fontsize=8)
    ax2.set_xlabel("Feature Importance")
    ax2.set_title("Top 15 Feature Importances")
    ax2.invert_yaxis()

    # 3. Prediction Probability Distribution
    ax3 = axes[0, 2]
    ax3.hist(
        evaluation_results["test_probabilities"],
        bins=30,
        alpha=0.7,
        density=True,
        edgecolor="black",
    )
    ax3.set_title("Test Prediction Probability Distribution")
    ax3.set_xlabel("Predicted Probability (Spam)")
    ax3.set_ylabel("Density")
    ax3.axvline(
        x=0.5, color="red", linestyle="--", alpha=0.7, label="Decision Threshold"
    )
    ax3.legend()

    # 4. Metrics Comparison
    ax4 = axes[1, 0]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    train_values = [evaluation_results["train_metrics"][m] for m in metrics]
    test_values = [evaluation_results["test_metrics"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(
        x - width / 2, train_values, width, label="Train", alpha=0.8, color="skyblue"
    )
    ax4.bar(
        x + width / 2, test_values, width, label="Test", alpha=0.8, color="lightcoral"
    )
    ax4.set_xlabel("Metrics")
    ax4.set_ylabel("Score")
    ax4.set_title("Train vs Test Performance")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=45)
    ax4.legend()
    ax4.set_ylim(0, 1)

    # 5. Feature Type Importance
    ax5 = axes[1, 1]
    feature_types = ["tfidf", "exclamation", "question", "upper", "url", "source"]
    type_importances = []
    for ftype in feature_types:
        mask = importance_df["feature"].str.contains(ftype, case=False)
        type_importances.append(importance_df[mask]["importance"].sum())

    ax5.bar(feature_types, type_importances, color="lightgreen", alpha=0.8)
    ax5.set_title("Feature Importance by Type")
    ax5.set_xlabel("Feature Type")
    ax5.set_ylabel("Total Importance")
    ax5.tick_params(axis="x", rotation=45)

    # 6. ROC Curve - FIXED: Use true labels instead of predictions
    ax6 = axes[1, 2]
    fpr, tpr, _ = roc_curve(y_test, evaluation_results["test_probabilities"])
    roc_auc = evaluation_results["test_metrics"]["roc_auc"]

    ax6.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    ax6.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax6.set_xlim([0.0, 1.0])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel("False Positive Rate")
    ax6.set_ylabel("True Positive Rate")
    ax6.set_title("ROC Curve")
    ax6.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(
        output_path / "xgb_evaluation_randomized.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Evaluation plots saved to: {output_path / 'xgb_evaluation_randomized.png'}")
