import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve


def evaluate_model_with_sources(model, X_test, y_test, test_sources, device="cuda"):
    """
    Evaluate model overall and by source, returning metrics and artifacts.

    Args:
        model (torch.nn.Module): Trained model.
        X_test (torch.Tensor): Test inputs.
        y_test (torch.Tensor): Test labels.
        test_sources (array-like): Source per sample.
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: test_metrics, per-source metrics, classification report, predictions,
              probabilities, and confusion matrix.
    """
    print("Evaluating final LSTM model...")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    # Create test data loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get predictions
    test_predictions = []
    test_probabilities = []
    test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)

            test_probabilities.extend(outputs.cpu().numpy())
            test_predictions.extend((outputs > 0.5).float().cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())

    # Convert to numpy arrays
    test_predictions = np.array(test_predictions)
    test_probabilities = np.array(test_probabilities)
    test_targets = np.array(test_targets)

    # Calculate overall metrics
    test_metrics = {
        "accuracy": accuracy_score(test_targets, test_predictions),
        "precision": precision_score(test_targets, test_predictions),
        "recall": recall_score(test_targets, test_predictions),
        "f1": f1_score(test_targets, test_predictions),
        "roc_auc": roc_auc_score(test_targets, test_probabilities),
    }

    # Print overall results
    print("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Performance by source - collect detailed metrics
    test_performance_by_source = {}
    print("\nTest Performance by Source:")
    for source in np.unique(test_sources):
        source_mask = test_sources == source
        source_y_true = test_targets[source_mask]
        source_y_pred = test_predictions[source_mask]
        source_y_proba = test_probabilities[source_mask]

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

    # Classification report
    print("\nDetailed Test Set Classification Report:")
    test_classification_report = classification_report(
        test_targets, test_predictions, target_names=["Ham", "Spam"], output_dict=True
    )
    print(
        classification_report(
            test_targets, test_predictions, target_names=["Ham", "Spam"]
        )
    )

    # Confusion matrix
    test_cm = confusion_matrix(test_targets, test_predictions)

    return {
        "test_metrics": test_metrics,
        "test_performance_by_source": test_performance_by_source,
        "test_classification_report": test_classification_report,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "test_confusion_matrix": test_cm,
    }


def create_lstm_evaluation_plots(evaluation_results, y_test, output_dir="models/lstm"):
    """
    Generate confusion matrix, per-source bars, probability histogram, and ROC curve.

    Args:
        evaluation_results (dict): Output from evaluate_model_with_sources.
        y_test (torch.Tensor): Test labels.
        output_dir (str): Directory to save plots.

    Returns:
        None

    Outputs:
        - models/lstm/lstm_evaluation.png
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("LSTM Model Evaluation", fontsize=16)

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

    # 2. Performance by Source
    ax2 = axes[0, 1]
    sources = list(evaluation_results["test_performance_by_source"].keys())
    f1_scores = [
        evaluation_results["test_performance_by_source"][s]["f1"] for s in sources
    ]
    acc_scores = [
        evaluation_results["test_performance_by_source"][s]["accuracy"] for s in sources
    ]

    x = np.arange(len(sources))
    width = 0.35
    ax2.bar(
        x - width / 2, f1_scores, width, label="F1 Score", alpha=0.8, color="skyblue"
    )
    ax2.bar(
        x + width / 2,
        acc_scores,
        width,
        label="Accuracy",
        alpha=0.8,
        color="lightcoral",
    )
    ax2.set_xlabel("Data Source")
    ax2.set_ylabel("Score")
    ax2.set_title("Performance by Source")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sources, rotation=45)
    ax2.legend()
    ax2.set_ylim(0, 1)

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

    # 4. Overall Metrics
    ax4 = axes[1, 0]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [evaluation_results["test_metrics"][m] for m in metrics]

    ax4.bar(metrics, values, color="lightgreen", alpha=0.8)
    ax4.set_xlabel("Metrics")
    ax4.set_ylabel("Score")
    ax4.set_title("Test Set Performance")
    ax4.tick_params(axis="x", rotation=45)
    ax4.set_ylim(0, 1)

    # 5. Sample Count by Source
    ax5 = axes[1, 1]
    sample_counts = [
        evaluation_results["test_performance_by_source"][s]["sample_count"]
        for s in sources
    ]
    spam_rates = [
        evaluation_results["test_performance_by_source"][s]["spam_rate"]
        for s in sources
    ]

    bars = ax5.bar(sources, sample_counts, color="orange", alpha=0.8)
    ax5.set_xlabel("Data Source")
    ax5.set_ylabel("Sample Count")
    ax5.set_title("Sample Distribution by Source")
    ax5.tick_params(axis="x", rotation=45)

    # Add spam rate annotations on bars
    for bar, spam_rate in zip(bars, spam_rates):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"Spam: {spam_rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 6. ROC Curve
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
    plt.savefig(output_path / "lstm_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"LSTM evaluation plots saved to: {output_path / 'lstm_evaluation.png'}")
