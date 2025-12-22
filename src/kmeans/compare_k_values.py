"""
Compare K-Means results across k values and configurations.

Generates combined figures for clustering and classification metrics across
(k in 2..8) and across configuration variants at fixed k.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results_for_all_k(
    base_dir="models/kmeans", config_name="current", k_values=range(2, 9)
):
    """
    Load results JSON for a single config across multiple k values.

    Args:
        base_dir (str): Base models directory.
        config_name (str): Configuration name.
        k_values (Iterable[int]): K values to load.

    Returns:
        dict[int, dict]: Mapping k→results.
    """
    results = {}

    print(f"Loading results for configuration: {config_name}")
    print("=" * 60)

    for k in k_values:
        json_path = Path(base_dir) / f"k{k}" / config_name / "clustering_results.json"

        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                results[k] = data
                print(f"[OK] Loaded k={k}")
            except Exception as e:
                print(f"[ERROR] Error loading k={k}: {e}")
        else:
            print(f"[MISSING] File not found for k={k}: {json_path}")

    print(f"\nSuccessfully loaded {len(results)} out of {len(list(k_values))} k values")
    return results


def load_results_for_all_configs(
    base_dir="models/kmeans", k_value=2, config_names=None
):
    """
    Load results JSON for multiple configs at a fixed k.

    Args:
        base_dir (str): Base models directory.
        k_value (int): Specific k.
        config_names (list[str]|None): Config names.

    Returns:
        dict[str, dict]: Mapping config→results.
    """
    if config_names is None:
        config_names = [
            "current",
            "tfidf_50",
            "tfidf_100",
            "tfidf_1000",
            "source_0.1",
            "source_0.3",
            "source_0.5",
        ]

    results = {}

    print(f"Loading results for k={k_value} across all configurations")
    print("=" * 60)

    for config in config_names:
        json_path = Path(base_dir) / f"k{k_value}" / config / "clustering_results.json"

        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                results[config] = data
                print(f"[OK] Loaded {config}")
            except Exception as e:
                print(f"[ERROR] Error loading {config}: {e}")
        else:
            print(f"[MISSING] File not found for {config}: {json_path}")

    print(
        f"\nSuccessfully loaded {len(results)} out of {len(config_names)} configurations"
    )
    return results


def plot_combined_clustering_metrics(
    x_values, x_label, data_dict, output_path, title_prefix=""
):
    """
    Plot silhouette, ARI, and NMI in a single 1x3 figure.

    Args:
        x_values (list): X-axis values (k or config names).
        x_label (str): X-axis label.
        data_dict (dict): {'metric': {'train': [..], 'test': [..]}, ...}.
        output_path (Path): Where to save PNG.
        title_prefix (str): Optional prefix for subplot titles.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3)

    metrics = ["silhouette_score", "adjusted_rand_index", "normalized_mutual_info"]
    titles = ["Silhouette Score", "Adjusted Rand Index", "Normalized Mutual Info"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # Extract train and test values
        train_values = [data_dict[metric]["train"][i] for i in range(len(x_values))]
        test_values = [data_dict[metric]["test"][i] for i in range(len(x_values))]

        # Plot
        ax.plot(
            x_values,
            train_values,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Train",
            color="blue",
        )
        ax.plot(
            x_values,
            test_values,
            marker="s",
            linewidth=2,
            markersize=8,
            label="Test",
            color="red",
        )

        # Formatting
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title_prefix}{title}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Rotate x labels if they are config names
        if isinstance(x_values[0], str):
            ax.tick_params(axis="x", rotation=45)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path.name}")


def plot_combined_classification_metrics(
    x_values, x_label, data_dict, output_path, title_prefix=""
):
    """
    Plot accuracy/precision/recall/F1/ROC-AUC/AvgPrecision on one chart.

    Args:
        x_values (list): X-axis values (k or config names).
        x_label (str): X-axis label.
        data_dict (dict): {metric_name: [values...] }.
        output_path (Path): Where to save PNG.
        title_prefix (str): Optional title prefix.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ]
    titles = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "Average Precision",
    ]
    colors = ["blue", "green", "red", "purple", "orange", "brown"]
    markers = ["o", "s", "^", "D", "v", "p"]

    for idx, (metric, title, color, marker) in enumerate(
        zip(metrics, titles, colors, markers)
    ):
        values = data_dict[metric]

        # Plot each metric as a separate line
        ax.plot(
            x_values,
            values,
            marker=marker,
            linewidth=2,
            markersize=8,
            color=color,
            label=title,
        )

    # Formatting
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(
        f"{title_prefix}Classification Metrics Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # Rotate x labels if they are config names
    if isinstance(x_values[0], str):
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path.name}")


def extract_metrics_across_k(results):
    """
    Build metric arrays across k values.

    Args:
        results (dict[int, dict]): Mapping k→results.

    Returns:
        tuple: (clustering_metrics, classification_metrics, k_values)
    """
    clustering_metrics = {
        "silhouette_score": {"train": [], "test": []},
        "adjusted_rand_index": {"train": [], "test": []},
        "normalized_mutual_info": {"train": [], "test": []},
    }

    classification_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
        "average_precision": [],
    }

    k_values = sorted(results.keys())

    for k in k_values:
        # Clustering metrics
        clustering_perf = results[k]["clustering_performance"]
        for metric_name in clustering_metrics.keys():
            clustering_metrics[metric_name]["train"].append(
                clustering_perf[metric_name]["train"]
            )
            clustering_metrics[metric_name]["test"].append(
                clustering_perf[metric_name]["test"]
            )

        # Classification metrics
        class_perf = results[k]["classification_performance"]
        for metric_name in classification_metrics.keys():
            classification_metrics[metric_name].append(class_perf[metric_name])

    return clustering_metrics, classification_metrics, k_values


def extract_metrics_across_configs(results, config_names):
    """
    Build metric arrays across configurations for a fixed k.

    Args:
        results (dict[str, dict]): Mapping config→results.
        config_names (list[str]): Ordered config names.

    Returns:
        tuple: (clustering_metrics, classification_metrics)
    """
    clustering_metrics = {
        "silhouette_score": {"train": [], "test": []},
        "adjusted_rand_index": {"train": [], "test": []},
        "normalized_mutual_info": {"train": [], "test": []},
    }

    classification_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
        "average_precision": [],
    }

    for config in config_names:
        if config in results:
            # Clustering metrics
            clustering_perf = results[config]["clustering_performance"]
            for metric_name in clustering_metrics.keys():
                clustering_metrics[metric_name]["train"].append(
                    clustering_perf[metric_name]["train"]
                )
                clustering_metrics[metric_name]["test"].append(
                    clustering_perf[metric_name]["test"]
                )

            # Classification metrics
            class_perf = results[config]["classification_performance"]
            for metric_name in classification_metrics.keys():
                classification_metrics[metric_name].append(class_perf[metric_name])
        else:
            # Fill with None if config not found
            for metric_name in clustering_metrics.keys():
                clustering_metrics[metric_name]["train"].append(None)
                clustering_metrics[metric_name]["test"].append(None)
            for metric_name in classification_metrics.keys():
                classification_metrics[metric_name].append(None)

    return clustering_metrics, classification_metrics


def create_comparison_metrics_charts(
    base_dir="models/kmeans", output_base_dir="src/kmeans/comparison_images"
):
    """
    Generate combined metrics charts across k for a single config.

    Args:
        base_dir (str): Models base dir.
        output_base_dir (str): Output base dir for images.

    Returns:
        None

    Outputs:
        - comparison_metrics/clustering_metrics/combined_clustering_metrics.png
        - comparison_metrics/classification_metrics/combined_classification_metrics.png
    """
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON_METRICS CHARTS (K VALUES COMPARISON)")
    print("=" * 60)

    # Create output directories
    comparison_metrics_dir = Path(output_base_dir) / "comparison_metrics"
    clustering_output_dir = comparison_metrics_dir / "clustering_metrics"
    classification_output_dir = comparison_metrics_dir / "classification_metrics"

    clustering_output_dir.mkdir(parents=True, exist_ok=True)
    classification_output_dir.mkdir(parents=True, exist_ok=True)

    # Load results for current config across all k values
    results = load_results_for_all_k(base_dir, config_name="current")

    if not results:
        print("[ERROR] No results found for comparison_metrics")
        return

    # Extract metrics
    clustering_metrics, classification_metrics, k_values = extract_metrics_across_k(
        results
    )

    # Plot combined clustering metrics
    print("\nGenerating combined clustering metrics chart...")
    plot_combined_clustering_metrics(
        k_values,
        "Number of Clusters (k)",
        clustering_metrics,
        clustering_output_dir / "combined_clustering_metrics.png",
        title_prefix="",
    )

    # Plot combined classification metrics
    print("Generating combined classification metrics chart...")
    plot_combined_classification_metrics(
        k_values,
        "Number of Clusters (k)",
        classification_metrics,
        classification_output_dir / "combined_classification_metrics.png",
        title_prefix="",
    )

    print(f"\n[COMPLETE] comparison_metrics charts saved to {comparison_metrics_dir}")


def create_comparison_params_charts(
    base_dir="models/kmeans", output_base_dir="src/kmeans/comparison_images"
):
    """
    Generate combined metrics charts across configurations for each k.

    Args:
        base_dir (str): Models base dir.
        output_base_dir (str): Output base dir for images.

    Returns:
        None

    Outputs:
        - comparison_params/k{}/clustering_metrics/combined_clustering_metrics.png
        - comparison_params/k{}/classification_metrics/combined_classification_metrics.png
    """
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON_PARAMS CHARTS (PARAMETER COMPARISON)")
    print("=" * 60)

    comparison_params_dir = Path(output_base_dir) / "comparison_params"

    config_names = [
        "current",
        "tfidf_50",
        "tfidf_100",
        "tfidf_1000",
        "source_0.1",
        "source_0.3",
        "source_0.5",
    ]
    k_values = range(2, 9)

    for k in k_values:
        print(f"\n[K={k}] Processing...")

        # Create directories for this k
        k_dir = comparison_params_dir / f"k{k}"
        clustering_output_dir = k_dir / "clustering_metrics"
        classification_output_dir = k_dir / "classification_metrics"

        clustering_output_dir.mkdir(parents=True, exist_ok=True)
        classification_output_dir.mkdir(parents=True, exist_ok=True)

        # Load results for all configs at this k
        results = load_results_for_all_configs(
            base_dir, k_value=k, config_names=config_names
        )

        if not results:
            print(f"[WARNING] No results found for k={k}")
            continue

        # Extract metrics
        clustering_metrics, classification_metrics = extract_metrics_across_configs(
            results, config_names
        )

        # Plot combined clustering metrics
        plot_combined_clustering_metrics(
            config_names,
            "Configuration",
            clustering_metrics,
            clustering_output_dir / "combined_clustering_metrics.png",
            title_prefix=f"K={k}: ",
        )

        # Plot combined classification metrics
        plot_combined_classification_metrics(
            config_names,
            "Configuration",
            classification_metrics,
            classification_output_dir / "combined_classification_metrics.png",
            title_prefix=f"K={k}: ",
        )

    print(f"\n[COMPLETE] comparison_params charts saved to {comparison_params_dir}")


def main():
    """Run both comparison chart generation processes."""
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING: COMPREHENSIVE COMPARISON ANALYSIS")
    print("=" * 60)

    base_dir = "models/kmeans"
    output_base_dir = "src/kmeans/comparison_images"

    # Generate comparison_metrics (k values comparison for 'current')
    create_comparison_metrics_charts(base_dir, output_base_dir)

    # Generate comparison_params (parameter comparison at each k)
    create_comparison_params_charts(base_dir, output_base_dir)


if __name__ == "__main__":
    main()
