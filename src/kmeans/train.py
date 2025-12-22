"""
K-Means training and evaluation for spam detection.

Provides utilities to load engineered features, train K-Means, evaluate with
clustering and classification-style metrics, create visualizations, and save
artifacts.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from collections import Counter

warnings.filterwarnings("ignore")


def load_engineered_features(features_dir="preprocessed/kmeans"):
    """
    Load engineered features and metadata for K-Means.

    Args:
        features_dir (str): Directory containing engineered features.

    Returns:
        tuple: (X_train, X_test, X_train_2d, X_test_2d, y_train, y_test,
                train_sources, test_sources, metadata)

    Expected files:
        X_train.npy, X_test.npy, X_train_2d.npy, X_test_2d.npy,
        y_train.npy, y_test.npy, train_sources.npy, test_sources.npy,
        feature_metadata.json
    """
    features_path = Path(features_dir)

    print(f"Loading features from: {features_path}")

    # Load standard features
    X_train = np.load(features_path / "X_train.npy")
    X_test = np.load(features_path / "X_test.npy")
    X_train_2d = np.load(features_path / "X_train_2d.npy")
    X_test_2d = np.load(features_path / "X_test_2d.npy")
    y_train = np.load(features_path / "y_train.npy")
    y_test = np.load(features_path / "y_test.npy")
    train_sources = np.load(features_path / "train_sources.npy", allow_pickle=True)
    test_sources = np.load(features_path / "test_sources.npy", allow_pickle=True)

    # Load metadata
    with open(features_path / "feature_metadata.json", "r") as f:
        metadata = json.load(f)

    print(f"Training features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")

    return (
        X_train,
        X_test,
        X_train_2d,
        X_test_2d,
        y_train,
        y_test,
        train_sources,
        test_sources,
        metadata,
    )


def train_kmeans_model(X_train, k_clusters):
    """
    Train a K-Means model.

    Args:
        X_train (np.ndarray): Training feature matrix.
        k_clusters (int): Number of clusters.

    Returns:
        sklearn.cluster.KMeans: Trained model.

    Notes:
        - k-means++, n_init=10, max_iter=300, random_state=42.
    """
    print(f"Training K-Means with k={k_clusters}...")

    kmeans = KMeans(
        n_clusters=k_clusters,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42,
    )

    kmeans.fit(X_train)
    print(f"Model trained. Inertia: {kmeans.inertia_:.0f}")

    return kmeans


def evaluate_clustering_performance(model, X_train, X_test, y_train, y_test):
    """
    Compute standard clustering metrics and cluster assignments.

    Args:
        model (KMeans): Trained model.
        X_train (np.ndarray): Train features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Train labels.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple: (clustering_metrics, train_clusters, test_clusters)
            clustering_metrics = {
                'silhouette_score': {'train', 'test'},
                'adjusted_rand_index': {'train', 'test'},
                'normalized_mutual_info': {'train', 'test'}
            }
    """
    print("Evaluating clustering performance...")

    # Get cluster predictions
    train_clusters = model.predict(X_train)
    test_clusters = model.predict(X_test)

    # Clustering metrics
    train_silhouette = silhouette_score(X_train, train_clusters)
    test_silhouette = silhouette_score(X_test, test_clusters)

    train_ari = adjusted_rand_score(y_train, train_clusters)
    test_ari = adjusted_rand_score(y_test, test_clusters)

    train_nmi = normalized_mutual_info_score(y_train, train_clusters)
    test_nmi = normalized_mutual_info_score(y_test, test_clusters)

    clustering_metrics = {
        "silhouette_score": {"train": train_silhouette, "test": test_silhouette},
        "adjusted_rand_index": {"train": train_ari, "test": test_ari},
        "normalized_mutual_info": {"train": train_nmi, "test": test_nmi},
    }

    print("Clustering Metrics:")
    print(
        f"  Silhouette Score - Train: {train_silhouette:.3f}, Test: {test_silhouette:.3f}"
    )
    print(f"  Adjusted Rand Index - Train: {train_ari:.3f}, Test: {test_ari:.3f}")
    print(f"  Normalized Mutual Info - Train: {train_nmi:.3f}, Test: {test_nmi:.3f}")

    return clustering_metrics, train_clusters, test_clusters


def evaluate_classification_performance(test_clusters, y_test):
    """
    Evaluate clusters as a classifier via majority mapping.

    Args:
        test_clusters (np.ndarray): Cluster IDs for test samples.
        y_test (np.ndarray): True labels.

    Returns:
        tuple: (classification_metrics, cluster_to_class, y_pred)
            classification_metrics includes accuracy, precision, recall,
            f1_score, roc_auc, average_precision, confusion_matrix.
    """
    print("Evaluating classification performance...")

    # Map clusters to spam/ham based on majority class in each cluster
    cluster_to_class = {}
    for cluster_id in np.unique(test_clusters):
        cluster_mask = test_clusters == cluster_id
        cluster_labels = y_test[cluster_mask]
        if len(cluster_labels) > 0:
            # Assign cluster to majority class
            cluster_to_class[cluster_id] = 1 if np.mean(cluster_labels) > 0.5 else 0

    # Convert cluster predictions to binary predictions
    y_pred = np.array([cluster_to_class.get(cluster, 0) for cluster in test_clusters])
    y_pred_proba = np.array(
        [
            (
                np.mean(y_test[test_clusters == cluster])
                if np.sum(test_clusters == cluster) > 0
                else 0.5
            )
            for cluster in test_clusters
        ]
    )

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
    except:
        roc_auc = 0.5
        avg_precision = np.mean(y_test)

    cm = confusion_matrix(y_test, y_pred)

    classification_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": cm.tolist(),
    }

    print("Classification Metrics:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")

    return classification_metrics, cluster_to_class, y_pred


def analyze_cluster_composition(clusters, labels, sources, dataset_name="Test"):
    """
    Summarize clusters by size, spam rate, and source distribution.

    Args:
        clusters (np.ndarray): Cluster IDs.
        labels (np.ndarray): Binary labels.
        sources (np.ndarray): Source names.
        dataset_name (str): Tag for printouts.

    Returns:
        list[dict]: One entry per cluster with size, spam_rate, dominant_source, distribution.
    """
    print(f"Analyzing cluster composition ({dataset_name})...")

    cluster_info = []
    for cluster_id in range(len(np.unique(clusters))):
        cluster_mask = clusters == cluster_id
        cluster_labels = labels[cluster_mask]
        cluster_sources = sources[cluster_mask]

        if len(cluster_labels) > 0:
            spam_rate = np.mean(cluster_labels)
            size = len(cluster_labels)
            source_dist = Counter(cluster_sources)
            dominant_source = source_dist.most_common(1)[0][0]

            cluster_info.append(
                {
                    "cluster_id": cluster_id,
                    "size": size,
                    "spam_rate": spam_rate,
                    "dominant_source": dominant_source,
                    "source_distribution": dict(source_dist),
                }
            )

            print(
                f"  Cluster {cluster_id}: size={size:4d}, spam_rate={spam_rate:.1%}, "
                f"dominant={dominant_source}"
            )

    return cluster_info


def create_visualizations(
    model,
    X_test_2d,
    test_clusters,
    y_test,
    test_sources,
    metadata,
    output_dir,
    y_pred=None,
):
    """
    Create cluster visualizations and optional confusion matrix.

    Args:
        model (KMeans): Trained model.
        X_test_2d (np.ndarray): 2D test features.
        test_clusters (np.ndarray): Cluster IDs (test).
        y_test (np.ndarray): Test labels.
        test_sources (np.ndarray): Test sources.
        metadata (dict): Feature metadata.
        output_dir (str|Path): Output directory.
        y_pred (np.ndarray|None): Optional predicted labels for confusion matrix.

    Returns:
        None

    Outputs:
        - clustering_analysis.png in output_dir
    """
    print("Creating enhanced visualizations with confusion matrix...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Color schemes
    spam_colors = {0: "blue", 1: "red"}
    source_markers = {"sms": "o", "email": "s", "youtube": "^", "review": "D"}

    # Create figure with 3x2 grid for 6 subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])  # Confusion Matrix
    ax6 = fig.add_subplot(gs[2, 1])  # Source distribution

    # 1. Cluster visualization colored by spam/ham (Top Left)
    for spam_label in [0, 1]:
        mask = y_test == spam_label
        color = spam_colors[spam_label]
        label = "Ham" if spam_label == 0 else "Spam"
        ax1.scatter(
            X_test_2d[mask, 0],
            X_test_2d[mask, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=50,
        )

    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.5,
        -0.15,
        "Data Distribution by Spam/Ham Labels",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    # 2. Cluster visualization colored by source (Top Right)
    unique_sources = np.unique(test_sources)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sources)))

    for i, source in enumerate(unique_sources):
        mask = test_sources == source
        marker = source_markers.get(source, "o")
        ax2.scatter(
            X_test_2d[mask, 0],
            X_test_2d[mask, 1],
            c=[colors[i]],
            label=source.title(),
            marker=marker,
            alpha=0.6,
            s=50,
        )

    ax2.set_xlabel("First Principal Component")
    ax2.set_ylabel("Second Principal Component")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.5,
        -0.15,
        "Data Distribution by Source",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    # 3. Cluster assignments (Middle Left)
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, model.n_clusters))

    for cluster_id in range(model.n_clusters):
        mask = test_clusters == cluster_id
        ax3.scatter(
            X_test_2d[mask, 0],
            X_test_2d[mask, 1],
            c=[cluster_colors[cluster_id]],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=50,
        )

    ax3.set_xlabel("First Principal Component")
    ax3.set_ylabel("Second Principal Component")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.5,
        -0.15,
        f"K-Means Cluster Assignments (k={model.n_clusters})",
        transform=ax3.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    # 4. Spam rate by cluster (Middle Right)
    cluster_spam_rates = []
    cluster_sizes = []

    for cluster_id in range(model.n_clusters):
        cluster_mask = test_clusters == cluster_id
        spam_rate = (
            np.mean(y_test[cluster_mask]) * 100 if np.sum(cluster_mask) > 0 else 0
        )
        size = np.sum(cluster_mask)
        cluster_spam_rates.append(spam_rate)
        cluster_sizes.append(size)

    bars = ax4.bar(
        range(model.n_clusters),
        cluster_spam_rates,
        color=[
            "lightcoral" if rate > 50 else "lightblue" for rate in cluster_spam_rates
        ],
        alpha=0.8,
        edgecolor="black",
    )

    ax4.set_xlabel("Cluster ID")
    ax4.set_ylabel("Spam Rate (%)")
    ax4.set_xticks(range(model.n_clusters))
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.text(
        0.5,
        -0.15,
        "Spam Rate by Cluster",
        transform=ax4.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    # Add labels on bars
    for i, (bar, rate, size) in enumerate(zip(bars, cluster_spam_rates, cluster_sizes)):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%\n(n={size})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 5. NEW: Confusion Matrix (Bottom Left)
    if y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)

        # Create confusion matrix heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax5,
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"],
            cbar_kws={"label": "Count"},
        )

        ax5.set_xlabel("Predicted Label")
        ax5.set_ylabel("True Label")
        # Title centered just below the plot
        ax5.text(
            0.5,
            -0.15,
            f"Confusion Matrix (k={model.n_clusters})",
            transform=ax5.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

        # Calculate and add performance metrics as text
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Removed metrics box per request
    else:
        ax5.text(
            0.5,
            0.5,
            "Confusion Matrix\nNot Available\n(y_pred not provided)",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=14,
        )
        ax5.set_xticks([])
        ax5.set_yticks([])

    # 6. Source distribution within spam messages by cluster (Bottom Right)
    all_sources = sorted(unique_sources)
    source_colors_map = {
        "email": "#FF6B6B",
        "sms": "#4ECDC4",
        "youtube": "#95E1D3",
        "review": "#F38181",
    }

    # Prepare data for stacked bar chart
    cluster_ids = range(model.n_clusters)
    source_percentages = {source: [] for source in all_sources}

    for cluster_id in cluster_ids:
        # Get spam messages in this cluster
        spam_mask = (test_clusters == cluster_id) & (y_test == 1)
        spam_sources = test_sources[spam_mask]

        total_spam = len(spam_sources)

        if total_spam > 0:
            # Calculate percentage for each source
            for source in all_sources:
                count = np.sum(spam_sources == source)
                percentage = (count / total_spam) * 100
                source_percentages[source].append(percentage)
        else:
            # No spam in this cluster
            for source in all_sources:
                source_percentages[source].append(0)

    # Create stacked horizontal bar chart
    y_pos = np.arange(len(cluster_ids))
    left = np.zeros(len(cluster_ids))

    for source in all_sources:
        color = source_colors_map.get(source, "#CCCCCC")
        percentages = source_percentages[source]
        ax6.barh(
            y_pos,
            percentages,
            left=left,
            label=source.title(),
            color=color,
            alpha=0.8,
            edgecolor="black",
        )

        # Add percentage labels on bars (only if percentage > 5%)
        for i, (p, l) in enumerate(zip(percentages, left)):
            if p > 5:
                ax6.text(
                    l + p / 2,
                    i,
                    f"{p:.1f}%",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=9,
                )

        left += percentages

    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([f"Cluster {i}" for i in cluster_ids])
    ax6.set_xlabel("Percentage (%)")
    ax6.set_ylabel("Cluster ID")
    ax6.set_xlim(0, 100)
    ax6.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title="Source")
    ax6.grid(True, alpha=0.3, axis="x")

    # Adjust subplot position for legend
    box = ax6.get_position()
    ax6.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    ax6.text(
        0.5,
        -0.15,
        "Source Distribution within Spam Messages by Cluster",
        transform=ax6.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    plt.suptitle(
        f"K-Means Clustering Analysis (k={model.n_clusters})", fontsize=16, y=0.98
    )
    plt.savefig(output_path / "clustering_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"Enhanced visualizations saved to: {output_path / 'clustering_analysis.png'}"
    )


def save_results(
    model,
    clustering_performance,
    classification_performance,
    cluster_info,
    metadata,
    output_dir,
    cluster_to_class,
    features_dir=None,
):
    """
    Persist model and results with metadata.

    Args:
        model (KMeans): Trained model.
        clustering_performance (dict): Clustering metrics.
        classification_performance (dict): Classification-style metrics.
        cluster_info (list[dict]): Cluster composition.
        metadata (dict): Feature metadata.
        output_dir (str|Path): Output directory.
        cluster_to_class (dict): Cluster→class mapping.

    Returns:
        dict: Results summary as written to JSON.

    Files saved:
        - kmeans_model.pkl, cluster_mapping.pkl, clustering_results.json, cluster_summary.csv
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save the model
    joblib.dump(model, output_path / "kmeans_model.pkl")

    # Save cluster-to-class mapping
    joblib.dump(cluster_to_class, output_path / "cluster_mapping.pkl")

    # CONVERT cluster_to_class keys to strings for JSON compatibility
    cluster_to_class_json = {str(k): int(v) for k, v in cluster_to_class.items()}

    # Create comprehensive results summary
    results_summary = {
        "model_summary": {
            "algorithm": "K-Means Clustering",
            "n_clusters": model.n_clusters,
            "total_features": metadata["n_features"],
            "training_samples": metadata["n_train_samples"],
            "test_samples": metadata["n_test_samples"],
        },
        "clustering_performance": clustering_performance,
        "classification_performance": classification_performance,
        "cluster_analysis": {"cluster_composition": cluster_info},
        # Use the JSON-friendly version with string keys
        "cluster_to_class_mapping": cluster_to_class_json,
        "training_info": {
            "timestamp": datetime.now().isoformat(),
            "feature_engineering": metadata.get("enhancements", []),
            "scaling": metadata.get("scaling_method", "StandardScaler"),
            "random_state": 42,
        },
    }

    # Save results
    with open(output_path / "clustering_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Create CSV summary for easy analysis
    cluster_df = pd.DataFrame(cluster_info)
    cluster_df.to_csv(output_path / "cluster_summary.csv", index=False)

    # Also save preprocessing artifacts needed for standalone inference
    try:
        if features_dir is not None:
            features_path = Path(features_dir)
            # Copy TF-IDF vectorizer, scaler, PCA model, and metadata into model folder
            tfidf_path = features_path / "tfidf_vectorizer.pkl"
            scaler_path = features_path / "scaler.pkl"
            pca_path = features_path / "pca_2d.pkl"
            meta_path = features_path / "feature_metadata.json"
            
            if tfidf_path.exists():
                tfidf = joblib.load(tfidf_path)
                joblib.dump(tfidf, output_path / "tfidf_vectorizer.pkl")
            
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                joblib.dump(scaler, output_path / "scaler.pkl")
            
            # Copy PCA model for 2D visualization
            if pca_path.exists():
                pca_2d = joblib.load(pca_path)
                joblib.dump(pca_2d, output_path / "pca_2d.pkl")
            
            if meta_path.exists():
                with open(meta_path, "r") as mf:
                    meta_json = json.load(mf)
                
                # Calculate and add normalization ranges from training data
                # Load 2D training data to compute min/max
                train_2d_path = features_path / "X_train_2d.npy"
                if train_2d_path.exists():
                    X_train_2d = np.load(train_2d_path)
                    meta_json["pca_2d_normalization"] = {
                        "x_min": float(np.min(X_train_2d[:, 0])),
                        "x_max": float(np.max(X_train_2d[:, 0])),
                        "y_min": float(np.min(X_train_2d[:, 1])),
                        "y_max": float(np.max(X_train_2d[:, 1]))
                    }
                    print(f"PCA 2D normalization ranges: x=[{meta_json['pca_2d_normalization']['x_min']:.2f}, {meta_json['pca_2d_normalization']['x_max']:.2f}], y=[{meta_json['pca_2d_normalization']['y_min']:.2f}, {meta_json['pca_2d_normalization']['y_max']:.2f}]")
                
                with open(output_path / "feature_metadata.json", "w") as wf:
                    json.dump(meta_json, wf, indent=2)
    except Exception as e:
        print(f"Warning: failed to copy preprocessing artifacts for inference: {e}")

    print("Results saved:")
    print(f"  - Model: {output_path / 'kmeans_model.pkl'}")
    print(f"  - Cluster mapping: {output_path / 'cluster_mapping.pkl'}")
    print(f"  - Results: {output_path / 'clustering_results.json'}")
    print(f"  - Cluster table: {output_path / 'cluster_summary.csv'}")

    return results_summary


def train_kmeans(features_dir, output_dir, k_clusters):
    """
    Train, evaluate, visualize, and save K-Means for a given k.

    Args:
        features_dir (str): Engineered features directory.
        output_dir (str): Output directory.
        k_clusters (int): Number of clusters.

    Returns:
        tuple: (model, results_summary)

    Steps:
        1) Load features, 2) Train, 3) Clustering metrics, 4) Classification metrics,
        5) Cluster analysis, 6) Visualizations, 7) Save artifacts.
    """
    print(f"Starting K-Means training with k={k_clusters}")
    print("=" * 60)

    # Load features
    (
        X_train,
        X_test,
        X_train_2d,
        X_test_2d,
        y_train,
        y_test,
        train_sources,
        test_sources,
        metadata,
    ) = load_engineered_features(features_dir)

    # Train model
    model = train_kmeans_model(X_train, k_clusters)

    # Evaluate clustering performance
    clustering_performance, train_clusters, test_clusters = (
        evaluate_clustering_performance(model, X_train, X_test, y_train, y_test)
    )

    # Evaluate as classification task - NOW RETURNS MAPPING AND PREDICTIONS
    classification_performance, cluster_to_class, y_pred = (
        evaluate_classification_performance(test_clusters, y_test)
    )

    # Analyze cluster composition
    cluster_info = analyze_cluster_composition(test_clusters, y_test, test_sources)

    # Create enhanced visualizations with confusion matrix
    create_visualizations(
        model,
        X_test_2d,
        test_clusters,
        y_test,
        test_sources,
        metadata,
        output_dir,
        y_pred,
    )

    # Save results
    results_summary = save_results(
        model,
        clustering_performance,
        classification_performance,
        cluster_info,
        metadata,
        output_dir,
        cluster_to_class,
        features_dir,
    )

    print("=" * 60)
    print(f"K-Means training completed (k={k_clusters})")
    print(f"Silhouette score: {clustering_performance['silhouette_score']['test']:.3f}")
    print(f"Classification F1: {classification_performance['f1_score']:.3f}")
    print(f"Results saved to: {output_dir}")

    return model, results_summary


if __name__ == "__main__":
    # Example usage
    model, results = train_kmeans(
        features_dir="preprocessed/kmeans", output_dir="models/kmeans", k_clusters=3
    )
