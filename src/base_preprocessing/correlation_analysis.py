"""
Engineered Feature Correlation Analysis (non-TF-IDF)

Computes correlations between interpretable engineered features and spam labels,
saves figures and reports for quick inspection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")


def analyze_spam_feature_correlations():
    """
    Run correlation analysis for engineered (non-TF-IDF) features.

    Returns:
        dict: Results per dataset key with correlations and summary stats.

    Outputs:
        - correlation_results/correlations_sorted.csv
        - correlation_results/engineered_features_analysis.png
        - correlation_results/top_engineered_features.png
        - correlation_results/engineered_features_correlation_report.txt
        - correlation_results/engineered_features_summary.json
    """

    print("SPAM DETECTION FEATURE CORRELATION ANALYSIS")
    print("Focus: Engineered Features (Excluding TF-IDF)")
    print("=" * 60)

    # Create output directory
    output_dir = Path("correlation_results")
    output_dir.mkdir(exist_ok=True)

    results = {}

    # Analyze only KMeans engineered features
    feature_sets = {
        "Engineered_Features": "preprocessed/kmeans/tfidf_1000",
    }

    for set_name, data_path in feature_sets.items():
        print(f"\nAnalyzing {set_name}...")

        try:
            # Load features and labels
            features, labels, feature_names = load_feature_data(data_path)

            if features is None:
                print(f"Could not load {set_name}")
                continue

            # Filter out TF-IDF features
            filtered_features, filtered_names = filter_non_tfidf_features(
                features, feature_names
            )

            if filtered_features.shape[1] == 0:
                print(f"No non-TF-IDF features found in {set_name}")
                continue

            print(f"   Original features: {features.shape[1]}")
            print(f"   Non-TF-IDF features: {filtered_features.shape[1]}")

            # Compute correlations with spam labels
            correlations = compute_spam_correlations(
                filtered_features, labels, filtered_names
            )

            # Sort by absolute correlation (highest to lowest)
            correlations = correlations.sort_values("abs_correlation", ascending=False)

            # Analyze correlation patterns
            analysis = analyze_correlation_patterns(correlations)

            # Create visualizations
            create_correlation_plots(correlations, output_dir)

            # Save results (sorted by absolute correlation)
            correlations.to_csv(output_dir / "correlations_sorted.csv", index=False)

            results[set_name] = {
                "correlations": correlations,
                "analysis": analysis,
                "n_features": len(filtered_names),
                "n_samples": filtered_features.shape[0],
            }

            print(f"Analysis complete for {set_name}")

        except Exception as e:
            print(f"Error analyzing {set_name}: {e}")
            results[set_name] = {"error": str(e)}

    # Generate summary report
    generate_summary_report(results, output_dir)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    return results


def load_feature_data(data_path):
    """
    Load features/labels and names from preprocessing outputs.

    Args:
        data_path (str|Path): Directory with feature arrays and metadata.

    Returns:
        tuple[np.ndarray|None, np.ndarray|None, list[str]|None]: (X, y, names) or (None,...).
    """
    data_dir = Path(data_path)

    try:
        # Try loading K-means style (dense arrays)
        if (data_dir / "X_train.npy").exists():
            X = np.load(data_dir / "X_train.npy")
            y = np.load(data_dir / "y_train.npy")

            # Load metadata for feature names
            with open(data_dir / "feature_metadata.json", "r") as f:
                metadata = json.load(f)
            feature_names = metadata.get(
                "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
            )

            return X, y, feature_names

        # Try loading XGBoost style (sparse arrays)
        elif (data_dir / "X_train.npz").exists():
            from scipy import sparse
            import joblib

            X_sparse = sparse.load_npz(data_dir / "X_train.npz")

            # Try both .npy and .pkl formats for labels
            if (data_dir / "y_train.npy").exists():
                y = np.load(data_dir / "y_train.npy")
            elif (data_dir / "y_train.pkl").exists():
                y = joblib.load(data_dir / "y_train.pkl")
            else:
                print(f"No y_train file found in {data_dir}")
                return None, None, None

            # Convert to dense (sample if too large)
            if X_sparse.shape[1] > 5000:
                print(
                    f"   Large feature space ({X_sparse.shape[1]} features). Sampling 5000 for analysis."
                )
                sample_indices = np.random.choice(
                    X_sparse.shape[1], 5000, replace=False
                )
                X = X_sparse[:, sample_indices].toarray()

                # Load metadata and sample feature names
                with open(data_dir / "feature_metadata.json", "r") as f:
                    metadata = json.load(f)
                all_feature_names = metadata.get(
                    "feature_names", [f"feature_{i}" for i in range(X_sparse.shape[1])]
                )
                feature_names = [all_feature_names[i] for i in sample_indices]
            else:
                X = X_sparse.toarray()
                with open(data_dir / "feature_metadata.json", "r") as f:
                    metadata = json.load(f)
                feature_names = metadata.get(
                    "feature_names", [f"feature_{i}" for i in range(X.shape[1])]
                )

            return X, y, feature_names

        else:
            print(f"No feature files found in {data_dir}")
            return None, None, None

    except Exception as e:
        print(f"Error loading data from {data_dir}: {e}")
        return None, None, None


def filter_non_tfidf_features(X, feature_names):
    """
    Keep only engineered features (exclude names starting with 'tfidf_').

    Args:
        X (np.ndarray): Feature matrix.
        feature_names (list[str]): All feature names.

    Returns:
        tuple[np.ndarray, list[str]]: Filtered X and names.
    """

    # Identify non-TF-IDF features
    non_tfidf_indices = []
    non_tfidf_names = []

    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        # Exclude TF-IDF features
        if not name_lower.startswith("tfidf_"):
            non_tfidf_indices.append(i)
            non_tfidf_names.append(name)

    if len(non_tfidf_indices) == 0:
        return np.array([]).reshape(X.shape[0], 0), []

    # Filter features
    filtered_X = X[:, non_tfidf_indices]

    print(
        f"   Filtered out {len(feature_names) - len(non_tfidf_indices)} TF-IDF features"
    )
    print(f"   Keeping {len(non_tfidf_indices)} engineered features")

    return filtered_X, non_tfidf_names


def compute_spam_correlations(X, y, feature_names):
    """
    Compute Pearson correlations between engineered features and labels.

    Args:
        X (np.ndarray): Engineered feature matrix.
        y (np.ndarray): Binary labels.
        feature_names (list[str]): Names for the columns in X.

    Returns:
        pd.DataFrame: Columns ['feature_name','correlation','abs_correlation','p_value','feature_type'].
    """
    print(f"   Computing correlations for {X.shape[1]} non-TF-IDF features...")

    correlations = []

    for i, feature_name in enumerate(feature_names):
        feature_values = X[:, i]

        # Skip constant features
        if np.std(feature_values) == 0:
            correlations.append(
                {
                    "feature_name": feature_name,
                    "correlation": 0.0,
                    "abs_correlation": 0.0,
                    "p_value": 1.0,
                    "feature_type": classify_feature_type(feature_name),
                }
            )
            continue

        # Compute Pearson correlation
        try:
            from scipy.stats import pearsonr

            corr, p_val = pearsonr(feature_values, y)

            correlations.append(
                {
                    "feature_name": feature_name,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "p_value": p_val,
                    "feature_type": classify_feature_type(feature_name),
                }
            )
        except:
            correlations.append(
                {
                    "feature_name": feature_name,
                    "correlation": 0.0,
                    "abs_correlation": 0.0,
                    "p_value": 1.0,
                    "feature_type": classify_feature_type(feature_name),
                }
            )

    # Convert to DataFrame
    corr_df = pd.DataFrame(correlations)

    return corr_df


def classify_feature_type(feature_name):
    """
    Heuristic categorization of engineered feature names.

    Args:
        feature_name (str): Feature name.

    Returns:
        str: Category label.
    """
    feature_name = feature_name.lower()

    if "source" in feature_name:
        return "Source"
    elif any(
        word in feature_name
        for word in ["count", "length", "ratio", "char", "word", "sentence"]
    ):
        return "Text_Stats"
    elif any(
        word in feature_name
        for word in ["url", "email", "phone", "money", "excited", "number"]
    ):
        return "Spam_Tokens"
    elif any(
        word in feature_name
        for word in ["caps", "upper", "exclamation", "question", "punctuation"]
    ):
        return "Formatting"
    elif any(
        word in feature_name
        for word in [
            "time_pressure",
            "scarcity",
            "superlatives",
            "call_to_action",
            "imperative",
        ]
    ):
        return "Urgency_Language"
    elif any(
        word in feature_name
        for word in ["positive_emotions", "fear_words", "excitement", "emotional"]
    ):
        return "Emotional_Indicators"
    elif any(
        word in feature_name
        for word in ["free", "offer", "promotional", "money", "price"]
    ):
        return "Commercial_Indicators"
    elif any(word in feature_name for word in ["grammar", "complexity", "readability"]):
        return "Linguistic_Quality"
    else:
        return "Other"


def analyze_correlation_patterns(corr_df):
    """
    Summarize correlation strengths, significance, and top predictors.

    Args:
        corr_df (pd.DataFrame): Sorted by abs_correlation desc.

    Returns:
        dict: Summary metrics and top predictors.
    """

    # Overall statistics
    total_features = len(corr_df)
    significant_features = len(corr_df[corr_df["p_value"] < 0.05])
    strong_predictors = len(corr_df[corr_df["abs_correlation"] > 0.1])

    # Top predictors (already sorted by abs_correlation)
    top_overall = corr_df.head(10)
    top_positive = corr_df[corr_df["correlation"] > 0].head(10)
    top_negative = corr_df[corr_df["correlation"] < 0].head(10)

    # Feature type analysis
    type_analysis = (
        corr_df.groupby("feature_type")
        .agg(
            {
                "abs_correlation": ["count", "mean", "max"],
                "p_value": lambda x: (x < 0.05).sum(),
            }
        )
        .round(4)
    )

    analysis = {
        "total_features": total_features,
        "significant_features": significant_features,
        "strong_predictors": strong_predictors,
        "significance_rate": (
            significant_features / total_features if total_features > 0 else 0
        ),
        "strong_predictor_rate": (
            strong_predictors / total_features if total_features > 0 else 0
        ),
        "top_overall_predictors": top_overall[
            ["feature_name", "correlation", "abs_correlation", "p_value"]
        ].to_dict("records"),
        "top_positive_predictors": top_positive[
            ["feature_name", "correlation", "p_value"]
        ].to_dict("records"),
        "top_negative_predictors": top_negative[
            ["feature_name", "correlation", "p_value"]
        ].to_dict("records"),
        "feature_type_stats": type_analysis.to_dict() if len(type_analysis) > 0 else {},
    }

    # Print summary
    print(f"   {significant_features}/{total_features} features significant (p<0.05)")
    print(f"   {strong_predictors} strong predictors (|r|>0.1)")

    if len(top_overall) > 0:
        best_predictor = top_overall.iloc[0]
        print(
            f"   Best predictor: {best_predictor['feature_name']} (|r|={best_predictor['abs_correlation']:.3f})"
        )

    return analysis


def create_correlation_plots(corr_df, output_dir):
    """
    Generate overview plots for engineered features and significance.

    Args:
        corr_df (pd.DataFrame): Correlation table.
        set_name (str): Dataset label.
        output_dir (Path|str): Output directory.

    Returns:
        None
    """

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Engineered Feature Correlations (No TF-IDF)",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Top 15 strongest correlations by absolute value
    top_features = corr_df.head(15)

    # Color code by correlation direction
    colors = ["red" if x > 0 else "blue" for x in top_features["correlation"]]

    axes[0, 0].barh(
        range(len(top_features)),
        top_features["abs_correlation"],
        color=colors,
        alpha=0.7,
    )
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(
        [
            name[:25] + "..." if len(name) > 25 else name
            for name in top_features["feature_name"]
        ],
        fontsize=9,
    )
    axes[0, 0].set_xlabel("Absolute Correlation with Spam Label")
    axes[0, 0].set_title(
        "Top 15 Predictive Features\n(Red=Spam Indicators, Blue=Ham Indicators)"
    )
    axes[0, 0].grid(axis="x", alpha=0.3)

    # Add correlation values as text
    for i, (corr, abs_corr) in enumerate(
        zip(top_features["correlation"], top_features["abs_correlation"])
    ):
        axes[0, 0].text(
            abs_corr + 0.005, i, f"{corr:.3f}", verticalalignment="center", fontsize=8
        )

    # 2. Correlation distribution
    axes[0, 1].hist(
        corr_df["correlation"], bins=30, alpha=0.7, edgecolor="black", color="skyblue"
    )
    axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel("Correlation with Spam Label")
    axes[0, 1].set_ylabel("Number of Features")
    axes[0, 1].set_title("Distribution of Feature Correlations")
    axes[0, 1].grid(alpha=0.3)

    # Add statistics text
    mean_corr = corr_df["correlation"].mean()
    std_corr = corr_df["correlation"].std()
    axes[0, 1].text(
        0.05,
        0.95,
        f"Mean: {mean_corr:.3f}\nStd: {std_corr:.3f}",
        transform=axes[0, 1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 3. Feature type analysis
    if len(corr_df) > 0:
        type_stats = (
            corr_df.groupby("feature_type")["abs_correlation"]
            .agg(["count", "mean"])
            .reset_index()
        )
        type_stats = type_stats.sort_values("mean", ascending=True)

        axes[1, 0].barh(
            range(len(type_stats)), type_stats["mean"], color="green", alpha=0.7
        )
        axes[1, 0].set_yticks(range(len(type_stats)))
        axes[1, 0].set_yticklabels(type_stats["feature_type"])
        axes[1, 0].set_xlabel("Average Absolute Correlation")
        axes[1, 0].set_title("Feature Type Effectiveness")
        axes[1, 0].grid(axis="x", alpha=0.3)

        # Add count annotations
        for i, (count, mean_val) in enumerate(
            zip(type_stats["count"], type_stats["mean"])
        ):
            axes[1, 0].text(
                mean_val + 0.001,
                i,
                f"(n={count})",
                verticalalignment="center",
                fontsize=8,
            )

    # 4. Significance analysis
    significance_data = corr_df.copy()
    significance_data["significance"] = significance_data["p_value"].apply(
        lambda x: (
            "Highly Significant (p<0.01)"
            if x < 0.01
            else "Significant (p<0.05)" if x < 0.05 else "Not Significant (p≥0.05)"
        )
    )

    sig_counts = significance_data["significance"].value_counts()
    colors_sig = ["darkgreen", "orange", "lightcoral"]

    wedges, texts, autotexts = axes[1, 1].pie(
        sig_counts.values,
        labels=sig_counts.index,
        autopct="%1.1f%%",
        colors=colors_sig[: len(sig_counts)],
        startangle=90,
    )
    axes[1, 1].set_title("Statistical Significance of Features")

    plt.tight_layout()
    plt.savefig(
        output_dir / "engineered_features_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create detailed top features plot
    create_detailed_top_features_plot(corr_df, output_dir)


def create_detailed_top_features_plot(corr_df, output_dir):
    """
    Plot the top engineered features by correlation magnitude.

    Args:
        corr_df (pd.DataFrame): Correlation table.
        output_dir (Path|str): Output directory.

    Returns:
        None
    """

    # Get top 20 features by absolute correlation
    top_20 = corr_df.head(20)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle(
        "Top 20 Most Predictive Engineered Features",
        fontsize=16,
        fontweight="bold",
    )

    # Create horizontal bar plot
    colors = ["red" if x > 0 else "blue" for x in top_20["correlation"]]
    bars = ax.barh(range(len(top_20)), top_20["correlation"], color=colors, alpha=0.7)

    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"{name}" for name in top_20["feature_name"]], fontsize=10)
    ax.set_xlabel("Correlation with Spam Label")
    ax.set_title(
        "Red = Spam Indicators (Positive Correlation)\nBlue = Ham Indicators (Negative Correlation)"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.5)

    # Add correlation values and significance markers
    for i, (corr, p_val, abs_corr) in enumerate(
        zip(top_20["correlation"], top_20["p_value"], top_20["abs_correlation"])
    ):
        # Position text based on correlation direction
        x_pos = corr + (0.01 if corr > 0 else -0.01)
        ha = "left" if corr > 0 else "right"

        # Add significance marker
        sig_marker = (
            "***"
            if p_val < 0.001
            else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        )

        ax.text(
            x_pos,
            i,
            f"{corr:.3f}{sig_marker}",
            verticalalignment="center",
            fontsize=9,
            ha=ha,
            fontweight="bold",
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="Spam Indicators"),
        Patch(facecolor="blue", alpha=0.7, label="Ham Indicators"),
        Patch(facecolor="white", label="*** p<0.001, ** p<0.01, * p<0.05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(
        output_dir / "top_engineered_features.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def generate_summary_report(results, output_dir):
    """
    Write text and JSON summaries of engineered feature findings.

    Args:
        results (dict): Per-dataset analysis outputs.
        output_dir (Path|str): Output directory.

    Returns:
        None
    """

    # Create summary statistics
    summary = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "analysis_focus": "Engineered Features Only (TF-IDF Excluded)",
        "datasets_analyzed": len([k for k, v in results.items() if "error" not in v]),
        "summary_by_dataset": {},
    }

    # Text report lines
    report_lines = [
        "SPAM DETECTION: ENGINEERED FEATURE CORRELATION ANALYSIS",
        "=" * 60,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Focus: Hand-crafted features only (TF-IDF features excluded)",
        "",
        "EXECUTIVE SUMMARY:",
        "This analysis examines correlations between engineered features and spam labels",
        "to identify the most interpretable and predictive features for spam detection.",
        "Results are sorted by absolute correlation strength (highest to lowest).",
        "",
    ]

    for dataset_name, data in results.items():
        if "error" not in data:
            analysis = data["analysis"]

            # Add to summary
            summary["summary_by_dataset"][dataset_name] = {
                "total_features": analysis["total_features"],
                "significant_features": analysis["significant_features"],
                "strong_predictors": analysis["strong_predictors"],
                "top_predictor": (
                    analysis["top_overall_predictors"][0]
                    if analysis["top_overall_predictors"]
                    else None
                ),
            }

            # Add to text report
            report_lines.extend(
                [
                    f"{dataset_name.upper()} ANALYSIS:",
                    f"  Engineered Features: {analysis['total_features']}",
                    f"  Statistically Significant: {analysis['significant_features']} ({analysis['significance_rate']:.1%})",
                    f"  Strong Predictors (|r|>0.1): {analysis['strong_predictors']} ({analysis['strong_predictor_rate']:.1%})",
                    "",
                ]
            )

            if analysis["top_overall_predictors"]:
                report_lines.append(
                    "  TOP 5 PREDICTIVE FEATURES (by absolute correlation):"
                )
                for i, predictor in enumerate(
                    analysis["top_overall_predictors"][:5], 1
                ):
                    direction = "SPAM" if predictor["correlation"] > 0 else "HAM"
                    report_lines.append(
                        f"    {i}. {predictor['feature_name']} "
                        f"(r={predictor['correlation']:.3f}, |r|={predictor['abs_correlation']:.3f}, "
                        f"p={predictor['p_value']:.4f}) [{direction} indicator]"
                    )
                report_lines.append("")

            report_lines.append("")

        else:
            report_lines.append(f"{dataset_name}: ERROR - {data['error']}")
            report_lines.append("")

    report_lines.extend(
        [
            "KEY INSIGHTS:",
            "- Results sorted by absolute correlation (strongest predictors first)",
            "- Features with |correlation| > 0.1 are strong predictors",
            "- P-values < 0.05 indicate statistical significance",
            "- Positive correlations = spam indicators, negative = ham indicators",
            "- Focus on interpretable features for model development",
            "",
            "RECOMMENDATIONS FOR ML MODELS:",
            "1. Use top 20-50 features for XGBoost feature selection",
            "2. Include emotional and urgency indicators in LSTM preprocessing",
            "3. Consider feature engineering patterns for new datasets",
            "4. Validate feature importance in final model training",
            "",
        ]
    )

    # Save reports
    with open(output_dir / "engineered_features_correlation_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    with open(output_dir / "engineered_features_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print to console
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    print("Starting Engineered Feature Correlation Analysis...")

    # Run the analysis
    results = analyze_spam_feature_correlations()

    print("\nAnalysis completed!")
    print("\nFiles generated:")
    print("correlation_results/")
    print("  ├── correlations_sorted.csv (sorted by absolute correlation)")
    print("  ├── engineered_features_analysis.png")
    print("  ├── top_engineered_features.png")
    print("  ├── engineered_features_correlation_report.txt")
    print("  └── engineered_features_summary.json")
    print(
        "\nCheck the correlation report for key insights about your spam detection features!"
    )
