import pandas as pd
import numpy as np
import json
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    make_scorer,
)
from scipy import sparse
from scipy.stats import randint, uniform, loguniform
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings

from src.xgb.evaluation import (
    evaluate_model,
    analyze_feature_importance,
    create_evaluation_plots,
)

warnings.filterwarnings("ignore")


def find_optimal_threshold(y_true, y_proba, metric="f1"):
    """
    Find probability threshold maximizing a chosen metric.

    Args:
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities.
        metric (str): 'f1' | 'precision' | 'recall'.

    Returns:
        tuple: (optimal_threshold, best_score, thresholds_df)
    """
    print(f"Finding optimal threshold for {metric.upper()} score...")

    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero

    # Create DataFrame for analysis
    thresholds_data = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],  # precision_recall_curve returns n+1 values
            "recall": recall[:-1],
            "f1": f1_scores[:-1],
        }
    )

    if metric == "f1":
        best_idx = np.argmax(thresholds_data["f1"])
        best_score = thresholds_data["f1"].iloc[best_idx]
    elif metric == "precision":
        best_idx = np.argmax(thresholds_data["precision"])
        best_score = thresholds_data["precision"].iloc[best_idx]
    elif metric == "recall":
        best_idx = np.argmax(thresholds_data["recall"])
        best_score = thresholds_data["recall"].iloc[best_idx]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    optimal_threshold = thresholds_data["threshold"].iloc[best_idx]

    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Best {metric.upper()} score: {best_score:.4f}")
    print(f"At optimal threshold:")
    print(f"  Precision: {thresholds_data['precision'].iloc[best_idx]:.4f}")
    print(f"  Recall: {thresholds_data['recall'].iloc[best_idx]:.4f}")
    print(f"  F1: {thresholds_data['f1'].iloc[best_idx]:.4f}")

    # Validate that we found a reasonable threshold
    if optimal_threshold < 0.1 or optimal_threshold > 0.9:
        print(f"WARNING: Optimal threshold ({optimal_threshold:.4f}) seems extreme!")
        print("This might indicate issues with the model or data distribution.")

    return optimal_threshold, best_score, thresholds_data


def create_validation_split(
    X_train, y_train, train_sources, val_size=0.1, random_state=42
):
    """
    Create a validation split stratified by (source,label).

    Args:
        X_train (array-like): Train features.
        y_train (array-like): Train labels.
        train_sources (array-like): Train sources.
        val_size (float): Validation fraction.
        random_state (int): RNG seed.

    Returns:
        tuple: (X_tr, X_val, y_tr, y_val, sources_tr, sources_val)
    """
    print(
        f"Creating {val_size*100:.0f}% validation split stratified by source+label..."
    )

    # Create stratification key
    stratify_labels = np.array(
        [f"{source}|||{label}" for source, label in zip(train_sources, y_train)]
    )

    # Split the data
    indices = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=random_state, stratify=stratify_labels
    )

    # Create splits
    X_train_split = X_train[train_idx]
    X_val_split = X_train[val_idx]
    y_train_split = y_train[train_idx]
    y_val_split = y_train[val_idx]
    train_sources_split = train_sources[train_idx]
    val_sources_split = train_sources[val_idx]

    print(f"Training split: {len(y_train_split):,} samples")
    print(f"Validation split: {len(y_val_split):,} samples")
    print(f"Training spam rate: {np.mean(y_train_split)*100:.1f}%")
    print(f"Validation spam rate: {np.mean(y_val_split)*100:.1f}%")

    # Verify source distribution
    print("Validation source distribution:")
    val_source_dist = pd.Series(val_sources_split).value_counts(normalize=True) * 100
    for source, pct in val_source_dist.items():
        print(f"  {source}: {pct:.1f}%")

    return (
        X_train_split,
        X_val_split,
        y_train_split,
        y_val_split,
        train_sources_split,
        val_sources_split,
    )


def check_gpu_availability():
    """Return True if XGBoost GPU training is available, else False."""
    try:
        # Try to create a simple XGBoost model with GPU
        temp_model = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=0, n_estimators=1)
        # Test with small dummy data
        X_dummy = np.random.random((10, 5))
        y_dummy = np.random.randint(0, 2, 10)
        temp_model.fit(X_dummy, y_dummy)
        print("âœ“ GPU is available and working with XGBoost")
        return True
    except Exception as e:
        print(f"âš  GPU not available or not working: {e}")
        print("  Falling back to CPU training")
        return False


def load_engineered_features(features_dir="datasets/features"):
    """
    Load engineered features and metadata for training.

    Args:
        features_dir (str): Directory with feature files.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_sources, test_sources, metadata)

    Expected files:
        X_train.npz, X_test.npz, y_train.pkl, y_test.pkl,
        train_sources.pkl, test_sources.pkl, feature_metadata.json
    """
    features_path = Path(features_dir)

    print("Loading engineered features...")

    # Load features
    X_train = sparse.load_npz(features_path / "X_train.npz")
    X_test = sparse.load_npz(features_path / "X_test.npz")
    y_train = joblib.load(features_path / "y_train.pkl")
    y_test = joblib.load(features_path / "y_test.pkl")
    train_sources = joblib.load(features_path / "train_sources.pkl")
    test_sources = joblib.load(features_path / "test_sources.pkl")

    # Load metadata
    with open(features_path / "feature_metadata.json", "r") as f:
        metadata = json.load(f)

    print(f"Training features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")
    print(f"Total features: {metadata['n_features']:,}")
    print(f"Training samples: {len(y_train):,}")
    print(f"Test samples: {len(y_test):,}")

    return X_train, X_test, y_train, y_test, train_sources, test_sources, metadata


def setup_stratified_kfold_cv(y_train, train_sources, n_splits=5, random_state=42):
    """
    Build StratifiedKFold using (source,label) keys; return CV object and keys.

    Args:
        y_train (np.ndarray): Labels.
        train_sources (array-like): Sources.
        n_splits (int): Number of folds.
        random_state (int): RNG seed.

    Returns:
        tuple: (skf, stratify_labels)
    """
    print(
        f"Setting up {n_splits}-fold cross-validation with source+label stratification..."
    )

    # Create stratification key combining source and label (same as preprocessing script)
    stratify_labels = np.array(
        [f"{source}_{label}" for source, label in zip(train_sources, y_train)]
    )

    print("Stratification groups found:")
    unique_groups, counts = np.unique(stratify_labels, return_counts=True)
    for group, count in zip(unique_groups, counts):
        print(f"  {group}: {count} samples")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Verify fold balance
    print(f"\nFold balance verification:")
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(y_train)), stratify_labels)
    ):
        fold_train_labels = y_train[train_idx]
        fold_val_labels = y_train[val_idx]
        fold_train_sources = train_sources[train_idx]
        fold_val_sources = train_sources[val_idx]

        # Calculate spam rates
        train_spam_rate = np.mean(fold_train_labels)
        val_spam_rate = np.mean(fold_val_labels)

        # Calculate source distributions
        train_source_dist = pd.Series(fold_train_sources).value_counts(normalize=True)
        val_source_dist = pd.Series(fold_val_sources).value_counts(normalize=True)

        print(f"Fold {fold_idx + 1}:")
        print(f"  Spam rates - Train: {train_spam_rate:.3f}, Val: {val_spam_rate:.3f}")

        # Show source distribution for validation fold
        val_dist_str = ", ".join(
            [f"{src}: {pct:.2f}" for src, pct in val_source_dist.items()]
        )
        print(f"  Val source dist: {val_dist_str}")

    return skf, stratify_labels


def define_xgb_param_distributions(use_gpu=True):
    """Define parameter distributions for RandomizedSearchCV (GPU/CPU aware)."""
    print("Defining XGBoost parameter distributions for RandomizedSearchCV...")

    # Base parameters
    base_params = {
        "objective": ["binary:logistic"],
        "eval_metric": ["aucpr"],
        "random_state": [42],
        "verbosity": [0],
    }

    # Add GPU or CPU specific parameters
    if use_gpu:
        print("Configuring for GPU training...")
        base_params.update(
            {
                "tree_method": ["gpu_hist"],
                "gpu_id": [0],
            }
        )
    else:
        print("Configuring for CPU training...")
        base_params.update(
            {
                "tree_method": ["hist"],  # Fast CPU method
                "n_jobs": [-1],  # Use all CPU cores
            }
        )

    # Parameter distributions (prioritize high-impact hyperparams)
    param_distributions = {
        **base_params,
        # n_estimators: sample a reasonable range (we will typically retrain with early stopping)
        "n_estimators": randint(100, 1001),
        # learning rate: log-uniform from 0.001 to 0.3
        "learning_rate": loguniform(1e-3, 3e-1),
        # tree structure
        "max_depth": randint(3, 12),  # 3..11
        "min_child_weight": randint(1, 10),  # 1..9
        # regularization: allow small to large values (log scale)
        "reg_alpha": loguniform(1e-8, 10),  # L1
        "reg_lambda": loguniform(1e-3, 10),  # L2
        # sampling
        "subsample": uniform(0.5, 0.5),  # 0.5 - 1.0
        "colsample_bytree": uniform(0.5, 0.5),  # 0.5 - 1.0
        "colsample_bylevel": uniform(0.5, 0.5),  # 0.5 - 1.0
        # additional
        "gamma": uniform(0, 1.0),  # 0 - 1
        "max_delta_step": [0, 1, 2],  # small discrete choices
        # optional: booster choice (uncomment if you want to try DART)
        # "booster": ["gbtree", "dart"]
    }

    # Print summary
    print("Parameter distributions prepared (key summary):")
    for k, v in param_distributions.items():
        if hasattr(v, "rvs"):
            print(f"  {k}: distribution {v}")
        else:
            print(f"  {k}: {v}")

    return param_distributions


def perform_randomized_search(
    X_train,
    y_train,
    stratify_labels,
    param_distributions,
    cv_folds,
    n_iter=100,
    scoring="average_precision",
    n_jobs=1,
    random_state=42,
    verbose=2,
):
    """
    Run RandomizedSearchCV with stratified folds.

    Args:
        X_train (csr_matrix): Train features.
        y_train (np.ndarray): Train labels.
        stratify_labels (np.ndarray): Stratification keys.
        param_distributions (dict): Distributions for search.
        cv_folds (StratifiedKFold): CV splits.
        n_iter (int): Random samples.
        scoring (str): Scoring metric (e.g., 'average_precision').
        n_jobs (int): Parallel jobs for CV.
        random_state (int): RNG seed.
        verbose (int): Verbosity.

    Returns:
        RandomizedSearchCV: Fitted search object.
    """
    print(f"Starting XGBoost randomized search with {scoring} scoring...")
    print(f"Testing {n_iter} random parameter combinations")
    print(
        f"Using {cv_folds.n_splits}-fold cross-validation with source+label stratification"
    )

    # Calculate expected time savings
    original_combinations = 1000  # Estimated for XGBoost grid search
    print(f"Original grid search would test ~{original_combinations} combinations")
    print(
        f"Randomized search will test {n_iter} combinations (~{100*n_iter/original_combinations:.1f}% of original)"
    )

    # Initialize base XGBoost classifier
    xgb_base = xgb.XGBClassifier()

    # Setup RandomizedSearchCV with our custom stratification
    randomized_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,  # Number of parameter settings that are sampled
        cv=list(cv_folds.split(X_train, stratify_labels)),  # Use our stratified splits
        scoring=scoring,
        n_jobs=n_jobs,  # Let XGBoost handle parallelism
        verbose=verbose,
        return_train_score=True,
        error_score="raise",
        random_state=random_state,
    )

    # Record start time
    start_time = time.time()
    print(
        f"Randomized search started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Fit randomized search
    randomized_search.fit(X_train, y_train)

    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    print(f"XGBoost randomized search completed in: {duration/60:.2f} minutes")
    print(f"Average time per parameter combination: {duration/n_iter:.2f} seconds")

    return randomized_search


def analyze_randomized_search_results(randomized_search):
    """Print best params/scores and return sorted results DataFrame."""
    print("Analyzing XGBoost randomized search results...")

    # Best parameters and score
    print(f"Best cross-validation score: {randomized_search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in randomized_search.best_params_.items():
        print(f"  {param}: {value}")

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(randomized_search.cv_results_)

    # Sort by test score
    results_df = results_df.sort_values("mean_test_score", ascending=False)

    # Display top 10 results
    print("\nTop 10 parameter combinations:")
    top_results = results_df[["mean_test_score", "std_test_score", "params"]].head(10)
    for idx, row in top_results.iterrows():
        print(
            f"Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f}) - {row['params']}"
        )

    # Analyze parameter importance by looking at top performers
    print("\nParameter analysis from top 20 combinations:")
    top_20 = results_df.head(20)

    # Extract parameter values from top performers
    param_analysis = {}
    for idx, row in top_20.iterrows():
        for param, value in row["params"].items():
            if param not in param_analysis:
                param_analysis[param] = []
            param_analysis[param].append(value)

    # Show most common values in top performers - FIXED VERSION
    for param, values in param_analysis.items():
        # Check if values are numeric (excluding None)
        numeric_values = [
            v for v in values if v is not None and isinstance(v, (int, float))
        ]

        if numeric_values and len(numeric_values) > 0:
            avg_val = np.mean(numeric_values)
            std_val = np.std(numeric_values)
            none_count = len([v for v in values if v is None])
            if none_count > 0:
                print(
                    f"  {param}: avg={avg_val:.3f} (+/- {std_val:.3f}), None: {none_count} times"
                )
            else:
                print(f"  {param}: avg={avg_val:.3f} (+/- {std_val:.3f})")
        else:
            from collections import Counter

            counter = Counter(values)
            most_common = counter.most_common(3)
            print(f"  {param}: {most_common}")

    return results_df


def retrain_with_early_stopping(
    best_params,
    X_train,
    y_train,
    train_sources,
    n_estimators=1000,
    early_stopping_rounds=50,
    val_size=0.1,
    random_state=42,
):
    """
    Retrain with best params using large n_estimators + early stopping.

    Args:
        best_params (dict): Best params from search.
        X_train, y_train: Training data.
        train_sources: Sources for stratified val split.
        n_estimators (int): Large count for early stopping.
        early_stopping_rounds (int): Patience.
        val_size (float): Validation fraction.
        random_state (int): RNG seed.

    Returns:
        tuple: (retrained_model, optimal_threshold, validation_results)
    """
    print("\n" + "=" * 60)
    print("RETRAINING WITH EARLY STOPPING")
    print("=" * 60)

    # Create validation split
    (
        X_train_split,
        X_val_split,
        y_train_split,
        y_val_split,
        train_sources_split,
        val_sources_split,
    ) = create_validation_split(
        X_train, y_train, train_sources, val_size=val_size, random_state=random_state
    )

    # Prepare parameters for retraining
    retrain_params = best_params.copy()
    retrain_params["n_estimators"] = n_estimators
    retrain_params["early_stopping_rounds"] = early_stopping_rounds
    retrain_params["eval_metric"] = "aucpr"  # For early stopping

    print(f"\nRetraining parameters:")
    for param, value in retrain_params.items():
        print(f"  {param}: {value}")

    # Create and train model with early stopping
    print(f"\nTraining with early stopping (patience={early_stopping_rounds})...")
    print(f"Training set shape: {X_train_split.shape}")
    print(f"Validation set shape: {X_val_split.shape}")
    print(f"Training labels shape: {y_train_split.shape}")
    print(f"Validation labels shape: {y_val_split.shape}")

    retrained_model = xgb.XGBClassifier(**retrain_params)

    retrained_model.fit(
        X_train_split,
        y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False,
    )

    print(f"Training stopped at iteration: {retrained_model.best_iteration}")
    print(f"Best validation aucpr: {retrained_model.best_score:.4f}")

    # Get validation predictions for threshold tuning
    y_val_proba = retrained_model.predict_proba(X_val_split)[:, 1]

    # Find optimal threshold using F1 score
    print("\nFinding F1-optimal threshold on validation set...")
    optimal_threshold, best_f1, thresholds_data = find_optimal_threshold(
        y_val_split, y_val_proba, metric="f1"
    )

    # Evaluate on validation set with optimal threshold
    y_val_pred_optimal = (y_val_proba >= optimal_threshold).astype(int)

    val_metrics_optimal = {
        "accuracy": accuracy_score(y_val_split, y_val_pred_optimal),
        "precision": precision_score(y_val_split, y_val_pred_optimal),
        "recall": recall_score(y_val_split, y_val_pred_optimal),
        "f1": f1_score(y_val_split, y_val_pred_optimal),
        "roc_auc": roc_auc_score(y_val_split, y_val_proba),
        "aucpr": average_precision_score(y_val_split, y_val_proba),
    }

    # Compare with default threshold (0.5)
    y_val_pred_default = (y_val_proba >= 0.5).astype(int)
    val_metrics_default = {
        "accuracy": accuracy_score(y_val_split, y_val_pred_default),
        "precision": precision_score(y_val_split, y_val_pred_default),
        "recall": recall_score(y_val_split, y_val_pred_default),
        "f1": f1_score(y_val_split, y_val_pred_default),
        "roc_auc": roc_auc_score(y_val_split, y_val_proba),
        "aucpr": average_precision_score(y_val_split, y_val_proba),
    }

    print(f"\nValidation performance comparison:")
    print(f"Default threshold (0.5):")
    for metric, value in val_metrics_default.items():
        print(f"  {metric.upper()}: {value:.4f}")

    print(f"\nOptimal threshold ({optimal_threshold:.4f}):")
    for metric, value in val_metrics_optimal.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Show improvement
    f1_improvement = val_metrics_optimal["f1"] - val_metrics_default["f1"]
    print(
        f"\nF1 improvement: {f1_improvement:+.4f} ({f1_improvement/val_metrics_default['f1']*100:+.1f}%)"
    )

    # Package validation results
    validation_results = {
        "optimal_threshold": optimal_threshold,
        "best_f1_score": best_f1,
        "validation_metrics_optimal": val_metrics_optimal,
        "validation_metrics_default": val_metrics_default,
        "f1_improvement": f1_improvement,
        "f1_improvement_percent": f1_improvement / val_metrics_default["f1"] * 100,
        "thresholds_analysis": thresholds_data,
        "early_stopping_iteration": retrained_model.best_iteration,
        "validation_aucpr": retrained_model.best_score,
        "val_size": val_size,
        "early_stopping_rounds": early_stopping_rounds,
    }

    return retrained_model, optimal_threshold, validation_results


def save_model_and_results(
    model,
    randomized_search,
    evaluation_results,
    metadata,
    importance_df,
    validation_results=None,
    optimal_threshold=0.5,
    output_dir="models/xgb",
    features_dir=None,
):
    """
    Save model, search, importance, plots, and a summary JSON.

    Args:
        model (xgb.XGBClassifier): Final model.
        randomized_search (RandomizedSearchCV): Fitted search.
        evaluation_results (dict): Metrics and artifacts.
        metadata (dict): Feature metadata.
        importance_df (pd.DataFrame): Feature importances.
        validation_results (dict|None): Early stopping details.
        optimal_threshold (float): Chosen threshold.
        output_dir (str): Directory to save.

    Returns:
        dict: Results summary content that was written to JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving XGBoost model and results to: {output_path}")

    # Save the trained model
    joblib.dump(model, output_path / "xgb_model_randomized.pkl")

    # Save randomized search results
    joblib.dump(randomized_search, output_path / "xgb_randomized_search.pkl")

    # Save feature importance
    importance_df.to_csv(output_path / "feature_importance_randomized.csv", index=False)

    # Compile all results
    results_summary = {
        "model_type": "XGBClassifier",
        "search_type": "RandomizedSearchCV",
        "n_iter": randomized_search.n_iter,
        "best_parameters": randomized_search.best_params_,
        "best_cv_score": randomized_search.best_score_,
        "cv_score_std": randomized_search.cv_results_["std_test_score"][
            randomized_search.best_index_
        ],
        "train_metrics": evaluation_results["train_metrics"],
        "test_metrics": evaluation_results["test_metrics"],
        "train_performance_by_source": evaluation_results[
            "train_performance_by_source"
        ],
        "test_performance_by_source": evaluation_results["test_performance_by_source"],
        "train_classification_report": evaluation_results[
            "train_classification_report"
        ],
        "test_classification_report": evaluation_results["test_classification_report"],
        "feature_metadata": metadata,
        "training_timestamp": datetime.now().isoformat(),
        "top_20_features": importance_df.head(20).to_dict("records"),
        "cross_validation": {
            "strategy": "StratifiedKFold with source+label stratification",
            "n_splits": 5,
            "scoring": "average_precision",  # Updated to AUCPR
        },
        "optimization": {
            "method": "RandomizedSearchCV + Early Stopping Retraining",
            "combinations_tested": randomized_search.n_iter,
            "random_state": 42,
            "gpu_used": "gpu_hist"
            in str(randomized_search.best_params_.get("tree_method", "cpu")),
        },
        "threshold_tuning": {
            "optimal_threshold": optimal_threshold,
            "tuning_metric": "f1_score",
            "validation_split_size": (
                validation_results.get("val_size", 0.1) if validation_results else 0.1
            ),
        },
        "early_stopping": validation_results if validation_results else None,
        "xgboost_specific": {
            "objective": randomized_search.best_params_.get(
                "objective", "binary:logistic"
            ),
            "eval_metric": randomized_search.best_params_.get("eval_metric", "aucpr"),
            "tree_method": randomized_search.best_params_.get("tree_method", "hist"),
            "importance_type": "gain-based (preferred) or weight-based (fallback)",
            "final_n_estimators": model.n_estimators,
            "early_stopping_iteration": (
                validation_results.get("early_stopping_iteration")
                if validation_results
                else None
            ),
        },
    }

    # Save results summary
    with open(output_path / "xgb_results_summary_randomized.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Save/copy required preprocessing artifacts for standalone inference
    try:
        if features_dir is not None:
            features_path = Path(features_dir)
            # Copy TF-IDF vectorizer
            tfidf_path = features_path / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                vec = joblib.load(tfidf_path)
                joblib.dump(vec, output_path / "tfidf_vectorizer.pkl")
            # Copy feature metadata
            meta_path = features_path / "feature_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as mf:
                    meta_json = json.load(mf)
                with open(output_path / "feature_metadata.json", "w") as wf:
                    json.dump(meta_json, wf, indent=2)
    except Exception as e:
        print(f"Warning: failed to copy preprocessing artifacts for inference: {e}")

    print("Saved files:")
    print("- xgb_model_randomized.pkl (trained XGBoost model)")
    print("- xgb_randomized_search.pkl (full randomized search object)")
    print("- feature_importance_randomized.csv (feature importance rankings)")
    print("- xgb_results_summary_randomized.json (comprehensive results)")
    print("- xgb_evaluation_randomized.png (evaluation plots)")

    return results_summary


def train_xgboost_model(
    features_dir="datasets/features",
    output_dir="models/xgb_aucpr",
    cv_folds=5,
    n_iter=100,
    scoring="average_precision",
    n_estimators_retrain=2000,
    early_stopping_rounds=50,
    val_size=0.1,
    random_state=42,
    verbose=2,
):
    """
    Full training pipeline using AUCPR search, early stopping, threshold tuning.

    Args:
        features_dir (str): Feature directory.
        output_dir (str): Output directory.
        cv_folds (int): Number of CV folds.
        n_iter (int): Randomized search iterations.
        scoring (str): Scoring metric for search.
        n_estimators_retrain (int): Estimators for early stopping retrain.
        early_stopping_rounds (int): Patience.
        val_size (float): Validation split fraction.
        random_state (int): RNG seed.
        verbose (int): Verbosity for search.

    Returns:
        tuple: (final_model, results_summary)
    """
    print(
        "Starting XGBoost training pipeline with AUCPR optimization and early stopping"
    )
    print("=" * 80)

    # Check GPU availability
    use_gpu = check_gpu_availability()

    # Load features
    X_train, X_test, y_train, y_test, train_sources, test_sources, metadata = (
        load_engineered_features(features_dir)
    )

    # Display class distribution information
    train_neg_count = np.sum(y_train == 0)
    train_pos_count = np.sum(y_train == 1)
    train_spam_rate = train_pos_count / len(y_train) * 100
    test_neg_count = np.sum(y_test == 0)
    test_pos_count = np.sum(y_test == 1)
    test_spam_rate = test_pos_count / len(y_test) * 100

    print(f"\nDataset class distribution:")
    print(
        f"Training: {train_neg_count:,} negative, {train_pos_count:,} positive ({train_spam_rate:.1f}% spam)"
    )
    print(
        f"Test: {test_neg_count:,} negative, {test_pos_count:,} positive ({test_spam_rate:.1f}% spam)"
    )
    print(f"Class ratio (neg/pos): {train_neg_count/train_pos_count:.3f}")

    # Setup stratified cross-validation with source+label stratification
    cv, stratify_labels = setup_stratified_kfold_cv(
        y_train, train_sources, n_splits=cv_folds, random_state=random_state
    )

    # Define parameter distributions
    param_distributions = define_xgb_param_distributions(use_gpu=use_gpu)

    # Perform randomized search with AUCPR
    print(f"\nStep 1: Hyperparameter search using {scoring} (AUCPR)...")
    randomized_search = perform_randomized_search(
        X_train,
        y_train,
        stratify_labels,
        param_distributions,
        cv,
        n_iter,
        scoring,
        n_jobs=1,
        random_state=random_state,
        verbose=verbose,
    )

    # Analyze randomized search results
    results_df = analyze_randomized_search_results(randomized_search)

    # Step 2: Retrain with early stopping
    print(f"\nStep 2: Retraining with early stopping...")
    final_model, optimal_threshold, validation_results = retrain_with_early_stopping(
        randomized_search.best_params_,
        X_train,
        y_train,
        train_sources,
        n_estimators=n_estimators_retrain,
        early_stopping_rounds=early_stopping_rounds,
        val_size=val_size,
        random_state=random_state,
    )

    # Step 3: Evaluate final model with optimal threshold
    print(f"\nStep 3: Final evaluation with optimal threshold...")

    # Get test predictions
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

    # Calculate test metrics with optimal threshold
    test_metrics_optimal = {
        "accuracy": accuracy_score(y_test, y_test_pred_optimal),
        "precision": precision_score(y_test, y_test_pred_optimal),
        "recall": recall_score(y_test, y_test_pred_optimal),
        "f1": f1_score(y_test, y_test_pred_optimal),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "aucpr": average_precision_score(y_test, y_test_proba),
    }

    # Compare test performance with default vs optimal threshold
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    test_metrics_default = {
        "accuracy": accuracy_score(y_test, y_test_pred_default),
        "precision": precision_score(y_test, y_test_pred_default),
        "recall": recall_score(y_test, y_test_pred_default),
        "f1": f1_score(y_test, y_test_pred_default),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "aucpr": average_precision_score(y_test, y_test_proba),
    }

    print(f"\nTest performance comparison:")
    print(f"Default threshold (0.5):")
    for metric, value in test_metrics_default.items():
        print(f"  {metric.upper()}: {value:.4f}")

    print(f"\nOptimal threshold ({optimal_threshold:.4f}):")
    for metric, value in test_metrics_optimal.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Show test improvement
    test_f1_improvement = test_metrics_optimal["f1"] - test_metrics_default["f1"]
    print(
        f"\nTest F1 improvement: {test_f1_improvement:+.4f} ({test_f1_improvement/test_metrics_default['f1']*100:+.1f}%)"
    )

    # Standard evaluation (for comparison and plots) - using optimal threshold
    evaluation_results = evaluate_model(
        final_model,
        X_train,
        X_test,
        y_train,
        y_test,
        train_sources,
        test_sources,
        threshold=optimal_threshold,
    )

    # Add optimal threshold results to evaluation
    evaluation_results["test_metrics_optimal_threshold"] = test_metrics_optimal
    evaluation_results["test_metrics_default_threshold"] = test_metrics_default
    evaluation_results["test_f1_improvement"] = test_f1_improvement
    evaluation_results["test_f1_improvement_percent"] = (
        test_f1_improvement / test_metrics_default["f1"] * 100
    )
    evaluation_results["optimal_threshold"] = optimal_threshold

    # Analyze feature importance
    importance_df = analyze_feature_importance(final_model, metadata)

    # Create evaluation plots
    create_evaluation_plots(evaluation_results, importance_df, y_test, output_dir)

    # Save everything
    results_summary = save_model_and_results(
        final_model,
        randomized_search,
        evaluation_results,
        metadata,
        importance_df,
        validation_results,
        optimal_threshold,
        output_dir,
        features_dir,
    )

    print("\n" + "=" * 80)
    print("XGBoost training with AUCPR optimization completed successfully!")
    print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled (using CPU)'}")
    print(f"Combinations tested: {randomized_search.n_iter}")
    print(
        f"Best CV AUCPR: {randomized_search.best_score_:.4f} (+/- {results_summary['cv_score_std']*2:.4f})"
    )
    print(
        f"Early stopping at iteration: {validation_results['early_stopping_iteration']}"
    )
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Test F1 Score (optimal threshold): {test_metrics_optimal['f1']:.4f}")
    print(f"Test AUCPR: {test_metrics_optimal['aucpr']:.4f}")
    print("Cross-validation used source+label stratification for balanced folds")

    return final_model, results_summary
