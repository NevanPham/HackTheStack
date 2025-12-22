import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
import json
from collections import Counter


def load_preprocessed_data(data_path="datasets/preprocessed"):
    """
    Load preprocessed train/test CSVs for feature engineering.

    Args:
        data_path (str): Directory with train.csv and test.csv.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
    """
    data_dir = Path(data_path)

    print("Loading preprocessed data...")
    train_data = pd.read_csv(data_dir / "train.csv")
    test_data = pd.read_csv(data_dir / "test.csv")

    print(f"Training data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")

    return train_data, test_data


def extract_text_statistics(texts):
    """
    Basic text stats: chars, words, sentences, avg word length.

    Args:
        texts (list[str]): Input texts.

    Returns:
        tuple[np.ndarray, list[str]]: (features, feature_names)
    """
    print("Extracting text statistics...")

    features = []
    for text in texts:
        text = str(text)

        # Length features
        char_count = len(text)
        word_count = len(text.split())

        # Sentence features (approximate)
        sentence_count = len(re.split(r"[.!?]+", text))
        avg_word_length = (
            np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
        )

        features.append([char_count, word_count, sentence_count, avg_word_length])

    feature_names = ["char_count", "word_count", "sentence_count", "avg_word_length"]
    return np.array(features), feature_names


def extract_special_character_features(texts):
    """
    Special-character/token features relevant to spam.

    Args:
        texts (list[str]): Input texts.

    Returns:
        tuple[np.ndarray, list[str]]: (features, feature_names)
    """
    print("Extracting special character features...")

    features = []
    for text in texts:
        text = str(text)
        text_length = len(text) if len(text) > 0 else 1

        # Special characters
        exclamation_count = text.count("!")
        question_count = text.count("?")
        dot_count = text.count(".")
        comma_count = text.count(",")

        # Ratios (normalized by text length)
        exclamation_ratio = exclamation_count / text_length
        question_ratio = question_count / text_length

        # Capital letters
        upper_count = sum(1 for c in text if c.isupper())
        upper_ratio = upper_count / text_length

        # Special tokens from preprocessing (these are crucial spam indicators)
        url_count = text.count("<URL>")
        email_count = text.count("<EMAIL>")
        phone_count = text.count("<PHONE>")
        money_count = text.count("<MONEY>")
        number_count = text.count("<NUMBER>")
        excited_count = text.count("<EXCITED>")
        question_token_count = text.count("<QUESTION>")
        dots_count = text.count("<DOTS>")

        features.append(
            [
                exclamation_count,
                question_count,
                dot_count,
                comma_count,
                exclamation_ratio,
                question_ratio,
                upper_ratio,
                url_count,
                email_count,
                phone_count,
                money_count,
                number_count,
                excited_count,
                question_token_count,
                dots_count,
            ]
        )

    feature_names = [
        "exclamation_count",
        "question_count",
        "dot_count",
        "comma_count",
        "exclamation_ratio",
        "question_ratio",
        "upper_ratio",
        "url_count",
        "email_count",
        "phone_count",
        "money_count",
        "number_count",
        "excited_count",
        "question_token_count",
        "dots_count",
    ]

    return np.array(features), feature_names


def extract_source_features(sources, all_sources):
    """One-hot encode sources in fixed order."""
    print("Extracting source features...")

    source_features = []
    for source in sources:
        source_vector = [1 if source == s else 0 for s in all_sources]
        source_features.append(source_vector)

    feature_names = [f"source_{source}" for source in all_sources]
    return np.array(source_features), feature_names


def create_tfidf_features(
    train_texts, test_texts, max_features=5000, ngram_range=(1, 2)
):
    """
    Create TF-IDF with unigrams+bigrams; preserve special tokens.

    Args:
        train_texts (list[str]): Train texts.
        test_texts (list[str]): Test texts.
        max_features (int): Vocab size.
        ngram_range (tuple[int,int]): N-gram range.

    Returns:
        tuple: (train_tfidf, test_tfidf, tfidf_vectorizer)
    """
    print(
        f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})..."
    )

    # Initialize TF-IDF vectorizer with improved settings
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=None,  # Keep all words including potential spam indicators
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"\b\w+\b|<\w+>",  # Include special tokens like <URL>, <EMAIL>
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True,  # Apply sublinear tf scaling (helps with very frequent terms)
    )

    # Fit on training data and transform both train and test
    print("Fitting TF-IDF on training data...")
    train_tfidf = tfidf.fit_transform(train_texts)

    print("Transforming test data...")
    test_tfidf = tfidf.transform(test_texts)

    print(f"TF-IDF matrix shape - Train: {train_tfidf.shape}, Test: {test_tfidf.shape}")
    print(
        f"TF-IDF sparsity - Train: {train_tfidf.nnz / np.prod(train_tfidf.shape):.4f}"
    )
    print(f"TF-IDF sparsity - Test: {test_tfidf.nnz / np.prod(test_tfidf.shape):.4f}")

    return train_tfidf, test_tfidf, tfidf


def combine_features(
    tfidf_features, statistical_features, special_char_features, source_features
):
    """
    Horizontally stack TF-IDF with numeric features (sparse result).

    Args:
        tfidf_features (csr_matrix): TF-IDF features.
        statistical_features (np.ndarray): Stats.
        special_char_features (np.ndarray): Special-char features.
        source_features (np.ndarray): One-hot sources.

    Returns:
        csr_matrix: Combined features.
    """
    print("Combining all features...")

    # Combine numerical features (no scaling for tree models)
    numerical_features = np.hstack(
        [statistical_features, special_char_features, source_features]
    )

    print("Note: Numerical features NOT scaled (appropriate for tree-based models)")

    # Convert numerical features to sparse matrix
    numerical_sparse = sparse.csr_matrix(numerical_features)

    # Combine TF-IDF and numerical features
    combined_features = sparse.hstack([tfidf_features, numerical_sparse])

    print(f"Combined feature matrix shape: {combined_features.shape}")
    print(
        f"Combined sparsity: {combined_features.nnz / np.prod(combined_features.shape):.4f}"
    )

    return combined_features


def create_feature_names(tfidf_vectorizer, stat_names, special_names, source_names):
    """Build combined feature name list with 'tfidf_' prefix for TF-IDF."""
    tfidf_names = [
        f"tfidf_{feature}" for feature in tfidf_vectorizer.get_feature_names_out()
    ]
    all_names = tfidf_names + stat_names + special_names + source_names
    return all_names


def save_features_and_metadata(
    features_dict, metadata_dict, output_dir="datasets/features"
):
    """Persist features (npz/pkl) and metadata.json to output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("Saving features and metadata...")

    # Save features
    for name, data in features_dict.items():
        if sparse.issparse(data):
            sparse.save_npz(output_path / f"{name}.npz", data)

        else:

            joblib.dump(data, output_path / f"{name}.pkl")

    # Save metadata as JSON
    with open(output_path / "feature_metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2, default=str)

    print(f"Features saved to: {output_path}")
    return output_path


def engineer_ml_features(
    data_path="datasets/preprocessed",
    output_dir="preprocessed/xgboost",
    max_tfidf_features=5000,
    ngram_range=(1, 2),
):
    """
    End-to-end feature engineering for XGBoost.

    Args:
        data_path (str): Preprocessed data directory.
        output_dir (str): Output directory for features.
        max_tfidf_features (int): TF-IDF vocab size.
        ngram_range (tuple[int,int]): N-gram range.

    Returns:
        tuple[Path, dict]: (output_path, metadata)
    """
    print("Starting ML feature engineering pipeline")
    print("=" * 60)

    # Load data
    train_data, test_data = load_preprocessed_data(data_path)

    # Get unique sources for one-hot encoding
    all_sources = sorted(train_data["source"].unique())
    print(f"Sources found: {all_sources}")

    # Extract text statistics
    train_stat_features, stat_names = extract_text_statistics(
        train_data["cleaned_text"]
    )
    test_stat_features, _ = extract_text_statistics(test_data["cleaned_text"])

    # Extract special character features
    train_special_features, special_names = extract_special_character_features(
        train_data["cleaned_text"]
    )
    test_special_features, _ = extract_special_character_features(
        test_data["cleaned_text"]
    )

    # Extract source features
    train_source_features, source_names = extract_source_features(
        train_data["source"], all_sources
    )
    test_source_features, _ = extract_source_features(test_data["source"], all_sources)

    # Create TF-IDF features
    train_tfidf, test_tfidf, tfidf_vectorizer = create_tfidf_features(
        train_data["cleaned_text"],
        test_data["cleaned_text"],
        max_features=max_tfidf_features,
        ngram_range=ngram_range,
    )

    # Combine all features (no scaling for tree models)
    train_combined = combine_features(
        train_tfidf, train_stat_features, train_special_features, train_source_features
    )
    test_combined = combine_features(
        test_tfidf, test_stat_features, test_special_features, test_source_features
    )

    # Create feature names
    feature_names = create_feature_names(
        tfidf_vectorizer, stat_names, special_names, source_names
    )

    # Prepare data to save
    features_dict = {
        "X_train": train_combined,
        "X_test": test_combined,
        "y_train": train_data["label"].values,
        "y_test": test_data["label"].values,
        "train_sources": train_data["source"].values,
        "test_sources": test_data["source"].values,
        "tfidf_vectorizer": tfidf_vectorizer,
    }

    # Prepare metadata
    metadata_dict = {
        "n_train_samples": len(train_data),
        "n_test_samples": len(test_data),
        "n_features": len(feature_names),
        "n_tfidf_features": train_tfidf.shape[1],
        "n_statistical_features": len(stat_names),
        "n_special_char_features": len(special_names),
        "n_source_features": len(source_names),
        "feature_names": feature_names,
        "statistical_feature_names": stat_names,
        "special_char_feature_names": special_names,
        "source_feature_names": source_names,
        "max_tfidf_features": max_tfidf_features,
        "ngram_range": ngram_range,
        "sources": all_sources,
        "train_spam_rate": train_data["label"].mean(),
        "test_spam_rate": test_data["label"].mean(),
        "tfidf_config": {
            "stop_words": None,
            "token_pattern": r"\b\w+\b|<\w+>",
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
        "scaling_applied": False,
        "notes": "Features optimized for tree-based models (RF, XGB). No scaling applied. Special tokens preserved.",
    }

    # Save everything
    output_path = save_features_and_metadata(features_dict, metadata_dict, output_dir)

    # Print summary
    print("=" * 60)
    print("Feature Engineering Summary:")
    print(f"Total features: {len(feature_names):,}")
    print(f"  - TF-IDF features: {train_tfidf.shape[1]:,}")
    print(f"  - Statistical features: {len(stat_names)}")
    print(f"  - Special character features: {len(special_names)}")
    print(f"  - Source features: {len(source_names)}")
    print(f"Training samples: {len(train_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Matrix sparsity: {train_combined.nnz / np.prod(train_combined.shape):.4f}")
    print("Optimizations applied:")
    print("  - No stop word removal (preserves spam indicators)")
    print("  - Special tokens preserved in TF-IDF")
    print("  - No numerical feature scaling (tree-model appropriate)")
    print("  - Sublinear TF scaling applied")
    print(f"Features saved to: {output_path}")

    return output_path, metadata_dict
